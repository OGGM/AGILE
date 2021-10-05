""" Handling of the local glacier map and masks. Adapted from oggm.core.gis.
"""
# Built ins
import os
import logging
import warnings
from distutils.version import LooseVersion
from functools import partial

# External libs
import numpy as np
import shapely.ops
import pandas as pd
import xarray as xr
import shapely.geometry as shpg
import scipy.signal
from scipy.ndimage.measurements import label
from scipy.ndimage import binary_erosion
from scipy.ndimage.morphology import distance_transform_edt
from scipy.interpolate import griddata
from scipy import optimize as optimization

# Optional libs
try:
    import salem
    from salem.gis import transform_proj
except ImportError:
    pass
try:
    import pyproj
except ImportError:
    pass
try:
    import geopandas as gpd
except ImportError:
    pass
try:
    import skimage.draw as skdraw
except ImportError:
    pass
try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.mask import mask as riomask
    try:
        # rasterio V > 1.0
        from rasterio.merge import merge as merge_tool
    except ImportError:
        from rasterio.tools.merge import merge as merge_tool
except ImportError:
    pass

# Locals
from oggm import entity_task, utils
import oggm.cfg as cfg
from oggm.exceptions import (InvalidParamsError, InvalidGeometryError,
                             InvalidDEMError, GeometryError,
                             InvalidWorkflowError)
from oggm.utils import (tuple2int, get_topo_file, is_dem_source_available,
                        nicenumber, ncDataset, tolist)


# Module logger
log = logging.getLogger(__name__)

# Needed later
label_struct = np.ones((3, 3))

from oggm.core.gis import glacier_grid_params


def _parse_source_text():
    fp = os.path.join(os.path.abspath(os.path.dirname(cfg.__file__)),
                      'data', 'dem_sources.txt')

    out = dict()
    cur_key = None
    with open(fp, 'r', encoding='utf-8') as fr:
        this_text = []
        for l in fr.readlines():
            l = l.strip()
            if l and (l[0] == '[' and l[-1] == ']'):
                if cur_key:
                    out[cur_key] = '\n'.join(this_text)
                this_text = []
                cur_key = l.strip('[]')
                continue
            this_text.append(l)
    out[cur_key] = '\n'.join(this_text)
    return out


DEM_SOURCE_INFO = _parse_source_text()


@entity_task(log, writes=['glacier_grid', 'dem', 'outlines'])
def define_glacier_region(gdir, gdir_ref=None, entity=None, source=None):
    """Very first task after initialization: define the glacier's local grid.
    Defines the local projection (Transverse Mercator), centered on the
    glacier. There is some options to set the resolution of the local grid.
    It can be adapted depending on the size of the glacier with::
        dx (m) = d1 * AREA (km) + d2 ; clipped to dmax
    or be set to a fixed value. See ``params.cfg`` for setting these options.
    Default values of the adapted mode lead to a resolution of 50 m for
    Hintereisferner, which is approx. 8 km2 large.
    After defining the grid, the topography and the outlines of the glacier
    are transformed into the local projection. The default interpolation for
    the topography is `cubic`.
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    entity : geopandas.GeoSeries
        the glacier geometry to process - DEPRECATED. It is now ignored
    source : str or list of str, optional
        If you want to force the use of a certain DEM source. Available are:
          - 'USER' : file set in cfg.PATHS['dem_file']
          - 'SRTM' : http://srtm.csi.cgiar.org/
          - 'GIMP' : https://bpcrc.osu.edu/gdg/data/gimpdem
          - 'RAMP' : http://nsidc.org/data/docs/daac/nsidc0082_ramp_dem.gd.html
          - 'REMA' : https://www.pgc.umn.edu/data/rema/
          - 'DEM3' : http://viewfinderpanoramas.org/
          - 'ASTER' : https://lpdaac.usgs.gov/products/astgtmv003/
          - 'TANDEM' : https://geoservice.dlr.de/web/dataguide/tdm90/
          - 'ARCTICDEM' : https://www.pgc.umn.edu/data/arcticdem/
          - 'AW3D30' : https://www.eorc.jaxa.jp/ALOS/en/aw3d30
          - 'MAPZEN' : https://registry.opendata.aws/terrain-tiles/
          - 'ALASKA' : https://www.the-cryosphere.net/8/503/2014/
          - 'COPDEM' : Copernicus DEM GLO-90 https://bit.ly/2T98qqs
          - 'NASADEM': https://lpdaac.usgs.gov/products/nasadem_hgtv001/
    """

    if gdir_ref == None:
        gdir_ref = gdir

    utm_proj, nx, ny, ulx, uly, dx = glacier_grid_params(gdir_ref)

    # Back to lon, lat for DEM download/preparation
    tmp_grid = salem.Grid(proj=utm_proj, nxny=(nx, ny), x0y0=(ulx, uly),
                          dxdy=(dx, -dx), pixel_ref='corner')
    minlon, maxlon, minlat, maxlat = tmp_grid.extent_in_crs(crs=salem.wgs84)

    # Open DEM
    # We test DEM availability for glacier only (maps can grow big)
    if not is_dem_source_available(source, *gdir.extent_ll):
        log.warning('Source: {} may not be available for glacier {} with '
                    'border {}'.format(source, gdir.rgi_id,
                                       cfg.PARAMS['border']))
    dem_list, dem_source = get_topo_file((minlon, maxlon), (minlat, maxlat),
                                         rgi_id=gdir.rgi_id,
                                         dx_meter=dx,
                                         source=source)
    log.debug('(%s) DEM source: %s', gdir.rgi_id, dem_source)
    log.debug('(%s) N DEM Files: %s', gdir.rgi_id, len(dem_list))

    # Decide how to tag nodata
    def _get_nodata(rio_ds):
        nodata = rio_ds[0].meta.get('nodata', None)
        if nodata is None:
            # badly tagged geotiffs, let's do it ourselves
            nodata = -32767 if source == 'TANDEM' else -9999
        return nodata

    # A glacier area can cover more than one tile:
    if len(dem_list) == 1:
        dem_dss = [rasterio.open(dem_list[0])]  # if one tile, just open it
        dem_data = rasterio.band(dem_dss[0], 1)
        if LooseVersion(rasterio.__version__) >= LooseVersion('1.0'):
            src_transform = dem_dss[0].transform
        else:
            src_transform = dem_dss[0].affine
        nodata = _get_nodata(dem_dss)
    else:
        dem_dss = [rasterio.open(s) for s in dem_list]  # list of rasters
        nodata = _get_nodata(dem_dss)
        dem_data, src_transform = merge_tool(dem_dss, nodata=nodata)  # merge

    # Use Grid properties to create a transform (see rasterio cookbook)
    dst_transform = rasterio.transform.from_origin(
        ulx, uly, dx, dx  # sign change (2nd dx) is done by rasterio.transform
    )

    # Set up profile for writing output
    profile = dem_dss[0].profile
    profile.update({
        'crs': utm_proj.srs,
        'transform': dst_transform,
        'nodata': nodata,
        'width': nx,
        'height': ny,
        'driver': 'GTiff'
    })

    # Could be extended so that the cfg file takes all Resampling.* methods
    if cfg.PARAMS['topo_interp'] == 'bilinear':
        resampling = Resampling.bilinear
    elif cfg.PARAMS['topo_interp'] == 'cubic':
        resampling = Resampling.cubic
    else:
        raise InvalidParamsError('{} interpolation not understood'
                                 .format(cfg.PARAMS['topo_interp']))

    dem_reproj = gdir.get_filepath('dem')
    profile.pop('blockxsize', None)
    profile.pop('blockysize', None)
    profile.pop('compress', None)
    with rasterio.open(dem_reproj, 'w', **profile) as dest:
        dst_array = np.empty((ny, nx), dtype=dem_dss[0].dtypes[0])
        reproject(
            # Source parameters
            source=dem_data,
            src_crs=dem_dss[0].crs,
            src_transform=src_transform,
            src_nodata=nodata,
            # Destination parameters
            destination=dst_array,
            dst_transform=dst_transform,
            dst_crs=utm_proj.srs,
            dst_nodata=nodata,
            # Configuration
            resampling=resampling)
        dest.write(dst_array, 1)

    for dem_ds in dem_dss:
        dem_ds.close()

    # Glacier grid
    x0y0 = (ulx+dx/2, uly-dx/2)  # To pixel center coordinates
    glacier_grid = salem.Grid(proj=utm_proj, nxny=(nx, ny), dxdy=(dx, -dx),
                              x0y0=x0y0)
    glacier_grid.to_json(gdir.get_filepath('glacier_grid'))

    # Write DEM source info
    gdir.add_to_diagnostics('dem_source', dem_source)
    source_txt = DEM_SOURCE_INFO.get(dem_source, dem_source)
    with open(gdir.get_filepath('dem_source'), 'w') as fw:
        fw.write(source_txt)
        fw.write('\n\n')
        fw.write('# Data files\n\n')
        for fname in dem_list:
            fw.write('{}\n'.format(os.path.basename(fname)))