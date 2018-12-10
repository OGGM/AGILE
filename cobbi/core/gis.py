import numpy as np
import json
import rasterio
from rasterio.warp import reproject, Resampling
try:
    # rasterio V > 1.0
    from rasterio.merge import merge as merge_tool
except ImportError:
    from rasterio.tools.merge import merge as merge_tool
import salem
import pyproj
from distutils.version import LooseVersion
from oggm import cfg, entity_task
import logging
from oggm.utils import get_demo_file, get_topo_file
from cobbi.core.utils import NonRGIGlacierDirectory

# -------------------------------
# Further initialization / extended import tasks
# Module logger
log = logging.getLogger(__name__)
with open(get_demo_file('dem_sources.json'), 'r') as fr:
    DEM_SOURCE_INFO = json.loads(fr.read())
# -------------------------------


# @entity_task(log, writes=['glacier_grid', 'dem', 'dem_source'])
def define_nonrgi_glacier_region(gdir:NonRGIGlacierDirectory):
    """
    Very first task: define the glacier's local grid.

    Defines the local projection (Transverse Mercator), centered on the
    glacier. The resolution of the local grid is dx.

    After defining the grid, the topography is transformed into the local
    projection. The default interpolation for the topography is `cubic`.

    Parameters
    ----------
    gdir : :py:class:`oggm.NonRGIGlacierDirectory`
        where to write the data
    dx : float
        grid spacing
    """
    xx, yy = gdir.extent_ll
    dx = gdir.case.dx

    # Make a local glacier map
    proj_params = dict(name='tmerc', lat_0=0., lon_0=gdir.cenlon,
                       k=0.9996, x_0=0, y_0=0, datum='WGS84')
    proj4_str = "+proj={name} +lat_0={lat_0} +lon_0={lon_0} +k={k} " \
                "+x_0={x_0} +y_0={y_0} +datum={datum}".format(**proj_params)
    # proj_in = pyproj.Proj("+init=EPSG:4326", preserve_units=True)
    proj_out = pyproj.Proj(proj4_str, preserve_units=True)

    merc_xx, merc_yy = salem.transform_proj(salem.wgs84, proj_out, xx, yy)

    # Corners, incl. a buffer of N pix
    ulx = np.min(merc_xx)
    lrx = np.max(merc_xx)
    uly = np.max(merc_yy)
    lry = np.min(merc_yy)

    # n pixels
    nx = np.int((lrx - ulx) / dx)
    ny = np.int((uly - lry) / dx)

    # Back to lon, lat for DEM download/preparation
    tmp_grid = salem.Grid(proj=proj_out, nxny=(nx, ny), x0y0=(ulx, uly),
                          dxdy=(dx, -dx), pixel_ref='corner')
    minlon, maxlon, minlat, maxlat = tmp_grid.extent_in_crs(crs=salem.wgs84)

    dem_list, dem_source = get_topo_file((minlon, maxlon), (minlat, maxlat),
                                         rgi_region=None,
                                         rgi_subregion=None,
                                         source='DEM3')
    log.debug('(%s) DEM source: %s', gdir.name, dem_source)

    # A glacier area can cover more than one tile:
    if len(dem_list) == 1:
        dem_dss = [rasterio.open(dem_list[0])]  # if one tile, just open it
        dem_data = rasterio.band(dem_dss[0], 1)
        if LooseVersion(rasterio.__version__) >= LooseVersion('1.0'):
            src_transform = dem_dss[0].transform
        else:
            src_transform = dem_dss[0].affine
    else:
        dem_dss = [rasterio.open(s) for s in dem_list]  # list of rasters
        dem_data, src_transform = merge_tool(dem_dss)  # merged rasters

    # Use Grid properties to create a transform (see rasterio cookbook)
    dst_transform = rasterio.transform.from_origin(
        ulx, uly, dx, dx  # sign change (2nd dx) is done by rasterio.transform
    )

    # Set up profile for writing output
    profile = dem_dss[0].profile
    profile.update({
        'crs': proj4_str,
        'transform': dst_transform,
        'width': nx,
        'height': ny
    })

    # Could be extended so that the cfg file takes all Resampling.* methods
    if cfg.PARAMS['topo_interp'] == 'bilinear':
        resampling = Resampling.bilinear
    elif cfg.PARAMS['topo_interp'] == 'cubic':
        resampling = Resampling.cubic
    else:
        raise ValueError('{} interpolation not understood'
                         .format(cfg.PARAMS['topo_interp']))

    dem_reproj = gdir.get_filepath('dem')

    with rasterio.open(dem_reproj, 'w', **profile) as dest:
        dst_array = np.empty((ny, nx), dtype=dem_dss[0].dtypes[0])
        reproject(
            # Source parameters
            source=dem_data,
            src_crs=dem_dss[0].crs,
            src_transform=src_transform,
            # Destination parameters
            destination=dst_array,
            dst_transform=dst_transform,
            dst_crs=proj4_str,
            # Configuration
            resampling=resampling)

        # TODO: ugly
        if gdir.case.name == 'Borden Peninsula':
            print('Anti icepatch used')
            dst_array[32, 27] = gdir.case.ela_h - 5
            dst_array[:2, -4:] = gdir.case.ela_h - 5
        if gdir.case.name == 'Borden Peninsula HR':
            print('Anti icepatch HR used')
            dst_array[-21:-16, 32:38] = gdir.case.ela_h - 5
            dst_array[-8:-2, 88:98] = gdir.case.ela_h - 5
            dst_array[:-109, 120:] = gdir.case.ela_h - 5
        dest.write(dst_array, 1)

    for dem_ds in dem_dss:
        dem_ds.close()

    # Glacier grid
    x0y0 = (ulx+dx/2, uly-dx/2)  # To pixel center coordinates
    glacier_grid = salem.Grid(proj=proj_out, nxny=(nx, ny),  dxdy=(dx, -dx),
                              x0y0=x0y0)
    glacier_grid.to_json(gdir.get_filepath('glacier_grid'))

    # Write DEM source info
    source_txt = DEM_SOURCE_INFO.get(dem_source, dem_source)
    with open(gdir.get_filepath('dem_source'), 'w') as fw:
        fw.write(source_txt)




def smooth_dem_borders(gdir, border_h=0., px_count=10,
                       filesuffix='_smooth_border'):
    """
    Manipulates the DEM and linearly interpolates the desired number of
    pixels from the border from the desired border height to the height at
    the inner preserved part of the DEM

    Deprecated

    Parameters
    ----------
    gdir : :py:class:`oggm.NonRGIGlacierDirectory`
        where to get the dem and obtain the data
    border_h : float
        height of the outside border
    px_count : int
        Number of pixels which should be linearly adapted
    filesuffix : string
        filesuffix with which the new file should be stored
    """
    with rasterio.open(gdir.get_filepath('dem')) as src:
        pixel_vals = src.read(1)
        profile = src.profile

    if px_count > 0:
        # TODO: fix bug for px_count = 1
        pixel_vals[:, :px_count] = [np.linspace(border_h, i, px_count) for i in
                                    pixel_vals[:, px_count]]

        pixel_vals[:, -px_count:] = [np.linspace(i, border_h, px_count) for i in
                                     pixel_vals[:, -px_count]]

        pixel_vals[:px_count, :] = np.array([np.linspace(border_h, i, px_count)
                                             for i in
                                             pixel_vals[px_count, :]]).T

        pixel_vals[-px_count:, :] = np.array([np.linspace(i, border_h, px_count)
                                              for i in
                                              pixel_vals[-px_count, :]]).T

    with rasterio.open(gdir.get_filepath('dem', filesuffix=filesuffix),
                       'w', **profile) as dst:
        dst.write(pixel_vals, 1)