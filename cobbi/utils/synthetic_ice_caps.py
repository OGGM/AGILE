"""

"""
import os
import json
import shutil
import salem
import numpy as np
import pyproj
import logging
import pandas as pd
from distutils.version import LooseVersion
from osgeo import gdal

import rasterio
from rasterio.warp import reproject, Resampling
try:
    # rasterio V > 1.0
    from rasterio.merge import merge as merge_tool
except ImportError:
    from rasterio.tools.merge import merge as merge_tool

from oggm import cfg, entity_task
from oggm.utils import GlacierDirectory, mkdir, get_demo_file, get_topo_file

# Module logger
log = logging.getLogger(__name__)

with open(get_demo_file('dem_sources.json'), 'r') as fr:
    DEM_SOURCE_INFO = json.loads(fr.read())


class NonRGIGlacierDirectory(GlacierDirectory):
    """GlacierDirectory for synthetic cases and glaciers, which are not
    registered with RGI"""

    def __init__(self, entity, base_dir=None, reset=False):
        """

        Parameters
        ----------
        entity : dictionary with entries for
                 min_x, max_x, min_y, max_y, name, (glacier_type,
                 terminus_type)
        base_dir : str
            path to the directory where to open the directory.
            Defaults to `cfg.PATHS['working_dir'] + /per_glacier/`
        reset : bool, default=False
            empties the directory at construction (careful!)
        """

        if base_dir is None:
            if not cfg.PATHS.get('working_dir', None):
                raise ValueError("Need a valid PATHS['working_dir']!")
            base_dir = os.path.join(cfg.PATHS['working_dir'], 'per_glacier')

        self.extent_ll = [[entity['min_x'], entity['max_x']],
                          [entity['min_y'], entity['max_y']]]

        # TODO
        self.rgi_id = 'RGI00-00.00000'
        self.glims_id = '00'
        self.rgi_area_km2 = 0.
        self.cenlon = (entity['max_x'] + entity['min_x']) / 2.
        self.cenlat = (entity['max_y'] + entity['min_y']) / 2.
        self.rgi_region = '00'
        self.rgi_subregion = '00-00'
        self.rgi_version = '00'

        self.name = entity['name']
        self.glacier_type = entity.get('glacier_type', 'Ice Cap')
        self.terminus_type = entity.get('terminus_type', 'Land-terminating')

        self.is_tidewater = self.terminus_type in ['Marine-terminating',
                                                   'Lake-terminating']
        self.inversion_calving_rate = 0.
        self.is_icecap = self.glacier_type == 'Ice cap'

        # Hemisphere
        self.hemisphere = 'sh' if self.cenlat < 0 else 'nh'

        # The divides dirs are created by gis.define_glacier_region, but we
        # make the root dir
        self.dir = os.path.join(base_dir, self.name)
        if reset and os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        mkdir(self.dir)

        # logging file
        self.logfile = os.path.join(self.dir, 'log.txt')

        # Optimization
        self._mbdf = None

    def __repr__(self):

        summary = ['<oggm.NonRGIGlacierDirectory>']
        summary += ['  RGI id: ' + self.rgi_id]
        summary += ['  Region: ' + self.rgi_region_name]
        summary += ['  Subregion: ' + self.rgi_subregion_name]
        if self.name :
            summary += ['  Name: ' + self.name]
        summary += ['  Glacier type: ' + str(self.glacier_type)]
        summary += ['  Terminus type: ' + str(self.terminus_type)]
        summary += ['  Area: ' + str(self.rgi_area_km2) + ' km2']
        summary += ['  Lon, Lat: (' + str(self.cenlon) + ', ' +
                    str(self.cenlat) + ')']
        if os.path.isfile(self.get_filepath('glacier_grid')):
            summary += ['  Grid (nx, ny): (' + str(self.grid.nx) + ', ' +
                        str(self.grid.ny) + ')']
            summary += ['  Grid (dx, dy): (' + str(self.grid.dx) + ', ' +
                        str(self.grid.dy) + ')']
        return '\n'.join(summary) + '\n'

    def copy_to_basedir(self, base_dir, setup='run'):
        """

        Parameters
        ----------
        base_dir
        setup

        Returns
        -------

        """
        raise NotImplementedError

    def get_ref_mb_data(self):
        """

        Returns
        -------

        """
        raise NotImplementedError

    def get_ref_length_data(self):
        """

        Returns
        -------

        """
        raise NotImplementedError

    def get_inversion_flowline_hw(self):
        """

        Returns
        -------

        """
        raise NotImplementedError


@entity_task(log, writes=['glacier_grid', 'dem', 'outlines'])
def define_nonrgi_glacier_region(gdir:NonRGIGlacierDirectory, dx=400.):
    """
    Very first task: define the glacier's local grid.

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
    gdir : :py:class:`oggm.NonRGIGlacierDirectory`
        where to write the data
    dx : float
        grid spacing
    """
    xx, yy = gdir.extent_ll

    # Make a local glacier map
    proj_params = dict(name='tmerc', lat_0=0., lon_0=gdir.cenlon,
                       k=0.9996, x_0=0, y_0=0, datum='WGS84')
    proj4_str = "+proj={name} +lat_0={lat_0} +lon_0={lon_0} +k={k} " \
                "+x_0={x_0} +y_0={y_0} +datum={datum}".format(**proj_params)
    #proj_in = pyproj.Proj("+init=EPSG:4326", preserve_units=True)
    proj_out = pyproj.Proj(proj4_str, preserve_units=True)

    merc_xx, merc_yy = salem.transform_proj(salem.wgs84, proj_out, xx, yy)

    # Corners, incl. a buffer of N pix
    # TODO: do we need the border?
    ulx = np.min(merc_xx) #- cfg.PARAMS['border'] * dx
    lrx = np.max(merc_xx) #+ cfg.PARAMS['border'] * dx
    uly = np.max(merc_yy) #+ cfg.PARAMS['border'] * dx
    lry = np.min(merc_yy) #- cfg.PARAMS['border'] * dx

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


    Parameters
    ----------
    gdir : :py:class:`oggm.NonRGIGlacierDirectory`
        where to get the dem and obtain the data
    border_h : float
        height of the outside border
    px_count : int
    filename : string
    """
    with rasterio.open(gdir.get_filepath('dem')) as src:
        pixel_vals = src.read(1)
        profile = src.profile

    if px_count > 0:
        #TODO: pixel_vals manipulation
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