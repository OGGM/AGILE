import numpy as np

import torch
torch.utils.backcompat.broadcast_warning.enabled = True
# import matplotlib.pyplot as plt

from cobbi.sia2d_adapted import Upstream2D
# from cobbi.sia2d_pytorch_full import Upstream2D as Up2D
# from cobbi.utils import test_cases
from oggm import cfg
from oggm import utils

from os import path
import salem

from cobbi.utils.synthetic_ice_caps import NonRGIGlacierDirectory
from cobbi.utils.synthetic_ice_caps \
    import define_nonrgi_glacier_region, smooth_dem_borders
from cobbi.utils.massbalance_pytorch \
    import LinearMassBalance
from oggm.core.centerlines import _mask_to_polygon
from cobbi.utils import test_cases
import os
import pyproj
import rasterio
from scipy.ndimage.interpolation import zoom
import geopandas as gpd
from skimage import measure
import matplotlib.pyplot as plt
import shapely.geometry as shpg
from scipy.ndimage.measurements import label

# Variable needed later
LABEL_STRUCT = np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]])

def upsample_surface_for_partitioning(gdir, inputfilename, inputsuffix,
                                      outputfilename, outputsuffix=''):
    '''
    Upsamples the coarse model surface to a dem for partitioning.
    Follows
    https://github.com/mapbox/rasterio/blob/master/docs/topics/resampling.rst

    :param gdir:
    :return:
    '''
    # TODO: Welche Oberfläche soll genommen werden? - egal, interessant sind sowieso erstmal nur die Geoinformationen, nicht die eigentlichen Daten
    dem_file = gdir.get_filepath(inputfilename, filesuffix=inputsuffix)

    with rasterio.open(dem_file) as src:
        data = src.read()
        profile = src.profile
        aff = src.transform

    # TODO: ask for original dx?
    zoom_factor = abs(aff.a) / 40.  # we need 40m resolution

    newarr = zoom(data, zoom=[1, zoom_factor, zoom_factor])

    newaff = rasterio.Affine(aff.a / zoom_factor, aff.b, aff.c, aff.d,
                             aff.e / zoom_factor, aff.f)
    profile['transform'] = newaff
    profile['width'] = newarr.shape[2]
    profile['height'] = newarr.shape[1]

    surf_file = gdir.get_filepath(outputfilename,
                                  filesuffix=outputsuffix + '_40m_res')
    with rasterio.open(surf_file, 'w', **profile) as dst:
        dst.write(newarr)
    return surf_file


def get_filesuffix(dx, year):
    return '_{:g}m_{:g}a'.format(dx, year)


def write_surface(gdir, filename, filesuffix, surf_arr):
    # Make a local glacier map
    dem_file = gdir.get_filepath('dem')
    with rasterio.open(dem_file) as src:
        profile = src.profile
    profile['dtype'] = 'float32'
    fname = gdir.get_filepath(filename, filesuffix=filesuffix)
    with rasterio.open(fname, 'w', **profile) as dst:
        dst.write(surf_arr, 1)
    return fname


def write_mask(gdir, filename, filesuffix, ice_mask):
    grid = salem.Grid.from_json(gdir.get_filepath('glacier_grid'))
    gdf = gpd.GeoDataFrame()
    # TODO: check for nunataks
    poly, poly_no = _mask_to_polygon(ice_mask, gdir=gdir)
    gdf.loc[0, 'geometry'] = poly
    gdf.crs = grid
    gdf.loc[0, 'Area'] = poly.area
    gdf.loc[0, 'Name'] = gdir.name
    cenlat = 0.
    cenlon = 0.
    for s in grid.proj.srs.split('+'):
        if s.startswith('lat_0'):
            cenlat = float(s.split('=')[-1])
        if s.startswith('lon_0'):
            cenlon = float(s.split('=')[-1])
    gdf.loc[0, 'CenLat'] = cenlat
    gdf.loc[0, 'CenLon'] = cenlon
    gdf.loc[0, 'GlacType'] = '10'
    gdf.loc[0, 'RGIId'] = 'RGI50-00.00000'
    gdf.loc[0, 'O1Region'] = '00'
    gdf.loc[0, 'O2Region'] = '0'
    gdf.loc[0, 'GLIMSId'] = 'G010758E46800N'  # just use glims of HEF TODO: should reflect lat/lon of centerpoint
    # gdf.loc[0, 'Aspect'] = 0
    gdf.loc[0, 'BgnDate'] = 20180101
    gdf.loc[0, 'EndDate'] = 20180101
    gdf.loc[0, 'RGIFlag'] = '0909'

    salem.transform_geopandas(gdf, gdir.grid.proj, inplace=True)
    if hasattr(gdf.crs, 'srs'):
        # salem uses pyproj
        gdf.crs = gdf.crs.srs
    fname = gdir.get_filepath(filename, filesuffix=filesuffix)
    gdf.to_file(fname)
    return fname


cfg.initialize()
case = test_cases.Nanisivik
entity = {'min_x': case.extent[0, 0],
          'max_x': case.extent[1, 0],
          'min_y': case.extent[0, 1],
          'max_y': case.extent[1, 1],
          'name': case.name}

y0 = 0
y_spinup_end = 2000
y_end = 2800

# Local working directory (where OGGM will write its output)
WORKING_DIR = path.join(path.expanduser('~'), 'tmp',
                        'synthetic_ice_cap', case.name)
utils.mkdir(WORKING_DIR, reset=False)
cfg.PATHS['working_dir'] = WORKING_DIR

gdir = NonRGIGlacierDirectory(entity)

define_nonrgi_glacier_region(gdir, dx=case.dx)
smooth_dem_borders(gdir, px_count=case.smooth_border_px,
                   border_h=case.smooth_border_h)
ds = salem.GeoTiff(gdir.get_filepath('dem', filesuffix='_smooth_border'))
bed_2d = torch.tensor(ds.get_vardata(), dtype=torch.float, requires_grad=False)

mb = LinearMassBalance(case.ela_h, grad=case.mb_grad)
# Create glacier
with torch.no_grad():
    reference_model = Upstream2D(bed_2d, dx=case.dx, mb_model=mb, y0=0,
                                 glen_a=cfg.PARAMS['glen_a'], ice_thick_filter=None)
    reference_model.run_until(y_spinup_end)
    start_surf = reference_model.surface_h
    reference_model.run_until(y_end)
    reference_surf = reference_model.surface_h
    ice_mask = (reference_surf > bed_2d)


write_surface(gdir, 'surface', get_filesuffix(case.dx, y_end),
              reference_surf.detach().numpy())
upsample_surface_for_partitioning(gdir, 'surface',
                                  get_filesuffix(case.dx, y_end), 'surface',
                                  get_filesuffix(case.dx, y_end))
write_mask(gdir, 'ice_mask', get_filesuffix(case.dx, y_end),
           ice_mask.detach().numpy())

print('end')




