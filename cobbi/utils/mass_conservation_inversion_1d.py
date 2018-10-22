import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from oggm import cfg, tasks
from oggm.utils import GlacierDirectory
from functools import partial
import pyproj
import shapely.ops
import shapely.geometry as shpg
import glob
import pandas as pd

cfg.initialize()
cfg.PATHS['working_dir'] = '/home/philipp/tmp/synthetic_ice_cap/Nanisivik Arctic Bay/per_glacier/Nanisivik Arctic Bay/'
cfg.PATHS['dem_file'] = cfg.PATHS['working_dir'] + 'surface1500m_2800a_40m_res.tiff'

entity = gpd.read_file(cfg.PATHS['working_dir'] + 'ice_mask_1500m_2800a.shp').iloc[0]

cfg.PARAMS['invert_with_sliding'] = False
cfg.PARAMS['border'] = 20
cfg.PARAMS['min_slope'] = 2
cfg.PARAMS['max_shape_param'] = 0.006
cfg.PARAMS['max_thick_to_width_ratio'] = 0.5
cfg.PARAMS['base_binsize'] = 100.
cfg.PARAMS['temp_use_local_gradient'] = False
cfg.PARAMS['use_intersects'] = False
cfg.PARAMS['d1'] = 100

divides = glob.glob(cfg.PATHS['working_dir'] + 'divide_*/outlines.shp')

divide_geometries = []
for f in divides:
    ent = gpd.read_file(f).iloc[0]
    divide_geometries.append(ent['geometry'])

multi = shpg.MultiPolygon(divide_geometries)
inif = 0.
add = 1e-5
while True:
    buff = multi.buffer(inif)
    if 'Multi' in buff.type:
        inif += add
    else:
        break
#x, y = multi.centroid.xy
if 'Multi' in buff.type:
    raise RuntimeError

entity['geometry'] = buff

cenlat = 0.
cenlon = 0.
for s in buff.proj.srs.split('+'):
    if s.startswith('lat_0'):
        cenlat = float(s.split('=')[-1])
    if s.startswith('lon_0'):
        cenlon = float(s.split('=')[-1])

intersections = []
for geom in divide_geometries


# neccessary for coordinate system conventions of OGGM
proj_params = dict(name='tmerc', lat_0=0., lon_0=gdir.cenlon,
                   k=0.9996, x_0=0, y_0=0, datum='WGS84')
proj4_str = "+proj={name} +lat_0={lat_0} +lon_0={lon_0} +k={k} " \
            "+x_0={x_0} +y_0={y_0} +datum={datum}".format(**proj_params)
proj_out = pyproj.Proj("+init=EPSG:4326", preserve_units=True)
proj_in = pyproj.Proj(proj4_str, preserve_units=True)
project = partial(pyproj.transform, proj_in, proj_out)
# transform geometry to map
geometry = shapely.ops.transform(project, entity['geometry'])

entity['geometry'] = geometry

divides = []
div_sel = dict()
for k, v in entity.iteritems():
    if k == 'geometry':
        div_sel[k] = multi
    elif k == 'RGIId':
        div_sel['RGIID'] = v
    else:
        div_sel[k] = v
divides.append(div_sel)


# Add divides to the original one
adf = pd.DataFrame(divides)
adf.to_pickle(cfg.PATHS['working_dir'] + 'divs')

gdir = GlacierDirectory(entity, base_dir=cfg.PATHS['working_dir'])
tasks.define_glacier_region(gdir, entity=entity)
