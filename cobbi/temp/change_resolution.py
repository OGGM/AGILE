import os
from oggm import cfg, utils
from cobbi.utils import test_cases
from cobbi.utils.synthetic_ice_caps import NonRGIGlacierDirectory


cfg.initialize()
case = test_cases.Nanisivik
entity = {'min_x': case.extent[0, 0],
          'max_x': case.extent[1, 0],
          'min_y': case.extent[0, 1],
          'max_y': case.extent[1, 1],
          'name': case.name}

# Local working directory (where OGGM will write its output)
WORKING_DIR = os.path.join(os.path.expanduser('~'), 'tmp',
                        'synthetic_ice_cap', case.name)
utils.mkdir(WORKING_DIR, reset=False)
cfg.PATHS['working_dir'] = WORKING_DIR

gdir = NonRGIGlacierDirectory(entity)

# get path to the input data
input_shp = gdir.get_filepath('ice_mask', filesuffix='_1500m_2800a')
input_dem = gdir.get_filepath('surface', filesuffix='_1500m_2800a')

# filter options
f_area = False
f_alt_range = True
f_perc_alt_range = True

# set paths to python 2.7 and to the partitioning package
python = '/usr/bin/python2.7'
project = '/home/philipp/partitioning'

script = os.path.join(project, 'examples/run_divides.py')

# run code from your console (PYTHON 2.7!)
os.system(python + ' ' + script + ' ' + input_shp + ' ' + input_dem + ' ' +
          str(f_area) + ' ' + str(f_alt_range) + ' ' + str(f_perc_alt_range))

print('Hintereisferner is divided into')#, gdir.n_divides, 'parts.')