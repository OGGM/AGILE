import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from combine.core import gis, test_cases
from combine.core.utils import NonRGIGlacierDirectory
from combine.core.first_guess import compile_first_guess
from combine.core.inversion import InversionDirectory
from combine.core.dynamics import create_glacier
from combine.core.cost_function import create_cost_func
from combine.core.inversion import InversionDirectory
from combine.core import data_logging
from combine.core.table_creation import create_case_table, eval_identical_twin
from combine.core.data_manipulation import create_perlin_noise, \
    add_surface_noise
from oggm import cfg

cfg.initialize()

basedir = '/path/to/example'
basedir = '/data/philipp/thesis/perturbed_surface'

# Choose a case
case = test_cases.Giluwe
gdir = NonRGIGlacierDirectory(case, basedir)
# only needed once:
# gis.define_nonrgi_glacier_region(gdir)

# create settings for inversion
scaling = 1
desired_rmse = 2
#desired_rmse = 6
#desired_rmse = 10

lambdas = np.zeros(6)
lambdas[0] = 0.2
lambdas[1] = 0.25
lambdas[2] = 100 * scaling
lambdas[3] = 1e5 * scaling
#lambdas[4] = 1e7
#lambdas[5] = 5

minimize_options = {
    'maxiter': 300,
    'ftol': 0.5e-3,
    #'xtol': 1e-30,
    'gtol': 1e-4,
    #'maxcor': 5,
    #'maxls': 10,
    'disp': True
}
inv_subdir = 'fin {:02d} scaling {:02d} {:1g}e7'.format(scaling, desired_rmse,
                                                     lambdas[4]/1e7)
gdir.write_inversion_settings(mb_spinup=None,
                              yrs_spinup=2000,
                              yrs_forward_run=200,
                              reg_parameters=lambdas,
                              solver='L-BFGS-B',
                              minimize_options=minimize_options,
                              inversion_subdir=inv_subdir,
                              fg_shape_factor=1.,
                              fg_slope_cutoff_angle=5,
                              #fg_min_height=-30,
                              fg_interp_boundary=False,
                              bounds_min_max=(2, 1000)
                              )

# Optional, if not reset=True and already ran once
# only needed once:
# create_glacier(gdir)
# compile_first_guess(gdir)

if os.path.exists(gdir.get_filepath('dem_noise')):
    os.remove(gdir.get_filepath('dem_noise'))
noise = create_perlin_noise(gdir, desired_rmse, octaves=4, base=2, freq=3,
                            glacier_only=True)
add_surface_noise(gdir, noise)

create_case_table(gdir)

idir = InversionDirectory(gdir)

# copy this script to inversion directory for reproducibility
path_to_file = '/home/philipp/COBBI/combine/sandbox/Giluwe' \
               '/perturbed_surface.py'
fname = os.path.split(path_to_file)[-1]
if not os.path.exists(idir.get_current_basedir()):
    os.makedirs(idir.get_current_basedir(), exist_ok=True)
shutil.copy(path_to_file, os.path.join(idir.get_current_basedir(), fname))

res = idir.run_minimize()
eval_identical_twin(idir)
#dl = data_logging.load_pickle(idir.get_current_basedir() + '/data_logger.pkl')

print('end')