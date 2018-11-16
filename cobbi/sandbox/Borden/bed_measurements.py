import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from cobbi.core import gis, test_cases
from cobbi.core.utils import NonRGIGlacierDirectory
from cobbi.core.first_guess import compile_first_guess
from cobbi.core.first_guess import compile_biased_first_guess
from cobbi.core.dynamics import create_glacier
from cobbi.core.inversion import InversionDirectory
from cobbi.core.table_creation import create_case_table, eval_identical_twin
from cobbi.sandbox import bed_measurement_masks
from cobbi.core.data_manipulation import add_bed_measurements
from cobbi.core.data_manipulation import generate_bed_measurements
from cobbi.core.data_manipulation import create_perlin_noise
from cobbi.core.data_manipulation import add_surface_noise
from oggm import cfg

cfg.initialize()

basedir = '/path/to/example'
basedir = '/data/philipp/thesis/bed_measurements'

# Choose a case
case = test_cases.Borden
gdir = NonRGIGlacierDirectory(case, basedir)
# only needed once:
#gis.define_nonrgi_glacier_region(gdir)
np.random.seed(0)
bed_measurements_mask = bed_measurement_masks.measurement_mask_Borden_horizontal
bed_measurements = generate_bed_measurements(gdir, bed_measurements_mask,
                                             std=10)
# reproducibility
add_bed_measurements(gdir, bed_measurements)

#remodel 3c
scaling = 200
#desired_rmse = 2
#desired_rmse = 6
desired_rmse = 10

additional_scaling = 1

lambdas = np.zeros(6)
lambdas[0] = 0.2
lambdas[1] = 0.25 * 1e-2
lambdas[2] = 100 * scaling * additional_scaling
lambdas[3] = 1e5 * scaling * additional_scaling
lambdas[4] = 1e7
lambdas[5] = 1e-1

minimize_options = {
    'maxiter': 300,
    'ftol': 0.5e-3,
    #'xtol': 1e-30,
    'gtol': 1e-4,
    #'maxcor': 5,
    #'maxls': 10,
    'disp': True
}

gdir.write_inversion_settings(mb_spinup=None,
                              yrs_spinup=2000,
                              yrs_forward_run=200,
                              reg_parameters=lambdas,
                              solver='L-BFGS-B',
                              minimize_options=minimize_options,
                              inversion_subdir='3c std 10 l5 1e-1 l1 1e-2',
                              fg_shape_factor=1.,
                              fg_slope_cutoff_angle=2.5,
                              fg_min_height=-30,
                              fg_interp_boundary=True,
                              bounds_min_max=(2, 1000)
                              )

# Optional, if not reset=True and already ran once
# only needed once:
#create_glacier(gdir)
compile_first_guess(gdir)

noise = create_perlin_noise(gdir, desired_rmse, octaves=4, base=2, freq=3,
                            glacier_only=True)
add_surface_noise(gdir, noise)


create_case_table(gdir)

idir = InversionDirectory(gdir)

# copy this script to inversion directory for reproducibility
path_to_file = '/home/philipp/COBBI/cobbi/sandbox/Borden' \
               '/bed_measurements.py'
fname = os.path.split(path_to_file)[-1]
if not os.path.exists(idir.get_current_basedir()):
    os.makedirs(idir.get_current_basedir(), exist_ok=True)
shutil.copy(path_to_file, os.path.join(idir.get_current_basedir(), fname))

res = idir.run_minimize()
eval_identical_twin(idir)
#dl = data_logging.load_pickle(idir.get_current_basedir() + '/data_logger.pkl')

print('end')