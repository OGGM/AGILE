import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from cobbi.core import gis, test_cases
from cobbi.core.utils import NonRGIGlacierDirectory
from cobbi.core.first_guess import compile_first_guess
from cobbi.core.inversion import InversionDirectory
from cobbi.core.dynamics import create_glacier
from cobbi.core.cost_function import create_cost_func
from cobbi.core.inversion import InversionDirectory
from cobbi.core import data_logging
from cobbi.core.data_manipulation import take_true_bed_as_first_guess
from cobbi.core.data_manipulation import add_noise_to_first_guess
from cobbi.core.data_manipulation import create_perlin_noise
from oggm import cfg

cfg.initialize()

desired_rmse = 5
freq = 8.  # determines coherence of noise. Larger frequ gives larger
# coherence
base = 3  # works like np.random.seed

basedir = '/path/to/example'
basedir = '/data/philipp/thesis_test/Giluwe/noisy_first_guess/'


# TODO: think about IceThicknesses for case Giluwe
# Choose a case
case = test_cases.Giluwe
gdir = NonRGIGlacierDirectory(case, basedir)
# only needed once:
#gis.define_nonrgi_glacier_region(gdir)

# create settings for inversion
lambdas = np.zeros(4)
lambdas[0] = 0.2  # TODO: better
lambdas[1] = 1.5  # TODO: really useful? (Better if smaller than 1 to focus
# on inner domain)
lambdas[2] = 2
lambdas[3] = 1e5


minimize_options = {
    'maxiter': 300,
    'ftol': 0.5e-3,
    #'xtol': 1e-30,
    'gtol': 1e-4,
    #'maxcor': 5,
    #'maxls': 10,
    'disp': True
}

subdir = 'rmse{:g}_freq{:g}_base_{:d}/'.format(desired_rmse, freq, base)

# Theoretically also only needed once
gdir.write_inversion_settings(mb_spinup=None,
                              yrs_spinup=2000,
                              yrs_forward_run=200,
                              reg_parameters=lambdas,
                              solver='L-BFGS-B',
                              minimize_options=minimize_options,
                              inversion_subdir=subdir,
                              fg_shape_factor=1.,
                              bounds_min_max=(2, 600)
                              )

# Optional, if not reset=True and already ran once
# only needed once:
#create_glacier(gdir)
#     compile_first_guess(gdir)

noise = create_perlin_noise(gdir, desired_rmse, base=base, freq=freq,
                            glacier_only=True)
plt.figure()
plt.imshow(noise)
plt.show()

take_true_bed_as_first_guess(gdir)
add_noise_to_first_guess(gdir, noise)

idir = InversionDirectory(gdir)

# copy this script to inversion directory for reproducibility
path_to_file = '/home/philipp/COBBI/cobbi/sandbox' \
               '/noisy_first_guess_inversion_run.py'
fname = os.path.split(path_to_file)[-1]
dst_dir = idir.get_current_basedir()
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
shutil.copy(path_to_file, os.path.join(dst_dir, fname))

# Finally run inversion
res = idir.run_minimize()
#dl = data_logging.load_pickle(idir.get_current_basedir() + '/data_logger.pkl')



print('end')