import torch
import matplotlib.pyplot as plt
import numpy as np

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

np.seed = 0  # needs to be fixed for reproducible results with noise

cfg.initialize()

desired_rmse = 5
base = 1  # works as np.random.seed
freq = 8.  # determines coherence of noise. Larger frequ gives larger
# coherence

basedir = '/path/to/example'
basedir = '/data/philipp/thesis_test/Giluwe/noisy_first_guess/'
basedir = basedir + '{:g}/{:g}/'.format(desired_rmse, freq)

# TODO: think about IceThicknesses for case Giluwe
# Choose a case
case = test_cases.Giluwe
gdir = NonRGIGlacierDirectory(case, basedir)
# only needed once:
gis.define_nonrgi_glacier_region(gdir)

# create settings for inversion
lambdas = np.zeros(11)
lambdas[0] = 0.2
lambdas[3] = 1.5
lambdas[7] = 1e5
lambdas[10] = 2

minimize_options = {
    'maxiter': 300,
    'ftol': 0.5e-3,
    #'xtol': 1e-30,
    'gtol': 1e-4,
    #'maxcor': 5,
    #'maxls': 10,
    'disp': True
}

# Theoretically also only needed once
gdir.write_inversion_settings(mb_spinup=None,
                              yrs_spinup=2000,
                              yrs_forward_run=200,
                              reg_parameters=lambdas,
                              solver='L-BFGS-B',
                              minimize_options=minimize_options,
                              inversion_counter=base,
                              fg_shape_factor=1.,
                              bounds_min_max=(2, 600)
                              )

# Optional, if not reset=True and already ran once
# only needed once:
# create_glacier(gdir)
#     compile_first_guess(gdir)

noise = create_perlin_noise(gdir, desired_rmse, base=base, freq=freq,
                            glacier_only=True)
plt.figure()
plt.imshow(noise)
plt.show()

take_true_bed_as_first_guess(gdir)
add_noise_to_first_guess(gdir, noise)

idir = InversionDirectory(gdir)
res = idir.run_minimize()
#dl = data_logging.load_pickle(idir.get_current_basedir() + '/data_logger.pkl')
print('end')