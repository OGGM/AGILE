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
from combine.core.data_manipulation import take_true_bed_as_first_guess
from combine.core.data_manipulation import add_noise_to_first_guess
from combine.core.data_manipulation import create_perlin_noise
from oggm import cfg
import itertools

cfg.initialize()

first_run = False  # set to true to (re)run spinup and creation of glacier

basedir = '/path/to/example'
basedir = '/data/philipp/thesis_test2/Giluwe/noisy_first_guess/'

# TODO: think about IceThicknesses for case Giluwe
# Choose a case
case = test_cases.Giluwe
gdir = NonRGIGlacierDirectory(case, basedir)
# only needed once:
if first_run:
    gis.define_nonrgi_glacier_region(gdir)

# create settings for inversion
lambdas = np.zeros(4)
lambdas[0] = 0.2  # TODO: better
lambdas[1] = 1.0  # TODO: really useful? (Better if smaller than 1 to focus
# on inner domain?)
lambdas[2] = 2
lambdas[3] = 1e7


minimize_options = {
    'maxiter': 300,
    'ftol': 0.5e-3,
    #'xtol': 1e-30,
    'gtol': 1e-4,
    #'maxcor': 5,
    #'maxls': 10,
    'disp': True
}

# ------ set parameters for noise creation
octaves = 4  # found to give good results
desired_rmses = [5, 10, 20, 30, 50]
freqs = [12, 8, 6, 4, 3] # determines coherence of noise. Larger freq gives
# larger
# coherence
bases = [0, 1, 2, 3, 4]  # works like np.random.seed
all_noise_parameters = itertools.product(desired_rmses, freqs, bases)
# print(list(all_noise_parameters))

for noise_parameters in all_noise_parameters:
    # TODO: allow for parallelization of this for loop.
    # Therefore, noise, inversion settings, ... have to be created in the
    # idir instead of gdir from beginning
    subdir = 'noise_{:02d}_{:02d}_{:d}/'.format(noise_parameters[0],
                                                noise_parameters[1],
                                                noise_parameters[2])

    # Theoretically also only needed once
    gdir.write_inversion_settings(mb_spinup=None,
                                  yrs_spinup=2000,
                                  yrs_forward_run=200,
                                  reg_parameters=lambdas,
                                  solver='L-BFGS-B',
                                  minimize_options=minimize_options,
                                  inversion_subdir=subdir,
                                  fg_shape_factor=1.,
                                  bounds_min_max=(2, 300)
                                  )

    # Optional, if not reset=True and already ran once
    # only needed once:
    if first_run:
        create_glacier(gdir)
        first_run = False
    # compile_first_guess(gdir)

    noise = create_perlin_noise(gdir, noise_parameters[0],
                                octaves=octaves,
                                base=noise_parameters[2],
                                freq=noise_parameters[1],
                                glacier_only=True)
    #plt.figure()
    #plt.imshow(noise)
    #plt.show()

    take_true_bed_as_first_guess(gdir)
    add_noise_to_first_guess(gdir, noise)

    idir = InversionDirectory(gdir)

    # copy this script to inversion directory for reproducibility
    path_to_file = '/home/philipp/COBBI/combine/sandbox/Giluwe' \
                   '/noisy_first_guess_inversion_run.py'
    fname = os.path.split(path_to_file)[-1]
    dst_dir = idir.get_current_basedir()
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(path_to_file, os.path.join(dst_dir, fname))

    # Finally run inversion
    try:
        res = idir.run_minimize()
    except MemoryError as merr:
        error_file = os.path.join(idir.get_current_basedir(),
                                  'memory_error.txt')
        err_text = 'Memory error due to extreme ice dynamics\n' + merr.args[0]
        idir.write_string_to_file(error_file, err_text)
    #dl = data_logging.load_pickle(idir.get_current_basedir() + '/data_logger.pkl')



print('end')