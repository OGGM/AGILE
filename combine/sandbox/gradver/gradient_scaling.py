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
from combine.sandbox.gradver.cost_function import create_cost_func
from combine.core.inversion import InversionDirectory
from combine.core import data_logging
from combine.core.visualization import plot_gradient
from oggm import cfg
from scipy.optimize import approx_fprime

np.random.seed(0)  # needs to be fixed for reproducible results with noise

cfg.initialize()

basedir = '/path/to/example'
basedir = '/data/philipp/thesis/gradient_verification'

# Choose a case
case = test_cases.Giluwe
gdir = NonRGIGlacierDirectory(case, basedir)
# only needed once:
gis.define_nonrgi_glacier_region(gdir)

# create settings for inversion
lambdas = np.zeros(4)
lambdas[0] = 0.2
lambdas[1] = 0.25
lambdas[2] = 100
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

gdir.write_inversion_settings(mb_spinup=None,
                              yrs_spinup=2000,
                              yrs_forward_run=200,
                              reg_parameters=lambdas,
                              solver='L-BFGS-B',
                              minimize_options=minimize_options,
                              inversion_subdir='test',
                              fg_shape_factor=1.,
                              fg_slope_cutoff_angle=2.5,
                              fg_min_height=-30,
                              fg_interp_boundary=True,
                              bounds_min_max=(2, 1000)
                              )

create_glacier(gdir)
first_guess = compile_first_guess(gdir) # TODO capsulate compile_first_guess in sole get_first guess
costs = []
grads = []
cost_func = create_cost_func(gdir, use_AD=True)
cost, grad = cost_func(first_guess)
costs.append(cost)
grads.append(grad)
name = 'pytorch'
filepath = os.path.join(gdir.dir, name + '.npy')
np.save(filepath, grad)
filepath = os.path.join(gdir.dir, name + '.png')
plot_gradient(filepath, grad, name, ref_shape=first_guess.shape)

for db in [3, 2, 1, 0.1, 0.01, 0.001]:
    with torch.no_grad():
        cost_func2 = create_cost_func(gdir, use_AD=False)
        b = np.array(first_guess, dtype=np.float32).flatten()
        fin_dif_grad = approx_fprime(b, cost_func2, db)
    fin_dif_grad = fin_dif_grad.reshape(first_guess.shape)

    filepath = os.path.join(gdir.dir, 'fd_db_{:g}.npy'.format(db))
    np.save(filepath, fin_dif_grad)
    filepath = os.path.join(gdir.dir, 'fd_db_{:g}.png'.format(db))
    plot_gradient(filepath, fin_dif_grad,
                  'finite difference approximated gradient\n$db = {:g}m$'.format(db),
                  ref_shape=first_guess.shape)


    filepath = os.path.join(gdir.dir, 'abs_diff_db_{:g}.png'.format(db))
    plot_gradient(filepath, grad - fin_dif_grad,
                  'absolute difference of PyTorch AD and \n finite difference'
                  'gradient\n$db = {:g}m$'.format(db),
                  ref_shape=first_guess.shape)


print('end')