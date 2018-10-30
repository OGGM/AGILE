from cobbi.utils.lcurve import create_logspaced_lamb_sequence
import numpy as np
import torch
from cobbi.utils import test_cases
from cobbi.utils.optimization import *
import matplotlib.pyplot as plt
# from itertools import product
from scipy.optimize import Bounds

np.random.seed(0)

n = 5

lambda_values = np.zeros((9, n))
lambda_values[0] = [1., 10., 70., 200., 2000]
lambda_values[0] = lambda_values[0][::-1]
lambda_values[1] = [1., 10., 70., 200., 2000]
lambda_values[1] = lambda_values[1][::-1]
lambda_values[2] = [10., 3., 30., None, None]
lambda_values[4] = [1., 20., 100., 300., 625.]
lambda_values[5] = [1., 10., 40., 100., 300.]
lambda_values[6] = [1., 10., 40., 100., 300.]
lambda_values[7] = [4., 40., 150., 400., 800.]

lambdas_list = []

lambdas_list = []
lambdas_list.append(np.zeros(9))

for i in [0, 1, 2, 4, 5, 6, 7]:
    for j in range(n):
        lambdas = np.zeros(9)
        if lambda_values[i, j] is not None:
            lambdas[i] = lambda_values[i, j]
            #if i == 6:
            #    lambdas[i - 1] = lambda_values[i - 1, j]
            #Spezialfall
            lambdas_list.append(lambdas)

case = test_cases.Giluwe
#case.dx = 200
#case.dx = 400
#case.smooth_border_px = 2

y0 = 0
y_spinup_end = 2000
y_end = 2200

solver = 'L-BFGS-B'
minimize_options = {
    'maxiter': 100,
    'ftol': 1e-3,
    #'xtol': 1e-30,
    'gtol': 1e-4,
    #'maxcor': 5,
    #'maxls': 10,
    'disp': True}

test = LCurveTest(case, y0, y_spinup_end, y_end,
                  minimize_options=minimize_options, solver=solver)

estimated_max_ice_thickness = 600.
estimated_min_ice_thickness = 2.
lower_bounds = test.reference_surf - estimated_max_ice_thickness * test.ice_mask
upper_bounds = test.reference_surf - estimated_min_ice_thickness * test.ice_mask
bounds = np.c_[lower_bounds.flatten(), upper_bounds.flatten()]
test.optimization_bounds = bounds

f = 1.
sca = 5.
test.first_guess = first_guess(test.reference_surf, test.ice_mask, test.case.dx,
                               slope_cutoff_angle=sca, factor=f)
test.basedir = '/data/philipp/tests/giluwe/no_noise/'
test.maxiter = 100
#test.optimization_counter =
# Disturb surface:
#test.add_surface_noise(std=5) #, zoom=1.5)
start_at = 0
test.optimization_counter = start_at + 3000
my_list = lambdas_list[start_at:]
for lambs in my_list:
    #try:
    test.lambdas = torch.tensor(lambs, dtype=torch.float, requires_grad=False)
    #test.run_minimize2(update_scaling=0.4)
    test.run_minimize()
    #except:
    #    pass

#TODO: compute curvature of noise for comparison, ...

print('end')