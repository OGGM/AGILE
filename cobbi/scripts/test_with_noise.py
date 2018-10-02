from cobbi.utils.lcurve import create_logspaced_lamb_sequence
import numpy as np
import torch
from cobbi.utils import test_cases
from cobbi.utils.optimization import *
import matplotlib.pyplot as plt
# from itertools import product

np.random.seed(0)

n = 5
lamb_min = 0.01
lamb_max = 30

ls = create_logspaced_lamb_sequence(lamb_max, lamb_min, n)

lambda_values = np.zeros((9, n))

lambda_values[0] = create_logspaced_lamb_sequence(60000., 6., n)
lambda_values[1] = create_logspaced_lamb_sequence(250., 0.025, n)
lambda_values[2] = create_logspaced_lamb_sequence(100., 0.05, n)
lambda_values[3] = create_logspaced_lamb_sequence(700, 10, n)
lambda_values[4] = create_logspaced_lamb_sequence(3.e9, 3.e5, n)
lambda_values[5] = create_logspaced_lamb_sequence(1.5e7, 1.5e3, n)
#lambda_values[7] = create_logspaced_lamb_sequence(, , n)
#lambda_values[8] = create_logspaced_lamb_sequence(, , n)

lambdas_list = []
for i in [3]:  # range(lambda_values.shape[0]):
    for j in range(0, lambda_values.shape[1]):
        lambdas = np.zeros(9)
        lambdas[i] = lambda_values[i, j]
        lambdas_list.append(lambdas)

lambdas_list.append(np.zeros(9))


case = test_cases.Kinabalu
case.dx = 600
case.smooth_border_px = 1

y0 = 0
y_spinup_end = 2000
y_end = 2500

solver = 'L-BFGS-B'
minimize_options = {
    'maxiter': 200,
    'ftol': 1e-2,
    #'xtol': 1e-10,
    'gtol': 1e-3,
    'disp': True}

test = LCurveTest(case, y0, y_spinup_end, y_end,
                  minimize_options=minimize_options, solver=solver)
test.basedir = '/data/philipp/tests/lcurve/reg_3_LBFGSB_noise/'
#test.maxiter = 30

# Disturb surface:
test.add_surface_noise(std=5, zoom=1.5)

for lambs in lambdas_list:
    test.lambdas = torch.tensor(lambs, dtype=torch.float, requires_grad=False)
    test.run_minimize()

print('end')