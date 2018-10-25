from cobbi.utils.lcurve import create_logspaced_lamb_sequence
import numpy as np
import torch
from cobbi.utils import test_cases
from cobbi.utils.optimization import *
import matplotlib.pyplot as plt
# from itertools import product

np.random.seed(0)

n = 8

lambda_values = np.zeros((9, n))
#lambda_values[0] = [1., 5., 12., 25., 50.]
#lambda_values[2] = [0.1, 0.5, 1., 3., 10.]
#lambda_values[4] = [1., 20., 100., 300., 625.]
#lambda_values[5] = [1., 10., 40., 100., 300.]
#lambda_values[6] = [1., 10., 40., 100., 300.]
#lambda_values[7] = [4., 40., 150., 400., 800.]
#lambda_values[8] = [0.1, 0.5, 0.8, 1., 1.5, 3., 5., 10.]

#lambdas_list = []

lambdas_list = []
lambdas_list.append(np.zeros(9))
lambdas_list.append(np.array([0, 0, 0.5, 0, 0, 0, 0, 0, 0]))
#for lamb in lambda_values[8][::-1]:
#    lambdas_list.append(np.array([0, 0, 1, 0, 0, 0, 0, 0, lamb]))

#for i in [0, 2, 4, 5, 6, 7]:
#    for j in range(n):
#        lambdas = np.zeros(9)
#        lambdas[i] = lambda_values[i, j]
#        if i == 6:
#            lambdas[i - 1] = lambda_values[i - 1, j]
#        #Spezialfall
#        lambdas_list.append(lambdas)

case = test_cases.Nanisivik
#case.dx = 3000
#case.smooth_border_px = 2

y0 = 0
y_spinup_end = 2000
y_end = 2800

solver = 'L-BFGS-B'
minimize_options = {
    'maxiter': 80,
    'ftol': 1e-3,
    #'xtol': 1e-30,
    'gtol': 1e-4,
    'disp': True}

test = LCurveTest(case, y0, y_spinup_end, y_end,
                  minimize_options=minimize_options, solver=solver)
f = 1.
sca = 2.5
test.first_guess = first_guess(test.reference_surf, test.ice_mask, test.case.dx,
                               slope_cutoff_angle=sca, factor=f)
test.basedir = '/data/philipp/tests/lcurve/test_nanisivik/'
#test.maxiter = 30
#test.optimization_counter =
# Disturb surface:
#test.add_surface_noise(std=5) #, zoom=1.5)
test.optimization_counter = 400

for lambs in lambdas_list:
    #try:
    test.lambdas = torch.tensor(lambs, dtype=torch.float, requires_grad=False)
    test.run_minimize2(update_scaling=0.85)
    #except:
    #    pass

#TODO: compute curvature of noise for comparison, ...

print('end')