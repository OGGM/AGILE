import torch
from cobbi.utils import test_cases
from cobbi.utils.optimization import *
import matplotlib.pyplot as plt
import numpy as np

case = test_cases.Blafell
case.dx = 600
case.smooth_border_px = 1

y0 = 0
y_spinup_end = 2000
y_end = 2500

lambs = torch.tensor([0.2, 0.2, 1.5, 2., 2., 2., 0., 10., 0.])

test = LCurveTest(case, y0, y_spinup_end, y_end)
test.lambdas = lambs

test.maxiter = 5
test.run_minimize()

print('end')