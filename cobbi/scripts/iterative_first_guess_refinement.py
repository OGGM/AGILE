import torch
from cobbi.utils import test_cases
from cobbi.utils.optimization import *
import matplotlib.pyplot as plt
import numpy as np

f = 1
df = 0.1

case = test_cases.arderin
case.dx = 800
case.smooth_border_px = 1

y0 = 0
y_spinup_end = 2000
y_end = 2500
a = '''
su = spin_up(case, y_spinup_end, y_end)
start_surf = su[0]
reference_surf = su[1].detach().numpy()
ice_mask = su[2].detach().numpy()
mb = su[3]
bed_2d = su[4]

b0 = first_guess(reference_surf, ice_mask, case.dx, factor=f)
s0 = None
i = 0
first_guess_bias_threshold = 5
max_iter = 20


def calc_first_guess_bias(guessed_bed):
    with torch.no_grad():
        bed = torch.tensor(guessed_bed, dtype=torch.float,
                           requires_grad=False)
        init_ice_thick = start_surf - bed
        model = Upstream2D(bed, dx=case.dx, mb_model=mb, y0=y_spinup_end,
                           glen_a=cfg.PARAMS['glen_a'], ice_thick_filter=None,
                           init_ice_thick=init_ice_thick)
        model.run_until(y_end)
        s = model.surface_h.detach().numpy()
        bias = ((s - reference_surf) * ice_mask).sum() / ice_mask.sum()
    return bias, s


first_guess_bias, s0 = calc_first_guess_bias(b0)

while i < max_iter and abs(first_guess_bias) >= first_guess_bias_threshold:
    f -= df * np.sign(first_guess_bias)
    df *= 0.7

    b0 = first_guess(reference_surf, ice_mask, case.dx, factor=f)
    first_guess_bias, s0 = calc_first_guess_bias(b0)
'''

lambs = torch.zeros(9)
lambs[3] = 30.

test = LCurveTest(case, y0, y_spinup_end, y_end)
test.lambdas = lambs
test.basedir = '/data/philipp/tests/first_guess/iterative/'

for f in [1, 0.8]:
    fg = first_guess(test.reference_surf.detach().numpy(),
                     test.ice_mask.detach().numpy(), case.dx, factor=f)
    test.first_guess = torch.tensor(fg, dtype=torch.float)
    test.maxiter = 30
    test.run_minimize()

print('end')