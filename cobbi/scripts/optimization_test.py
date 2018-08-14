from cobbi.utils import test_cases
from cobbi.inversion import *
from scipy.optimize import minimize

case = test_cases.arderin
y0 = 0
y_spinup_end = 900
y_end = 1000

start_surf, reference_surf, ice_mask, mb, bed_2d = spin_up(case, y_spinup_end,
                                                           y_end)

cost_func = create_cost_function(start_surf, reference_surf, case.dx, mb,
                                 y_spinup_end, y_end)

bed_0 = get_first_guess(reference_surf, ice_mask, case.dx)

res = minimize(fun=cost_func, x0=bed_0.detach().numpy().flatten(),
               #method='Newton-CG',
               jac=True,
               options={'maxiter': 100,
                        #'ftol': 1e1,
                        #'xtol': 5e0,
                        'disp': True}
               )

print('end')