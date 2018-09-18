from cobbi.utils import test_cases
from cobbi.inversion import *
#from cobbi import inversion
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

case = test_cases.mayan_ice_cap
case = test_cases.blafell
case.dx = 600
case.smooth_border_px = 5

y0 = 0
y_spinup_end = 2000
y_end = 2700
lamb1 = 1. # Problem with borders of glacier -> limit this regularization to
#  inner parts of glacier
lamb2 = 0.
lamb3 = 5
lamb4 = 1.e1


start_surf, reference_surf, ice_mask, mb, bed_2d = spin_up(case, y_spinup_end,
                                                           y_end)

cost_func = create_cost_function_true_surf(start_surf, reference_surf,
                                           ice_mask, case.dx, mb,
                                           y_spinup_end, y_end,
                                           lamb1=lamb1, lamb2=lamb2,
                                           lamb3=lamb3, lamb4=lamb4)

bed_0 = get_first_guess(reference_surf, ice_mask, case.dx)

extended_logging_in_cost_function = True


def iteration_info_callback(x0):
    i = len(costs) - 1
    if i >= 0:
        b = bed_2d.detach().numpy()
        im = ice_mask.detach().numpy()
        print('--------------------------------------------------')
        print('Iteration: {:d}'.format(i))
        print('Cost: {:g}'.format(costs[i]))
        print('Bed RMSE: ', RMSE(beds[i] * im, b * im))
        print('Bed Max_diff: ', np.max(np.abs(beds[i] * im - b * im)))
        print('Surface RMSE: ', RMSE(surfs[i] * im,
                                     reference_surf.detach().numpy() * im))
        print('Surface Max_diff: ',
              np.max(np.abs(surfs[i] * im
                            - reference_surf.detach().numpy() * im)))


res = minimize(fun=cost_func,
               x0=bed_0.detach().numpy().astype(np.float64).flatten(),
               method='L-BFGS-B',
               jac=True,
               options={'maxiter': 120,
                        #'ftol': 1e-4,
                        #'xtol': 5e0,
                        'maxcor': 40,
                        'disp': True},
               callback=iteration_info_callback
               )

plt.figure();plt.imshow(res.x.reshape(bed_2d.shape)); plt.show()
plt.figure();plt.imshow(bed_0.detach().numpy()); plt.show()
plt.figure();plt.imshow(bed_2d.detach().numpy()); plt.show()
plt.figure();plt.imshow(bed_2d.detach().numpy() - res.x.reshape(bed_2d.shape));
plt.show()

print('end')