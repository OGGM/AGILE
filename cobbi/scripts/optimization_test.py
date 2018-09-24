from cobbi.utils import test_cases
from cobbi.inversion import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

case = test_cases.kinabalu
case.dx = 600
case.smooth_border_px = 1

y0 = 0
y_spinup_end = 2000
y_end = 2500
lamb1 = 0.2
lamb2 = 0.2
lamb3 = 1.
lamb4 = 0.5
lamb5 = 1.
lamb6 = 1.
lamb7 = 0
lamb8 = 10

start_surf, reference_surf, ice_mask, mb, bed_2d = spin_up(case, y_spinup_end,
                                                           y_end)

cost_func = create_cost_function(start_surf, reference_surf,
                                 ice_mask, case.dx, mb, y_spinup_end, y_end,
                                 lamb1=lamb1, lamb2=lamb2, lamb3=lamb3,
                                 lamb4=lamb4, lamb5=lamb5, lamb6=lamb6,
                                 lamb7=lamb7, lamb8=lamb8)

bed_0 = get_first_guess(reference_surf, ice_mask, case.dx)

extended_logging_in_cost_function = True

actual_iteration_steps = []


def iteration_info_callback(x0):
    i = len(costs) - 1
    if i >= 0:
        actual_iteration_steps.append(i)
        b = bed_2d.detach().numpy()
        im = ice_mask.detach().numpy()
        print('--------------------------------------------------')
        print('Function Call: {:d}'.format(i))
        print('Iteration: {:d}'.format(actual_iteration_steps[i]))
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
               options={'maxiter': 100,
                        #'ftol': 1e-4,
                        #'xtol': 1e-8,
                        #'maxcor': 10,
                        'disp': True},
               callback=iteration_info_callback
               )

#bed = torch.tensor(bed_0.data.clone(), dtype=torch.float, requires_grad=True)
#optimizer = torch.optim.LBFGS([bed])

#fn = create_closure(optimizer, bed, start_surf, reference_surf,
#                    ice_mask, case.dx, mb, y_spinup_end, y_end,
#                    lamb1=lamb1, lamb2=lamb2, lamb3=lamb3, lamb4=lamb4)
#i = 0
#while i < 200 and fn().item() > 20:
#    i = i + 1
#    optimizer.step(fn)
#    iteration_info_callback(None)

#------------------------------------------------------------------------
#Plotting:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

plt.ioff()


basedir = '/data/philipp/tests/no2/'

# Filter all "exploratory" model runs to only get "real" iteration steps
my_grads = []
my_costs = []
my_beds = []
my_surfs = []
for i in actual_iteration_steps:
    my_grads.append(grads[i])
    my_costs.append(costs[i])
    my_beds.append(beds[i])
    my_surfs.append(surfs[i])
my_grads = np.array(my_grads)
my_costs = np.array(my_costs)
my_beds = np.array(my_beds)
my_surfs = np.array(my_surfs)

plt.figure()
plt.semilogy(my_costs)
plt.xlabel('function call #')
plt.ylabel('Cost')
plt.savefig(basedir + 'figs/cost.pdf')
plt.clf()


class MidpointNormalize(colors.Normalize):
    #see: https://matplotlib.org/users/colormapnorms.html#custom-normalization-two-linear-ranges
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_image(data, max_val, title, filename):
    plt.figure()
    im = plt.imshow(data, norm=MidpointNormalize(midpoint=0., vmin=-max_val, vmax=max_val), cmap='RdBu')
    plt.title(title)
    plt.colorbar(im)
    plt.savefig(basedir + 'figs/' + filename)
    plt.clf()

bed_rmses = []
surf_rmses = []

bed_differences = my_beds - bed_2d.detach().numpy()
bed_rmses = np.mean(bed_differences**2, axis=(1,2))
max_bed_difference = np.max(np.abs(bed_differences))
max_bed_differences = np.max(np.abs(bed_differences), axis=(1, 2))

surf_differences = my_surfs - reference_surf.detach().numpy()
surf_rmses = np.mean(surf_differences**2, axis=(1,2))
max_surf_difference = np.max(np.abs(surf_differences))
max_surf_differences = np.max(np.abs(surf_differences), axis=(1, 2))

plt.figure()
plt.semilogy(bed_rmses, label='Bed')
plt.semilogy(surf_rmses, label='Surface')
plt.xlabel('function call #')
plt.ylabel('RMSE')
plt.legend()
plt.savefig(basedir + 'figs/bed_surf_rmse.pdf')
plt.clf()

max_bed_difference = 50
max_surf_difference = 50

for i in range(my_beds.shape[0]):
    plot_image(bed_differences[i, :, :], max_bed_difference,
               'Bed difference #{:d}, |max_diff|={:g}'.format(
                   i, max_bed_differences[i]),
               'bed_diff{:d}.png'.format(i))
    plot_image(surf_differences[i, :, :], max_surf_difference,
               'Surface difference #{:d}, |max_diff|={:g}'.format(
                   i, max_surf_differences[i]),
               'surf_diff{:d}.png'.format(i))
    plot_image(my_grads[i, :].reshape(bed_2d.shape),
               np.max(np.abs(my_grads[i, :])),
               'Gradient #{:d}'.format(i),
               'grad{:d}.png'.format(i))

plt.ion()

print('end')