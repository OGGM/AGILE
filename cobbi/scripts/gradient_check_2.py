from cobbi.utils import test_cases
from cobbi.inversion import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

#case = test_cases.mayan_ice_cap
case = test_cases.Blafell
case.dx = 600
case.smooth_border_px = 4

y0 = 0
y_spinup_end = 2000
y_end = 2300
lamb1 = 0.
lamb2 = 0.
lamb3 = 0.
lamb4 = 0.


start_surf, reference_surf, ice_mask, mb, bed_2d = spin_up(case, y_spinup_end,
                                                           y_end)

cost_func = create_cost_function(start_surf, reference_surf,
                                           ice_mask, case.dx, mb,
                                           y_spinup_end, y_end,
                                           lamb1=lamb1, lamb2=lamb2,
                                           lamb3=lamb3, lamb4=lamb4)

def cost_function(bed):
    init_ice_thick = start_surf - bed
    model = Upstream2D(bed, dx=case.dx, mb_model=mb, y0=y_spinup_end,
                       glen_a=cfg.PARAMS['glen_a'], ice_thick_filter=None,
                       init_ice_thick=init_ice_thick)
    model.run_until(y_end)
    s = model.surface_h
    # cost = ((surface_to_match - s) * ice_mask).pow(2).sum()
    cost = (reference_surf - s).pow(2).sum()
    return cost


bed_0 = get_first_guess(reference_surf, ice_mask, case.dx)
b_0 = torch.tensor(bed_0.detach().numpy(), dtype=torch.float, requires_grad=True)
cost_torch = cost_function(b_0)
cost_torch.backward()
with torch.no_grad():
    grad_torch = b_0.grad
    grad_torch = grad_torch.detach().numpy().reshape(start_surf.shape).astype(np.float64)
    b_0.grad.zero_()
    cost_torch = cost_torch.detach().numpy().astype(np.float64)


max_grad_val = np.max(np.abs(grad_torch))


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


def plot_gradient(data, max_val, title, filename):
    plt.figure()
    im = plt.imshow(data, norm=MidpointNormalize(midpoint=0., vmin=-max_val, vmax=max_val), cmap='RdBu')
    plt.title(title)
    plt.colorbar(im)
    plt.savefig('/data/philipp/gradient_check/figs/' + filename)
    plt.clf()


filename_prefix = '{:s}_dx{:d}_ys{:d}_ye{:d}'.format(case.name, case.dx, y_spinup_end, y_end)
title_suffix = '{:s} with dx={:d} from year {:d} to {:d}'.format(case.name, case.dx, y_spinup_end, y_end)
plot_gradient(grad_torch, max_grad_val, 'Pytorch Gradient; ' + title_suffix, filename_prefix + 'grad_torch.png')


# #################################################
# finite difference gradient
dbs = [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]
rel_diff1 = []
rel_diff2 = []
abs_diff = []
rmse = []
grads_fin_dif = []

indices_to_check = np.argwhere(ice_mask.numpy())
b_0 = torch.tensor(bed_0, dtype=torch.float, requires_grad=False)

for db in dbs:
    fin_dif_grad = np.zeros(bed_2d.shape)
    with torch.no_grad():
        J_ref = cost_function(b_0)
        for ind in indices_to_check:
            b_1 = b_0.clone()
            b_1[ind[0], ind[1]] = b_1[ind[0], ind[1]] + db
            J_dev = cost_function(b_1)
            fin_dif_grad[ind[0], ind[1]] = J_dev - J_ref

        fin_dif_grad = fin_dif_grad / db
    rel_diff1.append((fin_dif_grad - grad_torch) / grad_torch)
    rel_diff2.append((fin_dif_grad - grad_torch) / fin_dif_grad)
    abs_diff.append(fin_dif_grad - grad_torch)
    rmse.append(np.mean((fin_dif_grad - grad_torch)**2))
    grads_fin_dif.append(fin_dif_grad)


max_grad_val = np.max(np.abs(grad_torch))
max_grad_val = 1000 #np.max([np.max(np.abs(grads_fin_dif)), max_grad_val])

filename_prefix = '{:s}_dx{:d}_ys{:d}_ye{:d}'.format(case.name, case.dx, y_spinup_end, y_end)
title_suffix = '{:s} with dx={:d} from year {:d} to {:d}'.format(case.name, case.dx, y_spinup_end, y_end)
plot_gradient(grad_torch, max_grad_val, 'Pytorch Gradient; ' + title_suffix, filename_prefix + 'grad_torch.png')

max_abs_diff_val = 100 #np.max(np.abs(abs_diff))

i = 0
for i in range(len(dbs)):
    plot_gradient(grads_fin_dif[i], max_grad_val, 'FinDiff Gradient, db={:g}m; '.format(dbs[i]),
                  filename_prefix + 'grad_fin_diff_db{:07.4f}.png'.format(dbs[i]))
    plot_gradient(abs_diff[i], max_abs_diff_val, 'Absolute deviation, db={:g}m; '.format(dbs[i]),
                  filename_prefix + 'abs_dev_db{:07.4f}.png'.format(dbs[i]))

ticks_at = np.arange(len(dbs))
ticklabels = np.array(dbs).astype(np.str).tolist()
plt.figure()
plt.plot(ticks_at, rmse)
plt.xlabel('db (m)')
plt.ylabel('RMSE: torch vs finite difference')
plt.xticks(ticks_at, ticklabels)
plt.title('RMSE')
plt.savefig('/data/philipp/gradient_check/figs/' + filename_prefix + 'rmse.pdf')


print('end')

