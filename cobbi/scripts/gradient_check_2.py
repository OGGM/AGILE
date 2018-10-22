from cobbi.utils.lcurve import create_logspaced_lamb_sequence
import numpy as np
import torch
from cobbi.utils import test_cases
from cobbi.utils.optimization import *
import matplotlib.pyplot as plt
import scipy.optimize
# from itertools import product

np.random.seed(0)

lambs = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])

case = test_cases.Giluwe
case.dx = 1000
#case.dx = 400
#case.smooth_border_px = 2

y0 = 0
y_spinup_end = 2000
y_end = 2100

solver = 'BFGS'
minimize_options = {
    'maxiter': 50,
    #'ftol': 1e-3,
    #'xtol': 1e-30,
    #'gtol': 1e-4,
    'disp': True}

test = LCurveTest(case, y0, y_spinup_end, y_end,
                  minimize_options=minimize_options, solver=solver)
f = 1.
sca = 5.
test.basedir = '/data/philipp/gradient_check/test_giluwe/2/'
if not os.path.exists(test.basedir):
    os.makedirs(test.basedir, exist_ok=True)
#test.maxiter = 30
#test.optimization_counter =
# Disturb surface:
#test.add_surface_noise(std=5) #, zoom=1.5)
#test.optimization_counter = 200

#test.lambdas = torch.tensor(lambs, dtype=torch.float, requires_grad=False)
test.lambdas = torch.zeros(9, dtype=torch.float, requires_grad=False)


def create_cost_function(spinup_surface, surface_to_match, glacier_mask,
                         dx, mb, y_spinup_end, y_end, lambs):
    n_ice_mask = float(glacier_mask.sum())
    n_grid = float(glacier_mask.size)

    # define cost_function
    def cost_function(bed):
        lambdas = lambs.detach().clone()
        ice_mask = torch.tensor(glacier_mask, dtype=torch.float,
                                requires_grad=False)
        start_surf = torch.tensor(spinup_surface, dtype=torch.float,
                                  requires_grad=False)
        reference_surf = torch.tensor(surface_to_match, dtype=torch.float,
                                      requires_grad=False)
        conv_filter = torch.ones((1, 1, 3, 3), requires_grad=False)

        #bed = torch.tensor(b.reshape(start_surf.shape), dtype=torch.float,
        #                   requires_grad=with_grad)
        init_ice_thick = (start_surf - bed) * ice_mask
        model = Upstream2D(bed, dx=dx, mb_model=mb, y0=y_spinup_end,
                           glen_a=cfg.PARAMS['glen_a'], ice_thick_filter=None,
                           init_ice_thick=init_ice_thick)
        model.run_until(y_end)
        s = model.surface_h
        # cost = ((surface_to_match - s) * ice_mask).pow(2).sum()

        ice_region = (s - bed) > 0
        ice_region = ice_region.type(dtype=torch.float)

        inner_mask = torch.zeros(start_surf.shape)
        inner_mask[1:-1, 1:-1] = torch.conv2d(
            ice_region.unsqueeze(0).unsqueeze(0),
            conv_filter) == 9

        c = get_costs(lambdas, reference_surf, s, bed, n_grid, ice_region,
                      inner_mask, ice_mask, n_ice_mask, dx)
        cost = c.sum()

        #if with_grad:
        #    cost.backward(retain_graph=False)
        #with torch.no_grad():
        #    if with_grad:
        #        grad = bed.grad
        #        grad_val = grad.detach().numpy().flatten().astype(np.float64)
        #        bed.grad.zero_()
        #    cost_val = cost.detach().numpy().astype(np.float64)
        return cost

    return cost_function

def get_costs(lambs, surface_to_match, s, bed, n_grid, ice_region, inner_mask,
              ice_mask, n_ice_mask, dx):
    n_inner_mask = inner_mask.sum()
    cost = torch.zeros(10)

    cost[-1] = (surface_to_match - s).pow(2).sum() \
               / ice_region.sum().type(dtype=torch.float)

    if lambs[0] != 0:
        # penalize large derivatives of ice thickness
        it = s - bed
        dit_dx = (it[:, :-2] - it[:, 2:]) / (2. * dx)
        dit_dy = (it[:-2, :] - it[2:, :]) / (2. * dx)
        dit_dx = dit_dx * inner_mask[:, 1:-1]
        dit_dy = dit_dy * inner_mask[1:-1, :]
        cost[0] = lambs[0] * (
                (dit_dx.pow(2).sum() + dit_dy.pow(2).sum()) / n_inner_mask)

    if lambs[1] != 0:
        # penalize large derivatives of bed inside glacier bounds
        db_dx = (bed[:, :-2] - bed[:, 2:]) / dx
        db_dy = (bed[:-2, :] - bed[2:, :]) / dx
        db_dx = db_dx * inner_mask[:, 1:-1]
        db_dy = db_dy * inner_mask[1:-1, :]
        cost[1] = lambs[1] * (
                (db_dx.pow(2).sum() + db_dy.pow(2).sum()) / n_inner_mask)

    if lambs[2] != 0:
        # penalizes ice thickness, where ice thickness should be 0
        cost[2] = lambs[2] * (((s - bed) * (1. - ice_mask)).pow(2).sum()
                              / (n_grid - n_ice_mask))

    if lambs[3] != 0:
        # penalizes bed != reference surf where we know about the bed
        # height because of ice thickness == 0
        cost[3] = lambs[3] * \
                  (((surface_to_match - bed) * (1. - ice_mask)).pow(2).sum()
                   / (n_grid - n_ice_mask))

    if lambs[4] != 0:
        # penalize high curvature of ice thickness (in glacier bounds)
        it = s - bed
        ddit_dx = (it[:, :-2] + it[:, 2:] - 2 * it[:, 1:-1]) / dx ** 2
        ddit_dy = (it[:-2, :] + it[2:, :] - 2 * it[1:-1, :]) / dx ** 2
        ddit_dx = ddit_dx * inner_mask[:, 1:-1]
        ddit_dy = ddit_dy * inner_mask[1:-1, :]
        cost[4] = lambs[4] * ((ddit_dx.pow(2).sum() + ddit_dy.pow(2).sum())
                              / n_inner_mask)

    if lambs[5] != 0:
        # penalize high curvature of bed (in glacier bounds)
        ddb_dx = (bed[:, :-2] + bed[:, 2:] - 2 * bed[:, 1:-1]) / dx ** 2
        ddb_dy = (bed[:-2, :] + bed[2:, :] - 2 * bed[1:-1, :]) / dx ** 2
        ddb_dx = ddb_dx * inner_mask[:, 1:-1]
        ddb_dy = ddb_dy * inner_mask[1:-1, :]
        cost[5] = lambs[5] * ((ddb_dx.pow(2).sum() + ddb_dy.pow(2).sum())
                              / n_inner_mask)

    if lambs[6] != 0:
        # penalize high curvature of bed exactly at boundary pixels of
        # glacier for a smooth transition from glacier-free to glacier
        ddb_dx = (bed[:, :-2] + bed[:, 2:] - 2 * bed[:, 1:-1]) / dx ** 2
        ddb_dy = (bed[:-2, :] + bed[2:, :] - 2 * bed[1:-1, :]) / dx ** 2
        ddb_dx = ddb_dx * (ice_region - inner_mask)[:, 1:-1]
        ddb_dy = ddb_dy * (ice_region - inner_mask)[1:-1, :]
        cost[6] = lambs[6] * ((ddb_dx.pow(2).sum() + ddb_dy.pow(2).sum())
                              / (ice_region - inner_mask)[1:-1, 1:-1].sum())

    if lambs[7] != 0:
        # penalize high curvature of surface inside glacier
        dds_dx = (s[:, :-2] + s[:, 2:] - 2 * s[:, 1:-1]) / dx ** 2
        dds_dy = (s[:-2, :] + s[2:, :] - 2 * s[1:-1, :]) / dx ** 2
        dds_dx = dds_dx * inner_mask[:, 1:-1]
        dds_dy = dds_dy * inner_mask[1:-1, :]
        cost[7] = lambs[7] * ((dds_dx.pow(2).sum() + dds_dy.pow(2).sum())
                              / n_inner_mask)

    if lambs[8] != 0:
        lmsd = LocalMeanSquaredDifference.apply
        cost[8] = lambs[8] * lmsd(s, surface_to_match, ice_region, ice_mask,
                                  bed)
    # if lambs[7] != 0:
    # penalizes not matching ice masks between reference and modelled
    # in comparison to lamb3 independent of icethickness at not matching
    # grid cells
    #    cost[7] = lambs[7]* (inner_mask - ice_mask).pow(2).sum() / n_grid

    # if lambs[8] != 0:
    # penalizes differences in surface height with power of 4 to put
    # more emphasize on larger deviations
    #    cost[8] = lambs[8] * ((surface_to_match - s).pow(2).sum()
    #                          / ice_region.sum().type(dtype=torch.float))

    return cost

cost_fun = create_cost_function(test.start_surf, test.reference_surf, test.ice_mask,
                                case.dx, test.mb, test.y_spinup_end, test.y_end,
                                test.lambdas)
bed = torch.tensor(test.first_guess, dtype=torch.float, requires_grad=True)

cost = cost_fun(bed)
cost.backward()
torch_grad = bed.grad.detach().numpy().copy()

np.save(test.basedir + 'torch_grad', torch_grad)
db = 0.01

with open(test.basedir + 'settings.txt', 'w') as f:
            f.write(test.get_setting_as_string() +
                    '\ndb = {:g}.\n'.format(db))


def cost_fun2(b):
    with torch.no_grad():
        bed = torch.tensor(b.reshape(test.start_surf.shape), dtype=torch.float,
                           requires_grad=False)
        return cost_fun(bed).detach().numpy().copy()

b = np.array(test.first_guess, dtype=np.float32).flatten()
fin_dif_grad = scipy.optimize.approx_fprime(b, cost_fun2, db)
fin_dif_grad = fin_dif_grad.reshape(test.first_guess.shape)
np.save(test.basedir + 'fin_dif_grad', fin_dif_grad)


class MidpointNormalize(colors.Normalize):
    # see: https://matplotlib.org/users/colormapnorms.html#custom-normalization-two-linear-ranges
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_gradient(gradient, title, cbar_min, cbar_max, label=None):
    # self.plot_image(self.grads[i].reshape(ref_shape), cbar_min_max,
    #                'Gradient #{:d}'.format(i),
    #                basedir + 'grad{:d}.png'.format(i))

    plt.figure()
    im = plt.imshow(gradient,
                    norm=MidpointNormalize(midpoint=0.,
                                           vmin=cbar_min,
                                           vmax=cbar_max),
                    cmap='RdBu_r')
    plt.title(title)
    cbar = plt.colorbar(im)
    if label is None:
        cbar.set_label('Gradient of cost-function (m$^{-1}$)')
    else:
        cbar.set_label(label)
    plt.show()

cbar_min = np.min([fin_dif_grad, torch_grad])
cbar_max = np.max([fin_dif_grad, torch_grad])
plot_gradient(torch_grad, 'PyTorch Gradient,\n{:s}, dx={:d}m, yr {:d}-{:d}'.format(case.name, case.dx, y_spinup_end, y_end), cbar_min, cbar_max)
plot_gradient(fin_dif_grad, 'Finite Difference Gradient db={:g},\n{:s}, dx={:d}m, yr {:d}-{:d}'.format(db, case.name, case.dx, y_spinup_end, y_end), cbar_min, cbar_max)

difference = (torch_grad - fin_dif_grad)
cbar_min = difference.min()
cbar_max = difference.max()
plot_gradient(difference, '(Pytorch - Finite Difference) Gradient db={:g},\n{:s}, dx={:d}m, yr {:d}-{:d}'.format(db, case.name, case.dx, y_spinup_end, y_end), cbar_min, cbar_max, label='$\Delta$ Gradient of cost-function (m$^{-1}$)')

difference = (torch_grad / torch_grad.max() - fin_dif_grad / fin_dif_grad.max())
cbar_min = difference.min()
cbar_max = difference.max()
plot_gradient(difference, 'scaled (Pytorch - Finite Difference) Gradient db={:g},\n{:s}, dx={:d}m, yr {:d}-{:d}'.format(db, case.name, case.dx, y_spinup_end, y_end), cbar_min, cbar_max, label='Difference of Scaled Gradients')

difference = ((torch_grad - fin_dif_grad ) / fin_dif_grad)
cbar_min = -1 #np.nanmin(np.ma.masked_invalid(difference))
cbar_max = 1 #np.nanmax(np.ma.masked_invalid(difference))
plot_gradient(difference, 'Relative Difference (Pytorch - Finite Difference) Gradient db={:g},\n{:s}, dx={:d}m, yr {:d}-{:d}'.format(db, case.name, case.dx, y_spinup_end, y_end), cbar_min, cbar_max, label='Relative Gradient Difference')


print('end')