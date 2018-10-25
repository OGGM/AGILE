import numpy as np

import torch
torch.utils.backcompat.broadcast_warning.enabled = True
# import matplotlib.pyplot as plt

from cobbi.sia2d_adapted import Upstream2D
# from cobbi.sia2d_pytorch_full import Upstream2D as Up2D
# from cobbi.utils import test_cases
from oggm import cfg
from oggm import utils

from os import path
import salem

from cobbi.utils.synthetic_ice_caps import NonRGIGlacierDirectory
from cobbi.utils.synthetic_ice_caps \
    import define_nonrgi_glacier_region, smooth_dem_borders
from cobbi.utils.massbalance_pytorch \
    import ClippedLinearMassBalance
#from cobbi.inversion import first_guess

#extended_logging_in_cost_function = True
#costs = []
#grads = []
#beds = []
#surfs = []


def first_thickness_guess(obs_surf, ice_mask, map_dx, smoothing=None):
    # TODO: multiple first guess possibilites imaginable, make them available?
    bed_h = first_guess(obs_surf, ice_mask, map_dx)
    # TODO: a lot of smoothing
    return bed_h


def first_guess(surf, ice_mask, map_dx, slope_cutoff_angle=5.0, factor=1):
    glacier_surf = np.ma.masked_array(surf, np.logical_not(ice_mask))
    h_difference = glacier_surf.max() - glacier_surf.min()
    # h_difference = min(1.6, h_difference / 1000.)
    # https://doi.org/10.1080/13658816.2011.627859
    tau = 0.005 + 1.598 * h_difference - 0.435 * h_difference ** 2
    if h_difference >= 1.6:
        tau = 1.5
    tau = tau * 1e5

    # f = 0.8  # valley glaciers only
    f = factor

    gradients = np.gradient(surf, map_dx)
    gradients = np.sqrt(gradients[0] ** 2 + gradients[1] ** 2)

    sin_angle = np.sin(np.arctan(gradients))

    min_sin_angle = np.sin(np.deg2rad(slope_cutoff_angle))

    sin_angle[sin_angle < min_sin_angle] = min_sin_angle

    thick = tau / (cfg.PARAMS['ice_density'] * cfg.G * f * sin_angle)
    bed_h = surf - thick * ice_mask
    return bed_h


def moving_average(arr, n):
    '''
    Computes the moving average over the n elements starting with the current array element upwards

    :param arr: array, for which the moving average should be computed
    :param n: number of array elements considered for the average
    :return: array of averages

    TODO: replace by convolution
    '''
    ret = np.cumsum(arr[::-1], dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[::-1] / n

def RMSE(arr1, arr2):
    return np.sqrt(np.mean((arr1 - arr2)**2))


# Initialize OGGM and set up the default run parameters
cfg.initialize()


def spin_up(case, y_spinup_end, y_end):
    entity = {'min_x': case.extent[0, 0],
              'max_x': case.extent[1, 0],
              'min_y': case.extent[0, 1],
              'max_y': case.extent[1, 1],
              'name': case.name}

    # Local working directory (where OGGM will write its output)
    WORKING_DIR = path.join(path.expanduser('~'), 'tmp',
                            'synthetic_ice_cap', case.name)
    utils.mkdir(WORKING_DIR, reset=False)
    cfg.PATHS['working_dir'] = WORKING_DIR

    gdir = NonRGIGlacierDirectory(entity)
    define_nonrgi_glacier_region(gdir, dx=case.dx)
    smooth_dem_borders(gdir, px_count=case.smooth_border_px,
                       border_h=case.smooth_border_h)
    ds = salem.GeoTiff(gdir.get_filepath('dem', filesuffix='_smooth_border'))
    bed_2d = torch.tensor(ds.get_vardata(), dtype=torch.float, requires_grad=False)

    mb = ClippedLinearMassBalance(case.ela_h, grad=case.mb_grad,
                                  max_mb_alt=case.mb_max_alt,
                                  min_mb_alt=case.mb_min_alt)
    # Create glacier
    with torch.no_grad():
        reference_model = Upstream2D(bed_2d, dx=case.dx, mb_model=mb, y0=0,
                                     glen_a=cfg.PARAMS['glen_a'], ice_thick_filter=None)
        reference_model.run_until(y_spinup_end)
        start_surf = torch.tensor(reference_model.surface_h,
                                      dtype=torch.float, requires_grad=False)
        reference_model.run_until(y_end)
        reference_surf = torch.tensor(reference_model.surface_h,
                                      dtype=torch.float, requires_grad=False)
        ice_mask = (reference_surf > bed_2d)
    return [start_surf, reference_surf, ice_mask, mb, bed_2d]


def get_first_guess(reference_surf, ice_mask, dx):
    return first_guess(reference_surf, ice_mask,dx)


def create_cost_function(spinup_surface, surface_to_match, glacier_mask,
                         dx, mb, y_spinup_end, y_end, lambs, data_logger=None):

    n_ice_mask = float(glacier_mask.sum())
    n_grid = float(glacier_mask.size)

    # define cost_function
    def cost_function(b):
        lambdas = lambs.detach().clone()
        ice_mask = torch.tensor(glacier_mask, dtype=torch.float,
                                requires_grad=False)
        start_surf = torch.tensor(spinup_surface, dtype=torch.float,
                                  requires_grad=False)
        reference_surf = torch.tensor(surface_to_match, dtype=torch.float,
                                      requires_grad=False)
        conv_filter = torch.ones((1, 1, 3, 3), requires_grad=False)

        bed = torch.tensor(b.reshape(start_surf.shape), dtype=torch.float,
                           requires_grad=True)
        init_ice_thick = start_surf - bed
        model = Upstream2D(bed, dx=dx, mb_model=mb, y0=y_spinup_end,
                           glen_a=cfg.PARAMS['glen_a'], ice_thick_filter=None,
                           init_ice_thick=init_ice_thick)
        model.run_until(y_end)
        s = model.surface_h
        #cost = ((surface_to_match - s) * ice_mask).pow(2).sum()

        ice_region = (s - bed) > 0
        ice_region = ice_region.type(dtype=torch.float)

        inner_mask = torch.zeros(start_surf.shape)
        inner_mask[1:-1, 1:-1] = torch.conv2d(
            ice_region.unsqueeze(0).unsqueeze(0),
            conv_filter) == 9

        c = get_costs(lambdas, reference_surf, s, bed, n_grid, ice_region,
                      inner_mask, ice_mask, n_ice_mask, dx)
        cost = c.sum()

        cost.backward(retain_graph=False)
        with torch.no_grad():
            grad = bed.grad
            grad_val = grad.detach().numpy().flatten().astype(np.float64)
            bed.grad.zero_()
            cost_val = cost.detach().numpy().astype(np.float64)

        if data_logger is not None:
            data_logger.c_terms.append(c.detach().numpy())
            data_logger.costs.append(cost_val)
            data_logger.grads.append(grad_val)
            data_logger.beds.append(bed.detach().numpy())
            data_logger.surfs.append(s.detach().numpy())

        del ice_mask, bed, s, reference_surf, start_surf, c, cost, grad, lambdas

        return cost_val, grad_val

    return cost_function


def get_costs(lambs, surface_to_match, s, bed, n_grid, ice_region, inner_mask,
              ice_mask, n_ice_mask, dx):

    n_inner_mask = inner_mask.sum()
    cost = torch.zeros(10)
    lamb00 = 0.2
    margin = (ice_region - inner_mask)
    cost[-1] = ((surface_to_match - s) * (1. - margin)).pow(2).sum() \
               / inner_mask.sum().type(dtype=torch.float)
    cost[-1] = cost[-1] + lamb00 *\
               ((surface_to_match - s) * margin).pow(2).sum() \
               / margin.sum().type(dtype=torch.float)
    #cost[-1] = ((surface_to_match - s)).pow(2).sum() \
    #           / ice_region.sum().type(dtype=torch.float)

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
        cost[8] = lambs[8] * lmsd(s, surface_to_match, ice_region, ice_mask, bed)
    #if lambs[7] != 0:
        # penalizes not matching ice masks between reference and modelled
        # in comparison to lamb3 independent of icethickness at not matching
        # grid cells
    #    cost[7] = lambs[7]* (inner_mask - ice_mask).pow(2).sum() / n_grid

    #if lambs[8] != 0:
        # penalizes differences in surface height with power of 4 to put
        # more emphasize on larger deviations
    #    cost[8] = lambs[8] * ((surface_to_match - s).pow(2).sum()
    #                          / ice_region.sum().type(dtype=torch.float))

    return cost


class LocalMeanSquaredDifference(torch.autograd.Function):

    @staticmethod
    def forward(ctx, modelled_surf, surface_to_match, ice_region, ice_mask, bed):
        ctx.save_for_backward(modelled_surf, surface_to_match, ice_region, ice_mask, bed)
        msd = (modelled_surf - surface_to_match).pow(2).sum() / ice_region.sum().type(dtype=torch.float)
        return msd

    @staticmethod
    def backward(ctx, grad_output):
        modelled_surf, observed_surf, ice_region, ice_mask, bed = ctx.saved_tensors
        grad_modelled_surf = (modelled_surf - observed_surf) * ice_mask
        return None, None, None, None, grad_modelled_surf
