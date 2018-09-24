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
    import LinearMassBalance
#from cobbi.inversion import first_guess

extended_logging_in_cost_function = True
costs = []
grads = []
beds = []
surfs = []

def first_thickness_guess(obs_surf, ice_mask, map_dx, smoothing=None):
    # TODO: multiple first guess possibilites imaginable, make them available?
    bed_h = first_guess(obs_surf, ice_mask, map_dx)
    # TODO: a lot of smoothing
    return bed_h

def first_guess(surf, ice_mask, map_dx, slope_cutoff_angle=5.0):
    h_difference = surf[ice_mask].max() - surf[ice_mask].min()
    h_difference = min(1.6, h_difference / 1000.)
    # https://doi.org/10.1080/13658816.2011.627859
    tau = 0.005 + 1.598 * h_difference - 0.435 * h_difference ** 2
    if h_difference == 1.6:
        tau = 1.5
    tau = tau * 1e5

    # f = 0.8  # valley glaciers only
    f = 1

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
    smooth_dem_borders(gdir, px_count=case.smooth_border_px)
    ds = salem.GeoTiff(gdir.get_filepath('dem', filesuffix='_smooth_border'))
    bed_2d = torch.tensor(ds.get_vardata(), dtype=torch.float, requires_grad=False)

    mb = LinearMassBalance(case.ela_h, grad=case.mb_grad)
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
    bed_0 = first_guess(reference_surf.clone().detach().numpy(),
                        ice_mask.clone().detach().numpy(),
                        dx)
    return torch.tensor(bed_0, dtype=torch.float, requires_grad=False)


def create_cost_function(spinup_surface, surface_to_match, ice_mask,
                         dx, mb, y_spinup_end, y_end, lamb1=0.,
                         lamb2=0., lamb3=0., lamb4=0., lamb5=0., lamb6=0.,
                         lamb7=0., lamb8=0):

    ice_mask = torch.tensor(ice_mask, dtype=torch.float)
    n_ice_mask = float(ice_mask.sum())
    n_grid = float(ice_mask.numel())
    conv_filter = torch.ones((1, 1, 3, 3))


    # define cost_function
    def cost_function(b):
        bed = torch.tensor(b.reshape(spinup_surface.shape), dtype=torch.float,
                           requires_grad=True)
        init_ice_thick = spinup_surface - bed
        model = Upstream2D(bed, dx=dx, mb_model=mb, y0=y_spinup_end,
                           glen_a=cfg.PARAMS['glen_a'], ice_thick_filter=None,
                           init_ice_thick=init_ice_thick)
        model.run_until(y_end)
        s = model.surface_h
        #cost = ((surface_to_match - s) * ice_mask).pow(2).sum()

        ice_region = (s - bed) > 0

        inner_mask = torch.zeros(spinup_surface.shape)
        inner_mask[1:-1, 1:-1] = torch.conv2d(
            torch.tensor(ice_region.unsqueeze(0).unsqueeze(0),
                         dtype=torch.float),
            conv_filter) == 9
        n_inner_mask = inner_mask.sum()

        cost = (surface_to_match - s).pow(2).sum() \
               / ice_region.sum().type(dtype=torch.float)
        if lamb1 != 0:
            it = s - bed
            dit_dx = (it[:, :-2] - it[:, 2:]) / (2. * dx)
            dit_dy = (it[:-2, :] - it[2:, :]) / (2. * dx)
            dit_dx = dit_dx * inner_mask[:, 1:-1]
            dit_dy = dit_dy * inner_mask[1:-1, :]
            cost = cost + lamb1 * (
                    (dit_dx.pow(2).sum() + dit_dy.pow(2).sum()) / n_inner_mask)

        if lamb2 != 0:
            db_dx = (bed[:, :-2] - bed[:, 2:]) / dx
            db_dy = (bed[:-2, :] - bed[2:, :]) / dx
            db_dx = db_dx * inner_mask[:, 1:-1]
            db_dy = db_dy * inner_mask[1:-1, :]
            cost = cost + lamb2 * (
                    (db_dx.pow(2).sum() + db_dy.pow(2).sum())/ n_inner_mask )

        if lamb3 != 0:
            # penalizes ice thickness, where ice thickness should be 0
            cost = cost + \
                   lamb3 * (((s - bed) * (1. - ice_mask)).pow(2).sum()
                            / (n_grid - n_ice_mask))

        if lamb4 != 0:
            # penalizes bed != reference surf where we know about the bed
            # height because of ice thickness == 0
            cost = cost + \
                   lamb4 * \
                   (((surface_to_match - bed) * (1. - ice_mask)).pow(2).sum()
                    / (n_grid - n_ice_mask))

        if lamb5 != 0:
            # penalize high curvature of ice thickness (in glacier bounds)
            it = s - bed
            ddit_dx = (it[:, :-2] + it[:, 2:] - 2*it[:, 1:-1]) / dx**2
            ddit_dy = (it[:-2, :] + it[2:, :] - 2*it[1:-1, :]) / dx**2
            ddit_dx = ddit_dx * inner_mask[:, 1:-1]
            ddit_dy = ddit_dy * inner_mask[1:-1, :]
            cost = cost + lamb5 * (
                    (ddit_dx.pow(2).sum() + ddit_dy.pow(2).sum())
                    / n_inner_mask)
        
        if lamb6 != 0:
            # penalize high curvature of bed (in glacier bounds)
            ddb_dx = (bed[:, :-2] + bed[:, 2:] - 2*bed[:, 1:-1]) / dx**2
            ddb_dy = (bed[:-2, :] + bed[2:, :] - 2*bed[1:-1, :]) / dx**2
            ddb_dx = ddb_dx * inner_mask[:, 1:-1]
            ddb_dy = ddb_dy * inner_mask[1:-1, :]
            cost = cost + lamb6 * (
                    (ddb_dx.pow(2).sum() + ddb_dy.pow(2).sum()) / n_inner_mask)

        if lamb7 != 0:
            # penalize high curvature of bed exactly at boundary pixels of
            # glacier for a smooth transition from glacier free to glacier
            ddb_dx = (bed[:, :-2] + bed[:, 2:] - 2 * bed[:, 1:-1]) / dx ** 2
            ddb_dy = (bed[:-2, :] + bed[2:, :] - 2 * bed[1:-1, :]) / dx ** 2
            ddb_dx = ddb_dx * (ice_region - inner_mask)[:, 1:-1]
            ddb_dy = ddb_dy * (ice_region - inner_mask)[1:-1, :]
            cost = cost + lamb7 * (
                    (ddb_dx.pow(2).sum() + ddb_dy.pow(2).sum())
                    / (ice_region - inner_mask)[1:-1, 1:-1].sum())

        if lamb8 != 0:
            # penalizes not matching ice masks between reference and modelled
            # in comparison to lamb3 independent of icethickness at not matching
            # grid cells
            cost = cost + \
                lamb8 * (inner_mask - ice_mask).pow(2).sum() / n_grid


        cost.backward()
        with torch.no_grad():
            grad = bed.grad
            grad = grad.detach().numpy().flatten().astype(np.float64)
            bed.grad.zero_()
            cost = cost.detach().numpy().astype(np.float64)

        if extended_logging_in_cost_function:
            costs.append(cost)
            grads.append(grad)
            beds.append(bed.detach().numpy())
            surfs.append(s.detach().numpy())

        return cost, grad

    return cost_function