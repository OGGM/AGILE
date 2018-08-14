import numpy as np
from oggm.cfg import RHO, G

import torch
torch.utils.backcompat.broadcast_warning.enabled = True
# import matplotlib.pyplot as plt

from cobbi.sia2d_pytorch_full import Upstream2D
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
from cobbi.inversion import first_guess



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

    thick = tau / (RHO * G * f * sin_angle)
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
    bed_2d = torch.tensor(ds.get_vardata(), dtype=torch.float)

    mb = LinearMassBalance(case.ela_h, grad=case.mb_grad)
    # Create glacier
    with torch.no_grad():
        reference_model = Upstream2D(bed_2d, dx=case.dx, mb_model=mb, y0=0,
                                     glen_a=cfg.A, ice_thick_filter=None)
        reference_model.run_until(y_spinup_end)
        start_surf = torch.tensor(reference_model.surface_h,
                                      dtype=torch.float, requires_grad=False)
        reference_model.run_until(y_end)
        reference_surf = torch.tensor(reference_model.surface_h,
                                      dtype=torch.float, requires_grad=False)
        ice_mask = (reference_surf > bed_2d)
    return [start_surf, reference_surf, ice_mask, mb, bed_2d]


def create_cost_function(spinup_surface, surface_to_match, dx, mb,
                         y_spinup_end, y_end):
    # define cost_function
    def cost_function(b):
        bed = torch.tensor(b.reshape(spinup_surface.shape), dtype=torch.float,
                           requires_grad=True)
        init_ice_thick = spinup_surface - bed
        model = Upstream2D(bed, dx=dx, mb_model=mb, y0=y_spinup_end,
                           glen_a=cfg.A, ice_thick_filter=None,
                           init_ice_thick=init_ice_thick)
        model.run_until(y_end)
        s = model.surface_h
        cost = (surface_to_match - s).pow(2).sum()
        cost.backward()
        with torch.no_grad():
            grad = bed.grad
            grad = grad.detach().numpy().flatten()
            bed.grad.zero_()
            cost = cost.detach().numpy()
        return cost, grad

    return cost_function

def get_first_guess(reference_surf, ice_mask, dx):
    bed_0 = first_guess(reference_surf.clone().detach().numpy(),
                        ice_mask.clone().detach().numpy(),
                        dx)
    return torch.tensor(bed_0, dtype=torch.float, requires_grad=True)
