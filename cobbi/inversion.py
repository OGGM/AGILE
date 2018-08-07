import numpy as np
from oggm.cfg import RHO, G

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