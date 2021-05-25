import numpy as np


def mean_BIAS(a1, a2, ice_mask=None):
    if ice_mask is None:
        dev = a1 - a2
    else:
        dev = np.ma.masked_array(a1 - a2, mask=np.logical_not(ice_mask))
    return dev.mean()


def RMSE(a1, a2, ice_mask=None):
    if ice_mask is None:
        dev = a1 - a2
    else:
        dev = np.ma.masked_array(a1 - a2, mask=np.logical_not(ice_mask))
    return np.sqrt((dev**2).mean())


def max_dif(a1, a2):
    return np.max(np.abs(a1 - a2))


def magnitude(x):
    x = np.where(x != 0,
                 x, 1e-100)
    return np.floor(np.log10(np.abs(x))).astype(float)
