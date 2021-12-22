import numpy as np


def get_first_guess(data_logger):
    fl = data_logger.flowline_init
    parameter_indices = data_logger.parameter_indices
    ice_mask = data_logger.ice_mask
    is_trapezoid = data_logger.is_trapezoid

    first_guess = np.empty(data_logger.len_unknown_parameter)
    first_guess[:] = np.nan

    for ind in parameter_indices:
        if ind in ['bed_h', 'surface_h']:
            prefix = ''
            parameter_mask = ice_mask
        elif ind in ['w0_m', 'lambdas']:
            prefix = '_'
            parameter_mask = is_trapezoid
        else:
            raise NotImplementedError(f'{ind} is not implemented!')

        first_guess[parameter_indices[ind]] = getattr(fl, prefix + ind)[parameter_mask]

    return first_guess
