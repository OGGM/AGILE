import numpy as np

from combine1d.core.first_guess import get_first_guess


def test_get_first_guess(data_logger):
    first_guess = get_first_guess(data_logger)

    fl = data_logger.flowline_init
    for con_var in data_logger.control_vars:
        if con_var in ['w0_m', 'lambdas']:
            prefix = '_'
        else:
            prefix = ''
        first_guess_ind = data_logger.parameter_indices[con_var]
        assert np.allclose(first_guess[first_guess_ind],
                           getattr(fl, prefix + con_var)[data_logger.ice_mask])

    assert not np.isnan(np.sum(first_guess))
    assert len(first_guess) == data_logger.len_unknown_parameter

