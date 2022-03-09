import numpy as np

from combine1d.core.first_guess import get_first_guess, get_first_guess_surface_h


def test_get_first_guess(data_logger):
    first_guess = get_first_guess(data_logger)

    fl = data_logger.flowline_init
    for con_var in data_logger.control_vars:
        if con_var in ['w0_m', 'lambdas']:
            prefix = '_'
            mask = (data_logger.is_trapezoid & data_logger.ice_mask)
            first_guess_ind = data_logger.parameter_indices[con_var]
            assert np.allclose(first_guess[first_guess_ind],
                               getattr(fl, prefix + con_var)[mask])
        elif con_var in ['bed_h']:
            prefix = ''
            mask = data_logger.ice_mask
            first_guess_ind = data_logger.parameter_indices[con_var]
            assert np.allclose(first_guess[first_guess_ind],
                               getattr(fl, prefix + con_var)[mask])
        elif con_var in ['area_bed_h']:
            prefix = ''
            mask = data_logger.ice_mask
            first_guess_ind = data_logger.parameter_indices[con_var]
            assert np.allclose(first_guess[first_guess_ind],
                               fl.bed_h[mask] * fl.widths_m[mask] /
                               np.mean(fl.widths_m[mask]))
        elif con_var in ['surface_h']:
            first_guess_ind = data_logger.parameter_indices[con_var]
            assert np.allclose(first_guess[first_guess_ind],
                               get_first_guess_surface_h(data_logger))
        elif con_var in ['height_shift_spinup']:
            first_guess_ind = data_logger.parameter_indices[con_var]
            assert np.allclose(
                first_guess[first_guess_ind],
                data_logger.spinup_options['height_shift']['mb_model']['fg_height_shift'])
        else:
            raise NotImplementedError

    assert not np.isnan(np.sum(first_guess))
    assert len(first_guess) == data_logger.len_unknown_parameter

