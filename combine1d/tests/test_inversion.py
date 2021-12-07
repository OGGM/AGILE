import numpy as np
import pytest

from combine1d.core.first_guess import get_first_guess
from combine1d.core.inversion import (prepare_for_combine_inversion,
                                      get_control_var_bounds)
from combine1d.core.data_logging import initialise_DataLogger
from combine1d.core.cost_function import create_cost_fct


class TestInversion:
    def test_default_initialisation(self, hef_gdir):
        # check if error is thrown if gdir is not prepared
        with pytest.raises(AttributeError, match='inversion_input_combine file not found!'):
            initialise_DataLogger(hef_gdir)

        # add default inversion_input_combine and try again
        prepare_for_combine_inversion(hef_gdir, inversion_settings=None, filesuffix='_combine')
        data_logger = initialise_DataLogger(hef_gdir, inversion_input_filesuffix='_combine')

        cost_fct = create_cost_fct(data_logger)

        first_guess = get_first_guess(data_logger)

        cost, grad = cost_fct(first_guess)
        assert cost.size == 1
        assert type(cost) == np.ndarray
        assert type(grad) == np.ndarray

    def test_get_control_var_bounds(self, data_logger):
        create_cost_fct(data_logger)

        first_guess = get_first_guess(data_logger)

        bounds = get_control_var_bounds(data_logger)

        assert len(bounds) == data_logger.len_unknown_parameter
        for i, val in enumerate(first_guess):
            assert bounds[i] != 0
            if bounds[i][0] is not None:
                assert val >= bounds[i][0]
            if bounds[i][1] is not None:
                assert val <= bounds[i][1]
