import numpy as np
import pytest

from combine1d.core.first_guess import get_first_guess
from combine1d.core.inversion import (prepare_for_combine_inversion,
                                      get_control_var_bounds,
                                      get_default_inversion_settings,
                                      combine_inversion)
from combine1d.core.data_logging import initialise_DataLogger
from combine1d.core.cost_function import create_cost_fct


pytestmark = pytest.mark.filterwarnings("ignore:<class 'combine1d.core.torch_interp1d.Interp1d'> "
                                        "should not be instantiated.:DeprecationWarning")


class TestInversion:
    def test_default_initialisation(self, hef_gdir):
        # check if error is thrown if gdir is not prepared
        with pytest.raises(AttributeError, match='inversion_input_combine file not found!'):
            initialise_DataLogger(hef_gdir)

        # add default inversion_input_combine and try again
        prepare_for_combine_inversion(hef_gdir, inversion_settings=None, filesuffix='_combine')
        data_logger = initialise_DataLogger(hef_gdir, inversion_input_filesuffix='_combine')

        assert len(data_logger.is_trapezoid) == (sum(data_logger.is_trapezoid) +
                                                 sum(data_logger.is_rectangular) +
                                                 sum(data_logger.is_parabolic))
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

    @pytest.mark.parametrize('control_vars', [['bed_h'], ['bed_h', 'w0_m'],
                                              ['area_bed_h'],
                                              'all'],
                             ids=['bed_h', 'bed_h & w0_m', 'area_bed_h', 'all'])
    @pytest.mark.parametrize('spinup_options', [None,
                                                {'surface_h': {'mb_model':
                                                                   {'type': 'constant',
                                                                    'years': np.array(
                                                                                [1980, 2000]),
                                                                    't_bias': -2}
                                                               }
                                                 },
                                                {'height_shift': {'mb_model':
                                                                      {'type': 'constant',
                                                                       'years': np.array(
                                                                                   [1980, 2000]),
                                                                       'fg_height_shift': 100}
                                                                  }
                                                 }
                                                ],
                             ids=['No_spinup', 'sfc_h_spinup', 'height_shift_spinup'])
    def test_combine_inversion(self, hef_gdir, control_vars, spinup_options,
                               all_supported_control_vars):
        # Final integration test that the inversion runs with no errors
        inversion_settings = get_default_inversion_settings(get_doc=False)
        inversion_settings['minimize_options'] = {'maxiter': 2,
                                                  'ftol': 1e-7,
                                                  'gtol': 1e-8,
                                                  'disp': False,
                                                  'maxcor': 50,
                                                  'maxls': 50,
                                                  'maxfun': 10}
        if control_vars == 'all':
            control_vars = all_supported_control_vars
        inversion_settings['control_vars'] = control_vars
        inversion_settings['spinup_options'] = spinup_options

        prepare_for_combine_inversion(hef_gdir, inversion_settings=inversion_settings,
                                      filesuffix='_combine')

        data_logger = combine_inversion(hef_gdir)

        assert data_logger.minimize_message is not None
        assert data_logger.minimize_status is not None
        assert data_logger.costs is not None
        assert data_logger.c_terms is not None
        assert data_logger.c_terms_description is not None
        assert data_logger.time_needed is not None
        assert data_logger.grads is not None
        assert data_logger.flowlines is not None
        assert data_logger.end_time is not None
        assert data_logger.known_parameters is not None
        assert data_logger.len_unknown_parameter is not None
        assert data_logger.parameter_indices is not None
        assert data_logger.unknown_parameters is not None
