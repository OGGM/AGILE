import types

import numpy as np
import pytest
import torch

from combine1d.core.cost_function import (get_indices_for_unknown_parameters, get_known_parameters,
                                          create_cost_fct, initialise_flowline,
                                          initialise_mb_models, cost_fct)
from combine1d.core.first_guess import get_first_guess


class TestCreateCostFct:
    def test_get_known_parameters(self, data_logger_init):
        data_logger = data_logger_init
        known_parameters = get_known_parameters(data_logger)
        fl = data_logger.flowline_init
        ice_mask = data_logger.ice_mask

        for key in known_parameters.keys():
            assert key in data_logger.control_vars
            if key in ['w0_m', 'lambdas']:
                prefix = '_'
            else:
                prefix = ''
            assert np.allclose(known_parameters[key], getattr(fl, prefix + key)[~ice_mask],
                               equal_nan=True)

    def test_create_indices_for_unknown_parameters(self, data_logger_init):
        data_logger = data_logger_init
        parameter_indices = get_indices_for_unknown_parameters(data_logger)

        all_indices = np.array([])
        for key in parameter_indices.keys():
            assert key in data_logger.control_vars
            all_indices = np.append(all_indices, parameter_indices[key])

        # check that start with 0, every index is only present once, ends at len() - 1
        values, counts = np.unique(all_indices, return_counts=True)
        assert np.allclose(counts, 1)
        assert np.min(values) == 0
        assert np.max(values) == len(all_indices) - 1
        assert data_logger.len_unknown_parameter == len(all_indices)

    def test_create_cost_fct(self, data_logger_init):
        data_logger = data_logger_init
        # just integration test for the final creation of the cost function
        cost_fct = create_cost_fct(data_logger)
        assert isinstance(cost_fct, types.FunctionType)


class TestCostFct:
    @pytest.fixture(scope='function')
    def unknown_parameters(self, data_logger):
        unknown_parameters = get_first_guess(data_logger)

        return unknown_parameters

    def test_initialise_flowline(self, data_logger, unknown_parameters):
        flowline, fl_control_vars = initialise_flowline(unknown_parameters, data_logger)

        potential_fl_control_vars = ['bed_h', 'surface_h', 'lambdas', 'w0_m']
        for key in fl_control_vars.keys():
            assert key in data_logger.control_vars
            assert key in potential_fl_control_vars
            assert fl_control_vars[key].requires_grad

        # check if w0_m is correctly calculated if not control_var
        if 'w0_m' not in data_logger.control_vars:
            assert flowline._w0_m.dtype == data_logger.torch_type
            assert np.allclose(
                flowline._w0_m.clone().to('cpu').detach().numpy().astype(np.float64),
                data_logger.flowline_init._w0_m, equal_nan=True)

    def test_initialise_mb_models(self, data_logger, unknown_parameters):
        mb_models, mb_control_var = initialise_mb_models(unknown_parameters, data_logger)

        # check that mb_control_var is empty (not supported now)
        assert mb_control_var == {}

        # check that all needed mb_models are created
        for key in mb_models.keys():
            assert key in data_logger.mb_models_settings.keys()

    def test_cost_fct(self, data_logger, unknown_parameters):
        data_logger.spinup_options = 'do_spinup'
        with pytest.raises(NotImplementedError, match='No spinup possibilities integrated!'):
            cost_fct(unknown_parameters, data_logger)

# TODO
# def test_cost_fct():


# TODO
# def test_calculate_difference_between_observation_and_model():
