import types

import numpy as np
import pytest
import torch

from combine1d.core.cost_function import (get_indices_for_unknown_parameters,
                                          get_known_parameters, create_cost_fct,
                                          initialise_flowline,
                                          initialise_mb_models, get_cost_terms,
                                          cost_fct,
                                          calculate_difference_between_observation_and_model,
                                          define_reg_parameters, get_gradients)
from combine1d.core.first_guess import get_first_guess
from combine1d.core.dynamics import run_model_and_get_temporal_model_data


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
        flowline, fl_control_vars = initialise_flowline(unknown_parameters,
                                                        data_logger)

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
        mb_models, mb_control_var = initialise_mb_models(unknown_parameters,
                                                         data_logger)

        # check that mb_control_var is empty (not supported now)
        assert mb_control_var == {}

        # check that all needed mb_models are created
        for key in mb_models.keys():
            assert key in data_logger.mb_models_settings.keys()

    def test_calculate_difference_between_observation_and_model(self, data_logger,
                                                                unknown_parameters,
                                                                observations):
        flowline, fl_control_vars = initialise_flowline(unknown_parameters,
                                                        data_logger)
        mb_models, mb_control_var = initialise_mb_models(unknown_parameters,
                                                         data_logger)

        observations_mdl, final_fl = \
            run_model_and_get_temporal_model_data(flowline, mb_models,
                                                  observations)

        # now add artificial observation values
        for obs_var in observations.keys():
            for year in observations[obs_var].keys():
                observations[obs_var][year] = \
                    observations_mdl[obs_var][year].detach().numpy().astype(
                        np.float64) + 10.

        dobs, nobs = calculate_difference_between_observation_and_model(
            observations,
            observations_mdl)

        # check that difference calculated for each observation
        for obs_var in observations.keys():
            for year in observations[obs_var].keys():
                assert dobs[obs_var][year] != []
                assert type(dobs[obs_var][year]) == torch.Tensor
                if obs_var in ['fl_total_area:m2', 'fl_total_area:km2',
                               'area:m2', 'area:km2', 'dh:m']:
                    assert dobs[obs_var][year] == 100.
        assert nobs == 11

    def test_define_reg_parameters(self, data_logger, observations):
        # fill observations with some values
        single_value = 10.
        for obs_var in observations.keys():
            for year in observations[obs_var].keys():
                if obs_var in ['fl_total_area:m2', 'fl_total_area:km2',
                               'area:m2', 'area:km2', 'dh:m']:
                    observations[obs_var][year] = single_value
                    single_value += 10.
                elif obs_var in ['fl_surface_h:m', 'fl_widths:m']:
                    length = data_logger.flowline_init.nx
                    observations[obs_var][year] = np.arange(single_value,
                                                            single_value +
                                                            length + 1,
                                                            1)
                    single_value += 10.
                else:
                    raise NotImplementedError(f'{obs_var}')

        data_logger.observations = observations
        data_logger.obs_reg_parameters = {'scale': {'fl_total_area:m2': 1.,
                                                    'fl_total_area:km2': 10.,
                                                    'area:m2': 1.,
                                                    'area:km2': 10.,
                                                    'dh:m': 1.,
                                                    'fl_surface_h:m': 2.,
                                                    'fl_widths:m': 5.}}

        define_reg_parameters(data_logger)

        assert 'scale' not in data_logger.obs_reg_parameters.keys()

        reg_parameters = data_logger.obs_reg_parameters
        for obs_var in observations.keys():
            for year in observations[obs_var].keys():
                assert len(reg_parameters[obs_var][year]) == 1
                assert type(reg_parameters[obs_var][year]) == torch.Tensor

    def test_get_cost_terms(self, data_logger, unknown_parameters,
                            observations):
        flowline, fl_control_vars = initialise_flowline(unknown_parameters,
                                                        data_logger)
        mb_models, mb_control_var = initialise_mb_models(unknown_parameters,
                                                         data_logger)

        observations_mdl, final_fl = \
            run_model_and_get_temporal_model_data(flowline, mb_models,
                                                  observations)

        # now add artificial observation values
        for obs_var in observations.keys():
            for year in observations[obs_var].keys():
                observations[obs_var][year] = \
                    observations_mdl[obs_var][year].detach().numpy().astype(
                        np.float64) + 10.

        data_logger.observations = observations
        data_logger.obs_reg_parameters = {'scale': {'fl_total_area:m2': 1.,
                                                    'fl_total_area:km2': 10.,
                                                    'area:m2': 1.,
                                                    'area:km2': 10.,
                                                    'dh:m': 1.,
                                                    'fl_surface_h:m': 2.,
                                                    'fl_widths:m': 5.}}

        c_terms = get_cost_terms(observations_mdl,
                                 final_fl,
                                 data_logger)

        assert len(c_terms) == 12
        for c_term in c_terms:
            assert c_term != []
            assert type(c_term) == torch.Tensor

    def test_get_gradients(self, data_logger, unknown_parameters,
                           observations):
        flowline, fl_control_vars = initialise_flowline(unknown_parameters,
                                                        data_logger)
        mb_models, mb_control_vars = initialise_mb_models(unknown_parameters,
                                                          data_logger)

        observations_mdl, final_fl = \
            run_model_and_get_temporal_model_data(flowline, mb_models,
                                                  observations)

        # now add artificial observation values
        for obs_var in observations.keys():
            for year in observations[obs_var].keys():
                observations[obs_var][year] = \
                    observations_mdl[obs_var][year].detach().numpy().astype(
                        np.float64) + 10.

        data_logger.observations = observations
        data_logger.obs_reg_parameters = {'scale': {'fl_total_area:m2': 1.,
                                                    'fl_total_area:km2': 10.,
                                                    'area:m2': 1.,
                                                    'area:km2': 10.,
                                                    'dh:m': 1.,
                                                    'fl_surface_h:m': 2.,
                                                    'fl_widths:m': 5.}}

        c_terms = get_cost_terms(observations_mdl,
                                 final_fl,
                                 data_logger)

        # sum up cost function terms
        c = c_terms.sum()

        # calculate the gradient for the optimisation parameter
        c.backward()

        # here try to get the gradients
        grad = get_gradients(fl_control_vars,
                             mb_control_vars,
                             data_logger,
                             length=len(unknown_parameters))

        assert len(grad) == len(unknown_parameters)
        assert type(grad) == np.ndarray

    def test_cost_fct(self, data_logger, unknown_parameters):
        data_logger.spinup_options = 'do_spinup'
        with pytest.raises(NotImplementedError, match='No spinup possibilities integrated!'):
            cost_fct(unknown_parameters, data_logger)
