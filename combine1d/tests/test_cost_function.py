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
                                          define_scaling_terms,
                                          do_height_shift_spinup, descale_unknown_parameters)
from combine1d.core.first_guess import get_first_guess
from combine1d.core.dynamics import run_model_and_get_temporal_model_data
from combine1d.core.flowline import MixedBedFlowline

pytestmark = [pytest.mark.filterwarnings("ignore:<class 'combine1d.core.torch_interp1d.Interp1d'> "
                                         "should not be instantiated.:DeprecationWarning"),
              pytest.mark.test_env("cost_function")]


class TestCreateCostFct:
    def test_get_known_parameters(self, data_logger_init):
        data_logger = data_logger_init
        known_parameters = get_known_parameters(data_logger)
        fl = data_logger.flowline_init

        for key in known_parameters.keys():
            if 'area_bed_h' in data_logger.control_vars:
                assert 'bed_h' in known_parameters
            else:
                assert key in data_logger.control_vars
            if key in ['w0_m', 'lambdas']:
                prefix = '_'
                mask = (data_logger.is_trapezoid & data_logger.ice_mask)
            elif key in ['bed_h']:
                prefix = ''
                mask = data_logger.ice_mask
            elif key in ['area_bed_h']:
                prefix = ''
                mask = data_logger.ice_mask
                key = 'bed_h'
            elif key in ['surface_h', 'section']:
                assert key not in known_parameters.keys()
                continue
            assert np.allclose(known_parameters[key],
                               getattr(fl, prefix + key)[~mask],
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


def get_prepared_data_for_cost_fct(unknown_parameters, data_logger,
                                   observations):
    flowline = initialise_flowline(unknown_parameters, data_logger)
    mb_models = initialise_mb_models(unknown_parameters, data_logger)
    dynamic_model = data_logger.dynamic_model

    if data_logger.spinup_type == 'height_shift_spinup':
        # Here a spinup run is conducted using the control variable
        # height_shift_spinup (vertically shift the whole mb profile)
        flowline = do_height_shift_spinup(
            flowline=flowline, unknown_parameters=unknown_parameters,
            data_logger=data_logger)
    elif data_logger.spinup_type not in [None, 'surface_h']:
        raise NotImplementedError(f'The spinup type {data_logger.spinup_type} '
                                  'possibility is not integrated!')

    observations_mdl, final_fl = \
        run_model_and_get_temporal_model_data(flowline=flowline,
                                              dynamic_model=dynamic_model,
                                              mb_models=mb_models,
                                              observations=observations)

    # now add artificial observation values
    for obs_var in observations.keys():
        for year in observations[obs_var].keys():
            observations[obs_var][year] = \
                observations_mdl[obs_var][year].detach().to('cpu').numpy(
                ).astype(np.float64) + 10.

    data_logger.observations = observations
    data_logger.obs_scaling_parameters = {'scale': {'fl_total_area:m2': 1.,
                                                    'fl_total_area:km2': 10.,
                                                    'area:m2': 1.,
                                                    'area:km2': 10.,
                                                    'dmdtda:kg m-2 yr-1': 1.,
                                                    'dmdtda:kg yr-1': 1.,
                                                    'us:myr-1': 1.,
                                                    'fl_surface_h:m': 2.,
                                                    'fl_widths:m': 5.}}

    define_scaling_terms(data_logger)

    return flowline, mb_models, observations_mdl, final_fl, data_logger


class TestCostFct:
    @pytest.fixture(scope='function')
    def unknown_parameters(self, data_logger):
        unknown_parameters = get_first_guess(data_logger)

        unknown_parameters, unknown_parameters_descaled = \
            descale_unknown_parameters(unknown_parameters, data_logger)

        return unknown_parameters_descaled

    def test_initialise_flowline(self, data_logger, unknown_parameters):
        # just checking that their is a flowline coming out
        flowline = initialise_flowline(unknown_parameters, data_logger)

        assert isinstance(flowline, MixedBedFlowline)

        # check if w0_m is correctly calculated if not control_var
        if 'w0_m' not in data_logger.control_vars:
            assert flowline._w0_m.dtype == data_logger.torch_type
            assert np.allclose(
                flowline._w0_m.clone().to('cpu').detach().numpy().astype(np.float64),
                data_logger.flowline_init._w0_m, equal_nan=True)

    def test_initialise_mb_models(self, data_logger, unknown_parameters):
        mb_models = initialise_mb_models(unknown_parameters, data_logger)

        # check that all needed mb_models are created
        for key in mb_models.keys():
            assert key in data_logger.mb_models_settings.keys()

    def test_calculate_difference_between_observation_and_model(self, data_logger,
                                                                unknown_parameters,
                                                                observations):
        flowline, mb_models, observations_mdl, final_fl, data_logger = \
            get_prepared_data_for_cost_fct(unknown_parameters, data_logger,
                                           observations)
        c_terms, reg_terms = get_cost_terms(observations_mdl,
                                            final_fl,
                                            data_logger)

        dobs, nobs = calculate_difference_between_observation_and_model(
            data_logger,
            observations,
            observations_mdl)

        # check that difference calculated for each observation
        for obs_var in observations.keys():
            for year in observations[obs_var].keys():
                assert dobs[obs_var][year] != []
                assert isinstance(dobs[obs_var][year], torch.Tensor)
                if obs_var in ['fl_total_area:m2', 'fl_total_area:km2',
                               'area:m2', 'area:km2', 'dmdtda:kg m-2 yr-1',
                               'dmdtda:kg yr-1', 'us:myr-1']:
                    assert np.isclose(
                        dobs[obs_var][year].detach().to('cpu').numpy().astype(
                            np.float64), 100.)  # 100 because of 10^2
        assert nobs == 13

    def test_define_scaling_terms(self, data_logger, observations):
        # fill observations with some values
        single_value = 10.
        for obs_var in observations.keys():
            for year in observations[obs_var].keys():
                if obs_var in ['fl_total_area:m2', 'fl_total_area:km2',
                               'area:m2', 'area:km2', 'dmdtda:kg m-2 yr-1',
                               'dmdtda:kg yr-1', 'us:myr-1']:
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
        data_logger.obs_scaling_parameters = {'scale': {'fl_total_area:m2': 1.,
                                                        'fl_total_area:km2': 10.,
                                                        'area:m2': 1.,
                                                        'area:km2': 10.,
                                                        'dmdtda:kg m-2 yr-1': 1.,
                                                        'dmdtda:kg yr-1': 1.,
                                                        'us:myr-1': 1.,
                                                        'fl_surface_h:m': 2.,
                                                        'fl_widths:m': 5.}}

        define_scaling_terms(data_logger)

        assert 'scale' not in data_logger.obs_scaling_parameters.keys()

        reg_parameters = data_logger.obs_scaling_parameters
        for obs_var in observations.keys():
            for year in observations[obs_var].keys():
                assert reg_parameters[obs_var][year].size == 1
                assert isinstance(reg_parameters[obs_var][year], np.float64)

    def test_get_cost_terms(self, data_logger, unknown_parameters,
                            observations):
        flowline, mb_models, observations_mdl, final_fl, data_logger = \
            get_prepared_data_for_cost_fct(unknown_parameters, data_logger,
                                           observations)

        c_terms, reg_terms = get_cost_terms(observations_mdl,
                                            final_fl,
                                            data_logger)

        assert len(c_terms) == 13
        for c_term in c_terms:
            assert c_term != []
            assert isinstance(c_term, torch.Tensor)

        assert len(reg_terms) == 1
        for reg_term in reg_terms:
            assert reg_term != []
            assert isinstance(reg_term, torch.Tensor)

    def test_cost_fct(self, data_logger, unknown_parameters, observations):
        flowline, mb_models, observations_mdl, final_fl, data_logger = \
            get_prepared_data_for_cost_fct(unknown_parameters, data_logger,
                                           observations)

        cost, grad = cost_fct(unknown_parameters, data_logger)

        assert type(cost) == np.ndarray
        assert cost.size == 1
        assert type(grad) == np.ndarray
        assert len(grad) == len(unknown_parameters)

        data_logger.spinup_options = {'do_spinup': 'whatever'}
        data_logger.spinup_type = 'do_spinup'
        with pytest.raises(NotImplementedError,
                           match=f'The spinup option {data_logger.spinup_type} '
                                 'is not implemented!'
                           ):
            cost_fct(unknown_parameters, data_logger)
