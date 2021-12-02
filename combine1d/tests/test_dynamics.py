import collections
import pytest
import numpy as np
import torch

from combine1d.core.dynamics import (run_model_and_get_temporal_model_data,
                                     calculate_model_observations,
                                     run_model_and_get_model_values,
                                     construct_needed_model_data)
from combine1d.core.first_guess import get_first_guess
from combine1d.core.cost_function import (initialise_flowline,
                                          initialise_mb_models)


@pytest.fixture(scope='function')
def unknown_parameters(data_logger):
    unknown_parameters = get_first_guess(data_logger)

    return unknown_parameters


@pytest.fixture(scope='function')
def flowline(data_logger, unknown_parameters):
    flowline, fl_control_vars = initialise_flowline(unknown_parameters,
                                                    data_logger)

    return flowline


@pytest.fixture(scope='function')
def mb_models(data_logger, unknown_parameters):
    mb_models, mb_control_var = initialise_mb_models(unknown_parameters,
                                                     data_logger)

    return mb_models


class TestDynamicRunWithModelObservations:
    @pytest.fixture(scope='function')
    def observations(self):
        return {'fl_surface_h:m': {'2003': []},
                'fl_widths:m': {'2003': []},
                'fl_total_area:m2': {'2003': []},
                'fl_total_area:km2': {'2003': []},
                'area:m2': {'2000': [],
                            '2005': [],
                            '2009': []},
                'area:km2': {'2002': [],
                             '2007': []},
                'dh:m': {'2000-2005': [],
                         '2005-2010': []}
                }

    def test_construct_needed_model_data(self, observations):
        needed_model_data = construct_needed_model_data(observations)

        assert type(needed_model_data) == collections.OrderedDict

        all_years = np.array([yr for yr in needed_model_data.keys()])
        years_should_have = [2000, 2002, 2003, 2005, 2007, 2009, 2010]
        # check that years are sorted
        assert np.all(np.diff(all_years) > 0.)

        # check all years needed are there (and only those)
        for yr in years_should_have:
            assert yr in all_years
        for yr in all_years:
            assert yr in years_should_have

        # check if wrong measurement unit is detected
        wrong_unit = 'something'
        for obs_name in ['fl_surface_h', 'fl_widths', 'fl_total_area', 'area',
                         'dh']:
            with pytest.raises(NotImplementedError,
                               match=f'Wrong unit for {obs_name}!'):
                construct_needed_model_data({f'{obs_name}:{wrong_unit}': {}})

    def test_run_model_and_get_model_values(self, flowline, mb_models,
                                            observations):
        needed_model_data = construct_needed_model_data(observations)
        actual_model_data, flowline = run_model_and_get_model_values(flowline,
                                                                     mb_models,
                                                                     needed_model_data)

        # check that every needed entry is their and has a value
        for year in needed_model_data.keys():
            for var in needed_model_data[year]:
                assert actual_model_data[year][var] != []
                assert type(actual_model_data[year][var]) == torch.Tensor

    def test_calculate_model_observations(self, flowline, mb_models,
                                          observations):
        needed_model_data = construct_needed_model_data(observations)
        actual_model_data, flowline = run_model_and_get_model_values(flowline,
                                                                     mb_models,
                                                                     needed_model_data)
        calculated_model_observations = calculate_model_observations(observations,
                                                                     actual_model_data)

        # check that every observation has a model counterpart
        for var in observations.keys():
            for year in observations[var].keys():
                assert calculated_model_observations[var][year] != []
                assert type(calculated_model_observations[var][year]) == torch.Tensor

    def test_run_model_and_get_temporal_model_data(self, flowline, mb_models,
                                                   observations):
        calculated_model_observations, flowline = \
            run_model_and_get_temporal_model_data(flowline, mb_models,
                                                  observations)

        # check that every observation has a model counterpart
        for var in observations.keys():
            for year in observations[var].keys():
                assert calculated_model_observations[var][year] != []
                assert type(calculated_model_observations[var][year]) == torch.Tensor
