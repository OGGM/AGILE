from collections import OrderedDict
import copy

import torch
import numpy as np

from combine1d.core.flowline import FluxBasedModel


def run_model_and_get_temporal_model_data(flowline, mb_models, observations,
                                          velocity_smoothing=False):
    """TODO

    Parameters
    ----------
    flowline: py:class:`oggm.Flowline`
    mb_models: dict
        {'MB1': {'mb_model': lala, 'years': np.array([1950, 2010])}}
    observations: dict
        {'Area', {'2010': np.array([23])}, 'dh': {'2010-2015': np.array([45])}}

    Returns
    -------
    : dict
        dict with the same structure as Obs
    """

    needed_model_data = construct_needed_model_data(observations)

    actual_model_data, flowline = run_model_and_get_model_values(
        flowline, mb_models, needed_model_data,
        velocity_smoothing=velocity_smoothing)

    # postprocessing model data and calculate actual observations (e.g. calculate
    # delta values)
    calculated_model_observations = calculate_model_observations(observations,
                                                                 actual_model_data)

    return calculated_model_observations, flowline


def construct_needed_model_data(observations):
    """creates a ordered dict with all years and model variables we must save
    during the forward run. Also conduct a unit check.
    """
    needed_model_data = {}

    def unit_check(variable, given_unit, all_units):
        if given_unit not in all_units:
            raise NotImplementedError(f'Wrong unit for {variable}!')

    def add_model_observation_years(new_years):
        for year in new_years:
            if year not in needed_model_data.keys():
                needed_model_data[year] = []

    def add_model_variable(year, variable):
        if variable not in needed_model_data[year]:
            needed_model_data[year].append(variable)

    for obs_key in observations.keys():
        obs_name, obs_unit = obs_key.split(':')
        if obs_name == 'dh':
            unit_check('dh', obs_unit, ['m'])

            # extract measurement years
            needed_years = []
            for period in observations[obs_key].keys():
                for new_year in period.split('-'):
                    needed_years.append(int(new_year))
            # to only have unique years
            needed_years = list(dict.fromkeys(needed_years))
            add_model_observation_years(needed_years)

            # add needed measurements
            for yr in needed_years:
                if obs_unit == 'm':
                    add_model_variable(yr, 'volume:m3')
                    add_model_variable(yr, 'area:m2')
                else:
                    raise NotImplementedError(f'{obs_unit} for dh')

        else:  # here now all measurements without a period
            # extract measurement years
            needed_years = [int(yr) for yr in observations[obs_key].keys()]
            add_model_observation_years(needed_years)

            if obs_name in ['area', 'fl_total_area']:
                unit_check(obs_name, obs_unit, ['m2', 'km2'])
            elif obs_name in ['fl_surface_h', 'fl_widths']:
                unit_check(obs_name, obs_unit, ['m'])
            elif obs_name in ['us']:
                unit_check(obs_name, obs_unit, ['myr-1'])
            else:
                raise NotImplementedError(f'{obs_name} not implemented!')

            # add needed measurements
            for yr in needed_years:
                add_model_variable(yr, obs_key)

    needed_model_data = OrderedDict(sorted(needed_model_data.items()))

    return needed_model_data


def run_model_and_get_model_values(flowline, mb_models, needed_model_data,
                                   velocity_smoothing=False):
    # start the actual forward run and get observations from model
    actual_model_data = {}
    for mb_key in mb_models.keys():
        flux_model = FluxBasedModel(flowline,
                                    mb_model=mb_models[mb_key]['mb_model'],
                                    y0=mb_models[mb_key]['years'][0],
                                    fs=0.,
                                    mb_elev_feedback='annual',
                                    velocity_smoothing=velocity_smoothing)

        # years with observations using the same mass balance model
        obs_yrs_current_mb_model = [k for k in needed_model_data.keys()
                                    if (mb_models[mb_key]['years'][0] < k <=
                                        mb_models[mb_key]['years'][1])]
        for obs_yr in obs_yrs_current_mb_model:
            # let the model run to the year of observation
            flux_model.run_until(obs_yr)

            actual_model_data[obs_yr] = {}
            # save model counterparts of observation
            for var in needed_model_data[obs_yr]:
                if var in ['area:m2', 'fl_total_area:m2']:
                    actual_model_data[obs_yr][var] = flux_model.area_m2
                elif var in ['area:km2', 'fl_total_area:km2']:
                    actual_model_data[obs_yr][var] = flux_model.area_km2
                elif var == 'volume:m3':
                    actual_model_data[obs_yr][var] = flux_model.volume_m3
                elif var == 'fl_surface_h:m':
                    actual_model_data[obs_yr][var] = flux_model.fls[0].surface_h
                elif var == 'fl_widths:m':
                    actual_model_data[obs_yr][var] = flux_model.fls[0].widths_m
                elif var == 'us:myr-1':
                    actual_model_data[obs_yr][var] = flux_model.u_stag * \
                        flux_model.surf_vel_fac * flux_model.sec_in_year
                else:
                    raise NotImplementedError(f'{var}')

        # after getting everything run the model to ye of the current mb_model
        flux_model.run_until(mb_models[mb_key]['years'][1])

        # save flowline for switching to the next mb_model
        flowline = flux_model.fls[0]

    return actual_model_data, flowline


def calculate_model_observations(observations, actual_model_data):
    out = copy.deepcopy(observations)
    # go through all observations and get/calculate model counterpart
    for var_key in out.keys():
        # first all observations where nothing need to be calculated
        if var_key in ['area:m2', 'area:km2', 'fl_total_area:m2',
                       'fl_total_area:km2', 'fl_surface_h:m', 'fl_widths:m',
                       'us:myr-1']:
            for year in out[var_key].keys():
                out[var_key][year] = actual_model_data[int(year)][var_key]

        # here start the variables which need some calculation
        elif var_key == 'dh:m':
            for period in out[var_key].keys():
                y1, y2 = [int(yr) for yr in period.split('-')]
                out[var_key][period] = \
                    (actual_model_data[y2]['volume:m3'] -
                     actual_model_data[y1]['volume:m3']) / \
                    ((actual_model_data[y2]['area:m2'] +
                      actual_model_data[y1]['area:m2']) /
                     torch.tensor(2.,
                                  dtype=actual_model_data[y1]['area:m2'].dtype,
                                  device=actual_model_data[y1]['area:m2'].device,
                                  requires_grad=False)
                     )
        else:
            raise NotImplementedError(f'{var_key}')

    return out
