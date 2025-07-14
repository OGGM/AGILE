# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python [conda env:oggm_env]
#     language: python
#     name: conda-env-oggm_env-py
# ---

from oggm import utils
import copy
import numpy as np
from get_reference_glaciers import get_fl_at_year
import os
import pickle


# # get statistics for single interation

def get_stats_for_single_iteration(iter, fl_runs, sfc_h_starts, glacier, glacier_state):
    fl_ref_1980 = get_fl_at_year(glacier, glacier_state, 1980)
    fl_ref_2020 = get_fl_at_year(glacier, glacier_state, 2020)

    if iter > fl_runs.iteration[-1]:
        iter = fl_runs.iteration[-1]

    fl = fl_runs.isel(iteration=iter).item()
    sfc_h_start = sfc_h_starts.isel(iteration=iter)

    def get_volume(fl):
        return fl.section * fl.dx_meter

    mad_today_volume= utils.mad(
            get_volume(fl_ref_2020),
            get_volume(fl)
        )

    mad_bed_h = utils.mad(
            fl_ref_2020.bed_h,
            fl.bed_h
        )

    #now recreate initial flowline state
    fl.surface_h = sfc_h_start
    mad_past_volume = utils.mad(
            get_volume(fl_ref_1980),
            get_volume(fl)
        )

    return mad_today_volume, mad_bed_h, mad_past_volume


# # Get statistics for all iterations

def get_stats_for_all_iterations(ds, glacier, glacier_state, ds_ref=None):

    fl_run = copy.deepcopy(ds.flowlines)
    sfc_h_starts = copy.deepcopy(ds.sfc_h_start)

    if ds_ref is None:
        ds_ref = ds

    mad_today_volume_ref, mad_bed_h_ref, mad_past_volume_ref = get_stats_for_single_iteration(
        0,
        copy.deepcopy(ds_ref.flowlines),
        copy.deepcopy(ds_ref.sfc_h_start),
        glacier, glacier_state)

    mad_today_volume = []
    mad_bed_h = []
    mad_past_volume = []

    for fl, iter, sfc_h_start in zip(fl_run, fl_run.iteration, sfc_h_starts):
        mad_today_volume_single, mad_bed_h_single, mad_past_volume_single = get_stats_for_single_iteration(
            iter, fl_run, sfc_h_starts, glacier, glacier_state)
        fl = fl.item()
        mad_today_volume.append(mad_today_volume_single)

        mad_bed_h.append(mad_bed_h_single)

        #now recreate initial flowline state
        fl.surface_h = sfc_h_start
        mad_past_volume.append(mad_past_volume_single)

    # scale all by the reference
    mad_today_volume = np.array(mad_today_volume)
    mad_today_volume /= mad_today_volume_ref

    mad_bed_h = np.array(mad_bed_h)
    mad_bed_h /= mad_bed_h_ref

    mad_past_volume = np.array(mad_past_volume)
    mad_past_volume /= mad_past_volume_ref

    return ((mad_today_volume, mad_today_volume_ref),
            (mad_bed_h, mad_bed_h_ref),
            (mad_past_volume, mad_past_volume_ref))


# # Get stats matrix for all settings

# ## get reference statistics

def get_ref_stats(main_results_dir, glacier, glacier_state,
                  lam, obs, fg_method, reg='reg0'):
    fp_pkl = os.path.join(main_results_dir,
                          f'{glacier}_{glacier_state}_full_run_{fg_method}_{lam}_{obs}_{reg}.pkl')

    with open(fp_pkl, 'rb') as handle:
        ds_run = pickle.load(handle)

    mad_today_volume_ref, mad_bed_h_ref, mad_past_volume_ref = get_stats_for_single_iteration(
    0,
    ds_run.flowlines,
    ds_run.sfc_h_start,
    glacier,
    glacier_state)

    return mad_today_volume_ref, mad_bed_h_ref, mad_past_volume_ref


# ## get statistics for single setting

def get_stats_for_setting(main_results_dir, glacier, glacier_state, iteration,
                          lam, obs, fg_method,
                          mad_today_volume_ref, mad_bed_h_ref,
                          reg='reg0'):

    fp_pkl = os.path.join(main_results_dir,
                          f'{glacier}_{glacier_state}_full_run_{fg_method}_{lam}_{obs}_{reg}.pkl')
    
    with open(fp_pkl, 'rb') as handle:
        ds_run = pickle.load(handle)
    
    mad_today_volume, mad_bed_h, mad_past_volume = get_stats_for_single_iteration(
        iteration,
        ds_run.flowlines,
        ds_run.sfc_h_start,
        glacier,
        glacier_state)
    
    return (mad_today_volume / mad_today_volume_ref,
            mad_bed_h / mad_bed_h_ref)


# ## create matrices of statistics for all settings

def create_stat_matrices(main_results_dir, glacier, glacier_state, iteration,
                         experiment_options,
                         fg_method='fg_oggm',
                         check_ref_values=False):

    if len(experiment_options['regularisation_terms']) > 1:
        raise ValueError('Only working for a single regularisation setting!')

    reg = list(experiment_options['regularisation_terms'].keys())[0]

    if check_ref_values:
        mad_today_volume_general_ref = None
        mad_bed_h_general_ref = None
        for lam in experiment_options['cost_lambda']:
            for obs in experiment_options['observations']:
                mad_today_volume_ref, mad_bed_h_ref, mad_past_volume_ref = get_ref_stats(
                    main_results_dir,
                    glacier, glacier_state,
                    lam, obs, fg_method, reg=reg)
        
                if mad_today_volume_general_ref is None:
                    mad_today_volume_general_ref = mad_today_volume_ref
                    mad_bed_h_general_ref = mad_bed_h_ref
                    continue
        
                diff_today_vol = mad_today_volume_ref - mad_today_volume_general_ref
                diff_bed_h = mad_bed_h_ref - mad_bed_h_general_ref
        
                assert diff_today_vol == 0, diff_today_vol
                assert diff_bed_h == 0, diff_bed_h

    mad_today_volume_ref, mad_bed_h_ref, mad_past_volume_ref = get_ref_stats(
        main_results_dir,
        glacier, glacier_state, 'lam0', 'obs0', fg_method, reg=reg)

    lambda_values = list(experiment_options['cost_lambda'].keys())
    observation_values = list(experiment_options['observations'].keys())

    # Initialize matrices for the two output statistics
    mad_today_volume_matrix = np.zeros((len(lambda_values), len(observation_values)))
    mad_bed_h_matrix = np.zeros((len(lambda_values), len(observation_values)))

    for i, lambda_val in enumerate(lambda_values):
        for j, obs_val in enumerate(observation_values):
            mad_today_v, mad_bed_h = get_stats_for_setting(
                main_results_dir,
                glacier, glacier_state, iteration,
                lambda_val, obs_val, fg_method,
                mad_today_volume_ref, mad_bed_h_ref,
                reg=reg
            )
            mad_today_volume_matrix[i, j] = mad_today_v
            mad_bed_h_matrix[i, j] = mad_bed_h

    return mad_today_volume_matrix, mad_bed_h_matrix, lambda_values, observation_values
