from copy import deepcopy

import numpy as np
# import matplotlib
# matplotlib.use('GTKAgg')
import oggm.cfg;

oggm.cfg.initialize()
from agile2d.sandbox.perform_run import perform_run
from agile2d.sandbox.perform_run import get_giluwe_inversion_settings
from agile2d.sandbox.perform_run import default_bed_measurement_dict
from agile2d.sandbox.perform_run import default_surface_noise_dict
from agile2d.sandbox.perform_run import default_biased_fg_dict
from agile2d.sandbox.perform_run import default_rmsed_fg_dict
# from agile2d.sandbox.perform_run import reg_parameters_set_A
# from agile2d.sandbox.perform_run import reg_parameters_set_B
# from agile2d.sandbox.perform_run import reg_parameters_set_C
from agile2d.sandbox.run_additional_setting_dicts import *

from agile2d.core.test_cases import Giluwe
from agile2d.core.visualization import plot_iterative_behaviour

reg_parameters_set_A = np.array([0.2, 1.25, 1e2, 1e5, 0, 0])
reg_parameters_set_B = np.array([0.2, 1.25, 1e3, 1e6, 0, 0])
reg_parameters_set_C = np.array([0.2, 1.25, 1e3, 1e6, 1e7, 0])

basedir = '/home/philipp/zweitversuch'
case = Giluwe
get_my_inversion_settings = get_giluwe_inversion_settings

run_identical_twin = True
run_first_guess = False
run_promised_land = True
run_bed_measurements = False

# ======= identical twin ===================================
if run_identical_twin:
    #identical_twin_inversion_settings = get_my_inversion_settings(
    #    'identical-twin', reg_parameters_set_A)
    #idir = perform_run(case, basedir, identical_twin_inversion_settings,
    #                   create_synthetic_glacier=True)
                       #create_synthetic_glacier=False)
    #plot_iterative_behaviour(idir.gdir, idir.inv_settings['inversion_subdir'])
    identical_twin_inversion_settings = get_my_inversion_settings(
        'identical-twin b', reg_parameters_set_B)
    idir = perform_run(case, basedir, identical_twin_inversion_settings,
                       create_synthetic_glacier=False)
    #plot_iterative_behaviour(idir.gdir, idir.inv_settings['inversion_subdir'])
    #identical_twin_inversion_settings = get_my_inversion_settings(
    #    'identical-twin c', reg_parameters_set_C)
    #idir = perform_run(case, basedir, identical_twin_inversion_settings,
    #                   create_synthetic_glacier=False)
    #plot_iterative_behaviour(idir.gdir, idir.inv_settings['inversion_subdir'])


# ======= first guess ======================================
if run_first_guess:
    first_guess_bias_1_prime_dict = deepcopy(first_guess_bias_1_dict)
    first_guess_bias_1_prime_dict['desired_mean_bias'] = 20.001
    fg_experiments = [
        ('first guess bias 1', first_guess_bias_1_dict, default_rmsed_fg_dict),
        ('first guess bias 2', first_guess_bias_2_dict, default_rmsed_fg_dict),
        ('first guess rmse 1', default_biased_fg_dict,
         first_guess_rmse_1_dict),
        ('first guess rmse 2', default_biased_fg_dict,
         first_guess_rmse_2_dict_giluwe),
        ('first guess bias 1 prime', first_guess_bias_1_prime_dict,
         default_rmsed_fg_dict),
        ]
    for exp_name, bias_dict, rmse_dict in fg_experiments:

        identical_twin_inversion_settings = get_my_inversion_settings(
            exp_name, reg_parameters_set_A)
        idir = perform_run(case, basedir, identical_twin_inversion_settings,
                           biased_fg_dict=bias_dict,
                           rmsed_fg_dict=rmse_dict,
                           create_synthetic_glacier=False)
        #plot_iterative_behaviour(idir.gdir, idir.inv_settings['inversion_subdir'])

# ======= promised land ====================================
if run_promised_land:
    set_A = reg_parameters_set_A.copy()
    set_B = reg_parameters_set_B.copy()
    set_C = reg_parameters_set_C.copy()
    promised_land_1_custom_dict = deepcopy(promised_land_1_dict)
    promised_land_1_custom_dict['desired_rmse'] = 2.001
    promised_land_3_custom_dict = deepcopy(promised_land_3_dict)
    promised_land_3_custom_dict['desired_rmse'] = 10.001

    promised_land_experiments = [
        #('promised land 1a', promised_land_1_dict, set_A),
        #('promised land 1b', promised_land_1_dict, set_B),
        #('promised land 1c', promised_land_1_dict, set_C),
        #('promised land 2a', promised_land_2_dict, set_A),
        #('promised land 2b', promised_land_2_dict, set_B),
        #('promised land 2c', promised_land_2_dict, set_C),
        #('promised land 3a', promised_land_3_dict, set_A),
        #('promised land 3b', promised_land_3_dict, set_B),
        #('promised land 3c', promised_land_3_dict, set_C),
        ('promised land 3b prime', promised_land_3_custom_dict, set_B),
        ('promised land 1c prime', promised_land_1_custom_dict, set_C),
        #('promised land 3a prime', promised_land_3_custom_dict, set_A),
        #('promised land 3b prime', promised_land_3_custom_dict, set_B),
        ]
    for exp_name, surf_noise_dict, reg_parameters_set in \
            promised_land_experiments:
        identical_twin_inversion_settings = get_my_inversion_settings(
            exp_name, reg_parameters_set)
        idir = perform_run(case, basedir, identical_twin_inversion_settings,
                           surface_noise_dict=surf_noise_dict,
                           create_synthetic_glacier=False)
        #plot_iterative_behaviour(idir.gdir, idir.inv_settings['inversion_subdir'])


# ======= bed measurements =================================
if run_bed_measurements:
    set_A = reg_parameters_set_A.copy()
    set_A[5] = 1
    set_B = reg_parameters_set_B.copy()
    set_B[5] = 1
    set_C = reg_parameters_set_C.copy()
    set_C[5] = 1
    bed_measurement_experiments = [
        ('promised land 3b star 1', promised_land_3_dict, set_B,
         bed_measurement_masks.Giluwe_cross),
        ('promised land 3b star 2', promised_land_3_dict, set_B,
         bed_measurement_masks.Giluwe_upper_tongue),
        ('identical-twin star 1', default_surface_noise_dict, set_A,
         bed_measurement_masks.Giluwe_cross),
        ('identical-twin star 2', default_surface_noise_dict, set_A,
         bed_measurement_masks.Giluwe_upper_tongue)
        ]

    for exp_name, surf_noise_dict, reg_parameters_set, measurement_mask in \
            bed_measurement_experiments:
        measurement_dict = default_bed_measurement_dict
        measurement_dict['use'] = True
        measurement_dict['measurement_mask'] = measurement_mask

        identical_twin_inversion_settings = get_my_inversion_settings(
            exp_name, reg_parameters_set)

        idir = perform_run(case, basedir, identical_twin_inversion_settings,
                           surface_noise_dict=surf_noise_dict,
                           bed_measurements_dict=measurement_dict,
                           create_synthetic_glacier=False)
        #plot_iterative_behaviour(idir.gdir, idir.inv_settings['inversion_subdir'])