from copy import deepcopy
import numpy as np
#import matplotlib
#matplotlib.use('GTK3Agg')
import oggm.cfg; oggm.cfg.initialize()
from cobbi.sandbox.perform_run import perform_run
from cobbi.sandbox.perform_run import get_borden_inversion_settings
from cobbi.sandbox.perform_run import default_bed_measurement_dict
from cobbi.sandbox.perform_run import default_surface_noise_dict
from cobbi.sandbox.perform_run import default_minimize_options
from cobbi.sandbox.perform_run import default_biased_fg_dict
from cobbi.sandbox.perform_run import default_rmsed_fg_dict
from cobbi.sandbox.perform_run import reg_parameters_set_A
from cobbi.sandbox.perform_run import reg_parameters_set_B
from cobbi.sandbox.perform_run import reg_parameters_set_C
from cobbi.sandbox.run_additional_setting_dicts import *

from cobbi.core.test_cases import Borden
from cobbi.core.visualization import plot_iterative_behaviour

basedir = '/data/philipp/erstabgabe'
case = Borden
get_my_inversion_settings = get_borden_inversion_settings

run_identical_twin = False
run_first_guess = False
run_promised_land = False
run_bed_measurements = True

# ======= identical twin ===================================
if run_identical_twin:
    identical_twin_inversion_settings = get_my_inversion_settings('identical-twin',
                                                                  reg_parameters_set_A)
    idir = perform_run(case, basedir, identical_twin_inversion_settings,
                       #create_synthetic_glacier=True)
                       create_synthetic_glacier=False)
    plot_iterative_behaviour(idir)


# ======= first guess ======================================
if run_first_guess:
    fg_experiments = [
        ('first guess bias 1', first_guess_bias_1_dict, default_rmsed_fg_dict),
        ('first guess bias 2', first_guess_bias_2_dict, default_rmsed_fg_dict),
        ('first guess rmse 1', default_biased_fg_dict, first_guess_rmse_1_dict),
        ('first guess rmse 2', default_biased_fg_dict,
         first_guess_rmse_2_dict_borden)
        ]
    for exp_name, bias_dict, rmse_dict in fg_experiments:

        identical_twin_inversion_settings = get_my_inversion_settings(
            exp_name, reg_parameters_set_A)
        idir = perform_run(case, basedir, identical_twin_inversion_settings,
                           biased_fg_dict=bias_dict,
                           rmsed_fg_dict=rmse_dict,
        #create_synthetic_glacier=True)
                           create_synthetic_glacier=False)
        plot_iterative_behaviour(idir)

# ======= promised land ====================================
if run_promised_land:
    set_A = reg_parameters_set_A.copy()
    set_B = reg_parameters_set_B.copy()
    set_C = reg_parameters_set_C.copy()
    set_C_custom = reg_parameters_set_C.copy()
    set_C_custom[4] = 1e7
    set_C_custom2 = reg_parameters_set_C.copy()
    set_C_custom2[4] = 1e8
    promised_land_experiments = [
#        ('promised land 1a', promised_land_1_dict, set_A),
#        ('promised land 1b', promised_land_1_dict, set_B),
#        ('promised land 1c', promised_land_1_dict, set_C),
#        ('promised land 1c custom', promised_land_1_dict, set_C_custom),
#        ('promised land 1c custom2', promised_land_1_dict, set_C_custom2),
#        ('promised land 2a', promised_land_2_dict, set_A),
#        ('promised land 2b', promised_land_2_dict, set_B),
#        ('promised land 2c', promised_land_2_dict, set_C),
#        ('promised land 2c custom', promised_land_2_dict, set_C_custom),
#        ('promised land 2c custom2', promised_land_2_dict, set_C_custom2),
#        ('promised land 3a', promised_land_3_dict, set_A),
#        ('promised land 3b', promised_land_3_dict, set_B),
#        ('promised land 3c', promised_land_3_dict, set_C)
        ('promised land 3c custom', promised_land_3_dict, set_C_custom),
        ('promised land 3c custom2', promised_land_3_dict, set_C_custom2)
        ]

    for exp_name, surf_noise_dict, reg_parameters_set in \
            promised_land_experiments:
        identical_twin_inversion_settings = get_my_inversion_settings(
            exp_name, reg_parameters_set)
        idir = perform_run(case, basedir, identical_twin_inversion_settings,
                           surface_noise_dict=surf_noise_dict,
        #create_synthetic_glacier=True)
                           create_synthetic_glacier=False)
        plot_iterative_behaviour(idir)

# ======= bed measurements =================================
if run_bed_measurements:
    set_A = reg_parameters_set_A.copy()
    set_A[5] = 1
    set_B = reg_parameters_set_B.copy()
    set_B[5] = 1
    set_C = reg_parameters_set_C.copy()
    set_C[5] = 1
    set_C_custom = set_C.copy()
    set_C_custom[4] = 1e7
    set_C_custom2 = set_C.copy()
    set_C_custom2[4] = 1e8
    bed_measurement_experiments = [
        ('promised land 3c star ', promised_land_3_dict, set_C,
         bed_measurement_masks.Borden_horizontal),
        ('promised land 3c custom star ', promised_land_3_dict, set_C_custom,
         bed_measurement_masks.Borden_horizontal),
#        ('promised land 3c custom2 star ', promised_land_3_dict, set_C_custom2,
#         bed_measurement_masks.Borden_horizontal),
#        ('identical-twin star', default_surface_noise_dict, set_A,
#         bed_measurement_masks.Borden_horizontal)
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
        #create_synthetic_glacier=True)
                           create_synthetic_glacier=False)
        plot_iterative_behaviour(idir)