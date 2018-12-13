from copy import deepcopy
import numpy as np
#import matplotlib
#matplotlib.use('GTK3Agg')
import oggm.cfg; oggm.cfg.initialize()
from combine.sandbox.perform_run import perform_run
from combine.sandbox.perform_run import get_borden_inversion_settings
from combine.sandbox.perform_run import get_giluwe_inversion_settings
from combine.sandbox.perform_run import get_fg_bias_dict
from combine.sandbox.perform_run import get_fg_rmse_dict
from combine.sandbox.perform_run import get_surf_rmse_dict
from combine.sandbox.perform_run import default_bed_measurement_dict
from combine.sandbox.perform_run import default_biased_fg_dict
from combine.sandbox.perform_run import default_rmsed_fg_dict
from combine.sandbox.perform_run import default_surface_noise_dict
# from combine.sandbox.perform_run import reg_parameters_set_A
# from combine.sandbox.perform_run import reg_parameters_set_B
# from combine.sandbox.perform_run import reg_parameters_set_C
from combine.sandbox.run_additional_setting_dicts import *

from combine.core.test_cases import Borden, Giluwe
from combine.core.visualization import plot_iterative_behaviour

basedir = '/home/philipp/final_03'
case = Giluwe
case = Borden
if case.name == 'Borden Peninsula':
    get_my_inversion_settings = get_borden_inversion_settings
    #reg_parameters_set = np.array([0.2, 1.25, 1e3, 1e6, 0, 0]) # final2
    #reg_parameters_set = np.array([0.2, 1.25, 1e3, 1e6, 1e6, 0]) # final3
    reg_parameters_set = np.array([0.2, 1.25, 2e4, 2e7, 0, 0])  # final2
elif case.name == 'Giluwe':
    get_my_inversion_settings = get_giluwe_inversion_settings
    reg_parameters_set = np.array([0.2, 1.25, 1e3, 1e6, 0, 0])

run_identical_twin = False
run_first_guess_bias = False
run_first_guess_rmse = False
run_promised_land = False
run_bed_measurements = True
print(basedir)
print(case.name)
print(run_identical_twin)
print(run_first_guess_bias)
print(run_first_guess_rmse)
print(run_promised_land)
print(run_bed_measurements)

# ======= identical twin ===================================
if run_identical_twin:
    identical_twin_inversion_settings = get_my_inversion_settings(
        'identical-twin', reg_parameters_set)
    idir = perform_run(case, basedir, identical_twin_inversion_settings,
                       create_synthetic_glacier=False)
                       #create_synthetic_glacier=True)

# ======= first guess ======================================
if run_first_guess_bias:
    fg_bias_experiments = []
    bias_range = np.arange(-80, 81, 5)
    for bias in bias_range:
        experiment = ('fg bias {:3d}'.format(bias), get_fg_bias_dict(bias))
        fg_bias_experiments.append(experiment)
    for exp_name, bias_dict in fg_bias_experiments:
        identical_twin_inversion_settings = get_my_inversion_settings(
            exp_name, reg_parameters_set)
        idir = perform_run(case, basedir, identical_twin_inversion_settings,
                           biased_fg_dict=bias_dict,
                           create_synthetic_glacier=False)
        #plot_iterative_behaviour(idir.gdir, idir.inv_settings['inversion_subdir'])
if run_first_guess_rmse:
    fg_rmse_experiments = []
    rmse_range = np.arange(5, 71, 5)
    for base in [2, 3, 4]:
        for rmse in rmse_range:
            experiment = ('fg rmse {:d} {:3d}'.format(rmse, base),
                          get_fg_rmse_dict(rmse, base))
            fg_rmse_experiments.append(experiment)
    for exp_name, rmse_dict in fg_rmse_experiments:
        identical_twin_inversion_settings = get_my_inversion_settings(
            exp_name, reg_parameters_set)
        idir = perform_run(case, basedir, identical_twin_inversion_settings,
                           rmsed_fg_dict=rmse_dict,
                           create_synthetic_glacier=False)

# ======= promised land ====================================
if run_promised_land:
    promised_land_experiments = []
    for rmse in [2, 4, 6, 8, 10, 12]:
        for i, base in enumerate([2, 3, 4]):
            experiment = ('promised land {:d} {:2d}'.format(base, rmse),
                          get_surf_rmse_dict(rmse, base), reg_parameters_set)
            promised_land_experiments.append(experiment)

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
    set = reg_parameters_set.copy()
    set[5] = 1
    set2 = reg_parameters_set.copy()
    set2[5] = 1. / 9.
    set3 = reg_parameters_set.copy()
    set3[5] = 0.5
    base = 3
    #TODO
    bed_measurement_experiments = [
        #('promised land 3b star ', promised_land_3_dict, set_B,
        # bed_measurement_masks.Borden_horizontal),
        #('identical-twin star', default_surface_noise_dict, set_A,
        # bed_measurement_masks.Borden_horizontal),
        #('promised land 3b star 2', promised_land_3_dict, set_B,
        # bed_measurement_masks.Borden_horizontal2),
        #('identical-twin star 2', default_surface_noise_dict, set_A,
        # bed_measurement_masks.Borden_horizontal2)
        ]
    if case is Giluwe:
        shape = bed_measurement_masks.Giluwe_cross
        shapename = 'cross'
        shape = bed_measurement_masks.Giluwe_upper_tongue
        shapename = 'upper tongue'
    if case is Borden:
        shape = bed_measurement_masks.Borden_horizontal
        shapename = 'horizontal'
        # shape  = bed_measurement_masks.Borden_horizontal2
        # shapename = 'horizontal2'
    bed_measurement_experiments = []
    for rmse in [10]: #2, 6
        experiment = ('bed measurements fg {:d} {:s} {:2d}'.format(base,
                                                                shapename,
                                                                rmse),
                      get_surf_rmse_dict(rmse, base), set2,
                      shape)
        bed_measurement_experiments.append(experiment)
    for exp_name, surf_noise_dict, reg_parameters_set, measurement_mask in \
            bed_measurement_experiments:
        measurement_dict = default_bed_measurement_dict
        measurement_dict['use'] = True
        measurement_dict['measurement_mask'] = measurement_mask

        identical_twin_inversion_settings = get_my_inversion_settings(
            exp_name, reg_parameters_set)
        print(exp_name)
        idir = perform_run(case, basedir, identical_twin_inversion_settings,
                           surface_noise_dict=surf_noise_dict,
                           bed_measurements_dict=measurement_dict,
                           create_synthetic_glacier=True,
                           # create_synthetic_glacier=False,
                           use_measurements_in_fg=True)
        #plot_iterative_behaviour(idir.gdir, idir.inv_settings['inversion_subdir'])