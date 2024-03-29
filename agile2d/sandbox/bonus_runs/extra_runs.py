from copy import deepcopy
import numpy as np
#import matplotlib
#matplotlib.use('GTK3Agg')
import oggm.cfg; oggm.cfg.initialize()
from agile2d.sandbox.perform_run import perform_run
from agile2d.sandbox.perform_run import get_borden_inversion_settings
from agile2d.sandbox.perform_run import get_giluwe_inversion_settings
from agile2d.sandbox.perform_run import get_fg_bias_dict
from agile2d.sandbox.perform_run import get_fg_rmse_dict
from agile2d.sandbox.perform_run import get_surf_rmse_dict
from agile2d.sandbox.perform_run import default_bed_measurement_dict
from agile2d.sandbox.perform_run import default_biased_fg_dict
from agile2d.sandbox.perform_run import default_rmsed_fg_dict
from agile2d.sandbox.perform_run import default_surface_noise_dict
# from agile.sandbox.perform_run import reg_parameters_set_A
# from agile.sandbox.perform_run import reg_parameters_set_B
# from agile.sandbox.perform_run import reg_parameters_set_C
from agile2d.sandbox.run_additional_setting_dicts import *

from agile2d.core.test_cases import Borden, Giluwe
from agile2d.core.visualization import plot_iterative_behaviour


basedir = '/home/philipp/extra1'
#case = Giluwe
case = Giluwe
if case.name == 'Borden Peninsula':
    get_my_inversion_settings = get_borden_inversion_settings
    reg_parameters_set = np.array([0.2, 1.25, 1e3, 1e6, 0, 0]) # final2
    reg_parameters_set = np.array([0.2, 1.25, 5e3, 5e6, 0, 0]) # final3
elif case.name == 'Giluwe':
    get_my_inversion_settings = get_giluwe_inversion_settings
    reg_parameters_set = np.array([0.2, 1.25, 1e3, 1e6, 0, 0])
    reg_parameters_set = np.array([1.0, 0, 0, 0, 0, 0])

run_identical_twin = False
run_first_guess_bias = True
run_first_guess_rmse = False
run_promised_land = False
run_bed_measurements = False
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
    bias_range = [0]  # extra 1
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
    set2[5] = 1e-2
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
        #plot_iterative_behaviour(idir.gdir, idir.inv_settings['inversion_subdir'])