import copy
import numpy as np

# here specifiy the glaciers which should be used for the experiments
use_experiment_glaciers = ['Baltoro',  # 'Hintereisferner'  # 'Aletsch',
                           # 'Artesonraju', 'Shallap'
                           ]

# here you can define glacier specific settings
# {'RGI_ID': {'individual_setting_1': value}}
inversion_settings_individual = None

# general description of the current experiments
general_description = 'test_functionality'

# define different experiments here with file_suffixes for experiment
# description, all possible combinations will be generated (caution reg
# parameters for all observations needed
experiment_options = \
    {
        'control_vars': {'con1': ['area_bed_h'],
                         #'con2': ['bed_h']
                         },

        'observations': {'obs1': ['fl_surface_h:m'],
                         #'obs2': ['fl_surface_h:m', 'dmdtda:kg m-2 yr-1'],
                         },
        'obs_scaling_parameters': {
            'obsreg1': {'uncertainty': {
                'fl_surface_h:m': {'absolute': 1.},
                'dmdtda:kg m-2 yr-1': {'absolute': 1.}}},
            # 'obsreg2': {'uncertainty': {
            #     'fl_surface_h:m': 10.,
            #     'dmdtda:kg m-2 yr-1': 1.}},
            # 'obsreg3': {'uncertainty': {
            #     'fl_surface_h:m': 100.,
            #     'dmdtda:kg m-2 yr-1': 1.}},
            # 'obsreg4': {'uncertainty': {
            #     'fl_surface_h:m': 1.,
            #     'dmdtda:kg m-2 yr-1': 10.}},
            # 'obsreg5': {'uncertainty': {
            #     'fl_surface_h:m': 1.,
            #     'dmdtda:kg m-2 yr-1': 100.}},
        },
        'regularisation_terms': {'reg1': {'smoothed_bed': 0., 'bed_h_grad_scale': 0},
                                 # 'reg2': {'smoothed_bed': 1., 'bed_h_grad_scale': 0},
                                 # 'reg3': {'smoothed_bed': 10., 'bed_h_grad_scale': 0},
                                 # 'reg4': {'smoothed_bed': 100., 'bed_h_grad_scale': 0},
                                 # 'reg5': {'smoothed_bed': 1000., 'bed_h_grad_scale': 0},
                                 },

        # Options: 'surface_h', 'height_shift'
        'spinup_options': {'spn1': {'height_shift': {'mb_model':
            {
                'type': 'constant',
                'years': np.array([1980, 2000]),
                'fg_height_shift': -100}}},
            # 'spn2': {'height_shift': {'mb_model':
            #     {
            #         'type': 'constant',
            #         'years': np.array([1980, 2000]),
            #         'fg_height_shift': -150}}},
            # 'spn3': {'height_shift': {'mb_model':
            #     {
            #         'type': 'constant',
            #         'years': np.array([1980, 2000]),
            #         'fg_height_shift': -200}}},
        }
    }
# here define the different default inversion settings options for the experiments
default_inversion_settings_options = \
    {
        # Options: 'bed_h', 'surface_h', 'lambdas', 'w0_m'
        'control_vars': ['bed_h'],

        # Options: define explicitly e.g. {'MB1: {'type': 'constant', 'years':
        # np.array([1999, 2005])} OR 'load_from_gdir' for loading from gdir
        'mb_models_settings': 'load_from_gdir',

        # Used observations (are loaded from gdir)
        # Options: fl_surface_h:m, fl_widths:m, fl_total_area:m2, fl_total_area:km2,
        # area:m2, area:km2, dmdtda:kg m-2 yr-1
        'observations': ['fl_surface_h:m', 'fl_total_area:km2', 'dmdtda:kg m-2 yr-1'],
        'obs_scaling_parameters': {'scale': {'fl_surface_h:m': 1.,
                                             'fl_total_area:km2': 1.,
                                             'dmdtda:kg m-2 yr-1': 1.}},

        # Options: 'smoothed_bed'
        'regularisation_terms': {'smoothed_bed': 1.},

        # Options: 'surface_h', 'height_shift'
        'spinup_options': {'height_shift': {'mb_model': {
            'type': 'constant',
            'years': np.array([1980, 2000]),
            'fg_height_shift': -100}}},

        # For options check \
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.
        'minimize_options': {'maxiter': 5, 'ftol': 1e-7, 'gtol': 1e-8,
                             'disp': False, 'maxcor': 50, 'maxls': 50},
        'solver': 'L-BFGS-B',

        'max_time_minimize': 60 * 60 * 2,  # in s

        # pytorch settings
        # Options: 'double', 'float'
        'torch_type': 'double',
        # Options: 'cpu', 'cuda'
        'device': 'cpu',

        # limits for control variables
        'min_w0_m': 10.,
        'min_ice_thickness': 0.1,
        'max_ice_thickness': 1000.,
        'max_deviation_surface_h': 1000.,
        'limits_lambda': (0., 4.),
        'limits_height_shift_spinup': (-1000., 1000.),

        'experiment_description': general_description + '-',
    }


def recursive_define_inversion_setting(inv_var_list, tmp_inversion_setting):
    tmp_description = copy.deepcopy(tmp_inversion_setting['experiment_description'])
    inv_var = inv_var_list.pop(0)
    for inv_var_opt in experiment_options[inv_var]:
        tmp_inversion_setting['experiment_description'] = tmp_description + \
                                                          '_' + inv_var_opt
        tmp_inversion_setting[inv_var] = experiment_options[inv_var][inv_var_opt]

        if len(inv_var_list) == 0:
            inversion_settings_all.append(copy.deepcopy(tmp_inversion_setting))
        else:
            recursive_define_inversion_setting(copy.deepcopy(inv_var_list),
                                               tmp_inversion_setting)


# set all experiment combinations together
inversion_settings_all = []
all_experiment_options = list(experiment_options.keys())
recursive_define_inversion_setting(copy.deepcopy(all_experiment_options),
                                   default_inversion_settings_options)
