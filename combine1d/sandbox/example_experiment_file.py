import copy
import numpy as np

# here specifiy the glaciers which should be used for the experiments
use_experiment_glaciers = ['Hintereisferner', 'Baltoro', 'Aletsch',
                           'Artesonraju', 'Shallap']

# general description of the current experiments
general_description = 'test_functionality'

# define different experiments here with file_suffixes for experiment
# description, all possible combinations will be generated (caution reg
# parameters for all observations needed
experiment_options = \
    {
        'control_vars': {'con1': ['bed_h'],
                         'con2': ['bed_h', 'w0_m']},

        'observations': {'obs1': ['fl_surface_h:m', 'fl_widths:m'],
                         'obs2': ['fl_surface_h:m', 'fl_widths:m', 'dh:m']},

        'obs_reg_parameters': {'reg1': {'scale': {'fl_surface_h:m': 1.,
                                                  'fl_widths:m': 1.,
                                                  'dh:m': 1.}},
                               'reg2': {'scale': {'fl_surface_h:m': 0.1,
                                                  'fl_widths:m': 0.1,
                                                  'dh:m': 1.}}
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
        # area:m2, area:km2, dh:m
        'observations': ['fl_surface_h:m', 'fl_widths:m', 'dh:m'],
        'obs_reg_parameters': {'scale': {'fl_surface_h:m': 1.,
                                         'fl_widths:m': 1.,
                                         'dh:m': 1.}},

        # Options: 'smoothed_bed'
        'regularisation_terms': {'smoothed_bed': 1.},

        # Options: 'surface_h'
        'spinup_options': {'surface_h': {'t_bias': -2,
                                         'mb_model': {'type': 'constant',
                                                      'years': np.array([1980, 2000])}
                                         }
                           },

        # For options check \
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.
        'minimize_options': {'maxiter': 10, 'ftol': 1e-7, 'gtol': 1e-8,
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

        'experiment_description': general_description,
    }


def recursive_define_inversion_setting(inv_var_list, tmp_inversion_setting):
    # global inversion_settings_all
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
recursive_define_inversion_setting(all_experiment_options,
                                   default_inversion_settings_options)
