import copy
import numpy as np
from agile1d.core.inversion import get_default_inversion_settings

# here specifiy the glaciers which should be used for the experiments
use_experiment_glaciers = ['Aletsch',
                           #'Baltoro',
                           #'Artesonraju',
                           #'Peyto'
                          ]

use_experiment_glacier_states = [#'equilibrium',
                                 'retreating',
                                 #'advancing'
                                 ]

# define some glacier specific parameters
inversion_settings_individual = None

# general description of the current experiments
general_description = 'full_run_fg_oggm'

# define different experiments here with file_suffixes for experiment
# description, all possible combinations will be generated
# 'key is inversion setting' = {'exp1': value, 'exp2': value}

# create lambdas dict
# set to True if you want to use all lambda values as in the publication
if False:
    start_ex = -4  # == 10^start_ex
    end_ex = 3
    nr_lam = 29
    all_lambdas = np.logspace(start_ex, end_ex, num=nr_lam)
    lam_dict = {'lam0': 0}
    for i, lam in enumerate(all_lambdas):
        lam_dict[f'lam{i + 1}'] = lam
else:
    # this is the lambda value used in the Aletsch retreating cases
    lam_dict = {'lam9': 0.01}

experiment_options = \
    {
        'cost_lambda': lam_dict,
        'observations': {
            # 'obs0': {'fl_surface_h:m': {}},
            # 'obs1': {'dmdtda:kg m-2 yr-1': {}},
            # 'obs2': {'volume:km3': {}},
            # 'obs3': {'fl_surface_h:m': {},
            #          'dmdtda:kg m-2 yr-1': {}},
            # 'obs4': {'fl_surface_h:m': {},
            #          'volume:km3': {}},
            # 'obs5': {'dmdtda:kg m-2 yr-1': {},
            #          'volume:km3': {}},
            'obs6': {'fl_surface_h:m': {},
                     'dmdtda:kg m-2 yr-1': {},
                     'volume:km3': {}},
        },
        'regularisation_terms': {
            'reg0': {'smoothed_bed': 1},
        },
    }

default_inversion_settings_options = get_default_inversion_settings()

default_inversion_settings_options['obs_scaling_parameters'] = {
    'uncertainty': {'fl_surface_h:m': {'absolute': 10.},
                    'dmdtda:kg m-2 yr-1': {'absolute': 100.},
                    'volume:km3': {'relative': 0.1}}
}

default_inversion_settings_options['spinup_options'] = {
    'section': {'extra_grid_points': 10,
                'limits': (0.6, 1.4),
                'fg_years': 0,
               }}
default_inversion_settings_options['experiment_description'] = general_description
default_inversion_settings_options['minimize_options']['disp'] = False
default_inversion_settings_options['minimize_options']['maxiter'] = 100
default_inversion_settings_options['max_time_minimize'] = 60 * 60 * 4,  # in s

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
