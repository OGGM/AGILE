import os
import copy
import pickle
import matplotlib.pyplot as plt
import matplotlib
from oggm import utils, workflow, cfg
import numpy as np
from agile1d.sandbox.calculate_statistics import add_1d_stats
from agile1d.sandbox.create_glaciers_with_measurements import create_idealized_experiments
import itertools

# initialize gdirs to get first guess later
cfg.initialize()
cfg.PARAMS['border'] = 160
cfg.PARAMS['use_multiprocessing'] = True

WORKING_DIR = os.environ["OGGM_WORKDIR"]
utils.mkdir(WORKING_DIR)
cfg.PATHS['working_dir'] = WORKING_DIR

from_prepro_level = 2
base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/' \
           'L1-L2_files/elev_bands/'
gcm='BCC-CSM2-MR'
ssp='ssp370'
glaciers = ['Aletsch', 'Baltoro', 'Artesonraju', 'Peyto']
glacier_states = ['equilibrium', 'retreating', 'advancing']

gdirs = create_idealized_experiments(glaciers,
                                 glacier_states,
                                 prepro_border=cfg.PARAMS['border'],
                                 from_prepro_level=from_prepro_level,
                                 base_url=base_url,
                                 gcm=gcm, ssp=ssp)

def change_dict_key(d, old_key, new_key, default_value=None):
    if old_key[-1] == '_':
        pass
    else:
        d[new_key] = d.pop(old_key, default_value)

def get_files_containing_phrase(all_files, phrases):
    return_files = copy.deepcopy(all_files)
    for phrase in phrases:
        return_files = [file for file in return_files if phrase in file]
    return return_files

def all_score_line_plots(glaciers, glacier_states, features, measure, input_folder,
                         extra_phrases=[],
                             experiment_description='',
                             compare_to_fg=True,  # False, True, 'dynamic', 'static'
                             length_top_list=10, length_worst_list=10,
                             filename='', output_folder='', save_fig=True,
                             add_pseudo_3d=False, add_feature_importance_plot=False,
                             add_l_curve_matrix=False, add_score_line=True,
                             add_score_board=False):
    # get all files of input folder
    all_files = os.listdir(input_folder)
    
    # create a template for the feature scores
    feature_scores_tmpl = {}
    for main_feature in features:
        for abbreviation in features[main_feature]:
            feature_scores_tmpl[abbreviation] = 0

    # function to go recusivly through features and sum up score, also maintain top list,
    # get Jobs and Jreg for L-Curve
    def open_file_and_get_score(phrases, features, feature_scores, feature_super_score, feature_bad_count,
                                feature_time_of_occurence,
                                list_of_all_scores, gdir, super_score, l_curve_data, glacier_state):
        keys = list(features.keys())
        
        current_feature = keys[0]
        for realisation in features[current_feature]:
            current_phrases = copy.deepcopy(phrases)
            current_phrases.append(realisation)

            # if their are more than one feature left we call the function again
            if len(keys) > 1:
                next_features = copy.deepcopy(features)
                next_features.pop(current_feature)
                
                open_file_and_get_score(
                    current_phrases, next_features, feature_scores, feature_super_score, feature_bad_count,
                    feature_time_of_occurence,
                    list_of_all_scores, gdir, super_score, l_curve_data, glacier_state)

            # ok we are left with one feature, so open file and get score
            else:
                file_name = get_files_containing_phrase(all_files, current_phrases)
                if len(file_name) > 1:
                    raise NotImplementedError(f'more than one filename found with '
                                              f'{current_phrases}: {file_name}! '
                                              'Need to be more specific!')
                elif len(file_name) == 0:
                    raise NotImplementedError(f'no filename found with {current_phrases}!')
                with open(os.path.join(input_folder, file_name[0]), 'rb') as handle:
                    open_file = pickle.load(handle)

                try:
                    if measure == 'mad_bed_h':
                        current_score = open_file.attrs['controls_stats']['area_bed_h']['mean_ad']
                    elif measure == 'mad_today_volume':
                        # need to recompute because of a bug
                        fl_mdl_end = open_file.flowlines[-1].item()
                        fl_true_end = gdir.read_pickle('model_flowlines',
                                                       filesuffix=f'_agile_true_end_{glacier_state}')[0]
                        current_score = add_1d_stats(fl_mdl_end.section * fl_mdl_end.dx_meter,
                                                     fl_true_end.section * fl_true_end.dx_meter)['mean_ad']
                    elif measure == 'mad_past_volume':
                        current_score = open_file.attrs['past_state_stats']['volume_m3']['mean_ad']
                    elif measure == 'total_today_volume':
                        # need to recompute because of a bug
                        fl_mdl_end = open_file.flowlines[-1].item()
                        fl_true_end = gdir.read_pickle('model_flowlines',
                                                       filesuffix=f'_agile_true_end_{glacier_state}')[0]
                        current_score = np.abs(np.sum(
                            add_1d_stats(fl_mdl_end.section * fl_mdl_end.dx_meter,
                                         fl_true_end.section * fl_true_end.dx_meter)['diff']))
                    else:
                        raise NotImplementedError(f'{measure}')
                except:
                    current_score = np.inf
                
                # add individual score
                list_of_all_scores.append((file_name[0], current_score))

                if add_l_curve_matrix:
                    # extract L-Curve data
                    l_curve_data[file_name[0]] = {
                        'lambda': open_file.attrs['cost_lambda'],
                        'Jobs': open_file.c_terms_description[-1].item()['J_obs'],
                        'Jreg': open_file.c_terms_description[-1].item()['J_reg'],
                    }

    # loop thourgh all glaciers and calculate scores, and 
    for glacier in glaciers:
        for glacier_state in glacier_states:
            feature_scores = copy.deepcopy(feature_scores_tmpl)
            
            # create a dict for super scores, 0 = not defined, 1 = not good, 2 = good
            feature_super_score = copy.deepcopy(feature_scores_tmpl)
    
            # count the number of occurence below super score
            feature_bad_count = copy.deepcopy(feature_scores_tmpl)
            feature_time_of_occurence = copy.deepcopy(feature_scores_tmpl)
    
            list_of_all_scores = []
            l_curve_data = {}
    
            rgi_id = {'Baltoro': 'RGI60-14.06794', 'Aletsch': 'RGI60-11.01450',
                      'Artesonraju': 'RGI60-16.02444', 'Peyto': 'RGI60-02.05098',
                     }[glacier]
            for gdir_tmp in gdirs:
                if gdir_tmp.rgi_id == rgi_id:
                    gdir = gdir_tmp
    
            if compare_to_fg:
                # add first guess run scores
                with open(os.path.join(gdir.dir, f'default_oggm_statistics_{glacier_state}.pkl'), 'rb') as handle:
                    oggm_default_stats = pickle.load(handle)
    
                super_score = None
                if measure == 'mad_bed_h':
                    stats = 'controls_stats'
                    for run, stats_suffix in zip(['oggm_dynamic_spinup', 'oggm_fixed_geometry'],
                                                 ['_dynamic', '_static']):
                        score_tmp = oggm_default_stats[stats + stats_suffix]['bed_h']['mean_ad']
                        list_of_all_scores.append((run, score_tmp))
    
                        if stats_suffix == '_static':
                            super_score = score_tmp
    
                elif measure == 'mad_today_volume':
                    stats = 'today_state_stats'
                    for run, stats_suffix in zip(['oggm_dynamic_spinup', 'oggm_fixed_geometry'],
                                                 ['_dynamic', '_static']):
                        score_tmp = oggm_default_stats[stats + stats_suffix]['volume_m3']['mean_ad']
                        list_of_all_scores.append((run, score_tmp))
    
                        if stats_suffix == '_static':
                            super_score = score_tmp
    
                elif measure == 'mad_past_volume':
                    stats = 'past_state_stats'
                    for run, stats_suffix in zip(['oggm_dynamic_spinup', 'oggm_fixed_geometry'],
                                                 ['_dynamic', '_static']):
                        score_tmp = oggm_default_stats[stats + stats_suffix]['volume_m3']['mean_ad']
                        list_of_all_scores.append((run, score_tmp))
    
                        if stats_suffix == '_static':
                            super_score = score_tmp
    
                elif measure == 'total_today_volume':
                    stats = 'today_state_stats'
                    for run, stats_suffix in zip(['oggm_dynamic_spinup', 'oggm_fixed_geometry'],
                                                 ['_dynamic', '_static']):
                        score_tmp = np.abs(np.sum(oggm_default_stats[stats + stats_suffix]['volume_m3']['diff']))
                        list_of_all_scores.append((run, score_tmp))
    
                        if stats_suffix == '_static':
                            super_score = score_tmp
    
                else:
                    raise NotImplementedError(f'{measure}')
    
            # actually getting the scores and other stuff
            phrases = [glacier, glacier_state, '.pkl']
            for ex_phrase in extra_phrases:
                phrases.append(ex_phrase)
            open_file_and_get_score(phrases, features,
                                    feature_scores, feature_super_score, feature_bad_count,
                                    feature_time_of_occurence,
                                    list_of_all_scores, gdir, super_score, l_curve_data, glacier_state)
    
            # sort all runs and assign ranking number
            all_scores_sorted = [
                (f'{i + 1}.', score[0], score[1])
                for i, score in
                enumerate(sorted(list_of_all_scores, key = lambda x: x[1], reverse=False))]
    
            # sort bad counts
            feature_bad_count_list = []
            for feature_tmp in feature_bad_count:
                if feature_bad_count[feature_tmp] > 0:
                    feature_bad_count_list.append((feature_tmp,
                                                   feature_bad_count[feature_tmp],
                                                   feature_time_of_occurence[feature_tmp]))
            feature_bad_count_sorted = sorted(feature_bad_count_list, key = lambda x: x[1], reverse=True)
            feature_bad_count_sorted = sorted(feature_bad_count_sorted, key = lambda x: x[0][0], reverse=True)
    
            if compare_to_fg:
                # get rank number of first guess
                oggm_dynamic_rank = None
                oggm_static_rank = None
                for score_tmp in all_scores_sorted:
                    if score_tmp[1] == 'oggm_dynamic_spinup':
                        oggm_dynamic_rank = int(float(score_tmp[0]))
                    if score_tmp[1] == 'oggm_fixed_geometry':
                        oggm_static_rank = int(float(score_tmp[0]))
                    if oggm_dynamic_rank is not None:
                        if oggm_static_rank is not None:
                            break
        
                # get features which are always worse than first guess
                feature_worse_fg = copy.deepcopy(feature_scores_tmpl)
                if isinstance(compare_to_fg, str):
                    if compare_to_fg == 'dynamic':
                        max_score_to_compare = oggm_dynamic_rank
                    elif compare_to_fg == 'static':
                        max_score_to_compare = oggm_static_rank
                    else:
                        raise NotImplementedError(f'{compare_to_fg}')
                else:
                    max_score_to_compare = max(oggm_dynamic_rank, oggm_static_rank)
                for score_tmp in all_scores_sorted:
                    if int(float(score_tmp[0])) > max_score_to_compare:
                        break
                    for feature in feature_worse_fg:
                        if feature in score_tmp[1]:
                            feature_worse_fg[feature] = 1
    
            if add_l_curve_matrix:
                all_obs = list(features['observations'].keys())
                if 'regularisation_terms' in features.keys():
                    all_reg = list(features['regularisation_terms'].keys())
                else:
                    all_reg = None
                all_lam = list(features['cost_lambda'].keys())
                
                l_curve_phrases = [glacier, '.pkl']
                add_single_value_features(l_curve_phrases, features)
    
                title = f'{glacier}, {experiment_description}, {measure}'
                plot_l_curve_matrix(l_curve_phrases, l_curve_data, all_obs, all_reg, all_lam, title)
    
            if add_pseudo_3d:
                if 'regularisation_terms' in features.keys():
                    x = list(features['cost_lambda'].keys())
                    z = list(features['observations'].keys())
                    y = list(features['regularisation_terms'].keys())
                else:
                    x = list(features['cost_lambda'].keys())
                    y = list(features['observations'].keys())
                    z = None
                ref = super_score
                data = list_of_all_scores
                sorted_data = all_scores_sorted
                measure_title = {
                    'mad_bed_h': 'MAD bed height',
                    'mad_today_volume': 'MAD 2020 dist. volume',
                }[measure]
                title = f'{glacier}, {experiment_description}, {measure_title}'
                plot_pseudo_3D(x, y, z, ref, data, sorted_data, title,
                               add_text=True)

            if add_score_line:
                fig, ax = plt.subplots(figsize=(5, 2))

                plot_score_line(ax, all_scores_sorted)

                measure_title = {
                    'mad_bed_h': 'MAD bed height in m',
                    'mad_today_volume': 'MAD 2020 dist. volume in m³',
                }[measure]
                ax.set_title(f'{glacier} {glacier_state}, {measure_title}')

                if save_fig:
                    fig.savefig(os.path.join(output_folder,
                                             f'{glacier}_{glacier_state}_{extra_phrases[0]}_{measure}.png'), 
                                facecolor='white', bbox_inches='tight')
                plt.show()
    
            if add_feature_importance_plot:
                # create plot with feature box plots and top_score list
                # depending on number of features create figure layout
                fig_rows = int((len(features) - 1) / 3) + 1
                
                fig, axs = plt.subplots(fig_rows, 3, layout='constrained', figsize=(20, fig_rows * 2))
                
                fig.suptitle(f'{glacier} ({measure})')
                for i, feature in enumerate(features.keys()):
                    # get the individual score values
                    score_labels = [abbre for abbre in features[feature]]
                    score_values = [feature_scores[abbre] for abbre in features[feature]]
                    if compare_to_fg:
                        bar_colors = [{0: 'red', 1: 'blue'}[feature_worse_fg[abbre]]
                                      for abbre in features[feature]]
                    else:
                        bar_colors = ['blue' for abbre in features[feature]]
                    
                    row = int(i / 3)
                    col = i % 3
                    if fig_rows > 1:
                        ax = axs[row][col]
                    else:
                        ax = axs[col]
                    
                    ax.bar(score_labels, score_values, color=bar_colors)
        
                    ax.set_title(feature)
                    ax.tick_params(axis='x', labelrotation=90)
                    ax_range = max(score_values) - min(score_values)
                    if ax_range == 0:
                        ax_range = 1
                    ax.set_ylim((min(score_values) - 0.05 * ax_range,
                                 max(score_values) + 0.05 * ax_range))
    
            else:
                if add_score_board:
                    fig, axs = plt.subplots(1, 1, layout='constrained', figsize=(20, 1))

            if add_score_board:
            # add top score board at end of plot
                top_scores = all_scores_sorted[:length_top_list]
                final_str = 'Top Score List\n'
                for score in top_scores:
                    final_str += str(score)
                    final_str += '\n'
        
                if compare_to_fg:
                    final_str += f'\n{all_scores_sorted[oggm_dynamic_rank-1]}\n'
                    final_str += f'{all_scores_sorted[oggm_static_rank-1]}\n'
        
                worst_scores = all_scores_sorted[-length_worst_list:]
                final_str += '\nWorst Score List\n'
                for score in worst_scores:
                    final_str += str(score)
                    final_str += '\n'
        
                # add super settings
                final_str += f'\nSuper Settings (always better than {print_super_settings} * fg = {super_score})\n'
                for feature_tmp in feature_super_score:
                    if feature_super_score[feature_tmp] == 2:
                        final_str += f'{feature_tmp}, '
        
                # print bad count
                final_str += '\n\nCount of bad occurences (below super score)\n'
                for bad_item_tmp in feature_bad_count_sorted:
                    final_str += f'\n{bad_item_tmp}'
                # add all option descritpions at the end as well
                final_str += '\n\n'
                final_str += 'Selected options\n'
                for feature in features:
                    final_str += '\n' + str(feature) + '\n'
                    for abbre in features[feature]:
                        final_str += f'{abbre}: {features[feature][abbre]}\n'
        
                plt.figtext(0.5, -0.02, final_str, horizontalalignment='center', verticalalignment='top', fontsize=15)
                
                if save_fig:
                    utils.mkdir(output_folder)
                    
                    fig.savefig(os.path.join(output_folder, filename + f'_{glacier}.png'), 
                                facecolor='white', bbox_inches='tight')
                
                plt.show()

    #return all_scores_sorted

def plot_score_line(ax, sorted_list):
    # get fg value to compere to
    for entry in sorted_list:
        if entry[1] == 'oggm_fixed_geometry':
            fg_rank = entry[0]
            fg_score = entry[2]
            break
    
    # define plotting values with colors and labels
    c1 = 'C0'
    c2 = 'C1'
    x = []
    y = []
    bar_colors = []
    bar_labels = []
    for entry in sorted_list:
        x.append(entry[0])
        y.append(entry[2] / fg_score)
        if entry[2] >= fg_score:
            bar_colors.append(c2)
        else:
            bar_colors.append(c1)
        if entry[1] == 'oggm_dynamic_spinup':
            bar_labels.append(f'OGGM\nDYN\n{entry[0]}')
        elif entry[1] == 'oggm_fixed_geometry':
            bar_labels.append(f'OGGM\nSTAT\n{entry[0]}')
        else:
            bar_labels.append('')
    
        if entry[2] == sorted_list[-1][2]:
            if entry[1] not in ['oggm_dynamic_spinup', 'oggm_fixed_geometry']:
                bar_labels[-1] = f'{bar_labels[-1]}\n{entry[0]}'

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(10, 1))

    ax.bar(x, y, color=bar_colors, width=1)
    ax.axhline(1, zorder=0, color='gray', ls='--', lw=1)
    #ax.text(float(x[0]), 1, 'OGGM DYN score', ha='right', va='center')
    ax.set_xticks(x, bar_labels)
    ax.set_xlim([float(x[0]), float(x[-1])])


fg_options = [#'fg_glabtop',
              'fg_oggm']

# create feature dict
start_ex = -4  # == 10^start_ex
end_ex = 3
nr_lam = 29
all_lambdas = np.logspace(start_ex, end_ex, num=nr_lam)
lam_dict = {'lam0': 0
}
for i, lam in enumerate(all_lambdas):
    lam_dict[f'lam{i + 1}'] = lam

features = \
    {
        'cost_lambda': lam_dict,
        'observations': {
            'obs0': {'fl_surface_h:m': {}},
            'obs1': {'dmdtda:kg m-2 yr-1': {}},
            'obs2': {'volume:km3': {}},
            'obs3': {'fl_surface_h:m': {},
                     'dmdtda:kg m-2 yr-1': {}},
            'obs4': {'fl_surface_h:m': {},
                     'volume:km3': {}},
            'obs5': {'dmdtda:kg m-2 yr-1': {},
                     'volume:km3': {}},
            'obs6': {'fl_surface_h:m': {},
                     'dmdtda:kg m-2 yr-1': {},
                     'volume:km3': {}},
        },
        'regularisation_terms': {
            'reg0': {'smoothed_bed': 1},
            'reg1': {'smoothed_flux': 0.1},
            'reg2': {'smoothed_bed': 1,
                      'smoothed_flux': 1e-5},
            'reg3': {'smoothed_bed': 1,
                      'smoothed_flux': 1e-4},
            'reg4': {'smoothed_bed': 1,
                      'smoothed_flux': 1e-3},
            'reg5': {'smoothed_bed': 1,
                      'smoothed_flux': 1e-2},
            'reg6': {'smoothed_bed': 1,
                      'smoothed_flux': 1e-1},
            'reg7': {'smoothed_bed': 1,
                      'smoothed_flux': 1},
            'reg8': {'smoothed_bed': 1,
                      'smoothed_flux': 1e1},
            'reg9': {'smoothed_bed': 1,
                      'smoothed_flux': 1e2},
        },
    }

for key in copy.deepcopy(features['cost_lambda']):
    change_dict_key(features['cost_lambda'],
                    key,
                    f'{key}_')

input_folder = '/home/www/pschmitt/agile/final_runs_glacier_states/full_run/results/'
output_folder = '/home/www/pschmitt/agile/final_runs_glacier_states/full_run/simple_score_plots_oggm_workshop24_updated_title/'

for fg_method in fg_options:
    for measure in ['mad_bed_h', 'mad_today_volume']:
        all_scores_sorted = all_score_line_plots(glaciers, glacier_states, features, measure, input_folder,
                            extra_phrases=[fg_method], output_folder=output_folder)
