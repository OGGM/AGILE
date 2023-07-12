import numpy as np
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from agile2d.core.data_logging import load_pickle
from agile2d.sandbox.quick_n_dirty_eval.experiment_naming_engine import \
    get_experiment_name2
from agile2d.core import test_cases

from oggm import cfg;
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from oggm import cfg;

from agile2d.core.data_logging import load_pickle
from agile2d.sandbox.quick_n_dirty_eval.experiment_naming_engine import \
    get_experiment_name2

cfg.initialize()
import os

basedir = '/media/philipp/Daten/Dokumente/Studium/Master/Masterarbeit' \
          '/Ergebnisse'

output_dir = '/media/philipp/Daten/Dokumente/Studium/Master/Masterarbeit' \
             '/Thesis/figs'

borden_experiments = [(('identical-twin',),
                       # ('identical-twin b', 'identical-twin c'),
                       ('identical-twin star',)),
                      (('first guess bias 1', 'first guess bias 2'),
                       ('first guess rmse 1', 'first guess rmse 2')),
                      (('promised land 1a', 'promised land 1b',
                        'promised land 1c'),
                       ('promised land 2a', 'promised land 2b',
                        'promised land 2c'),
                       ('promised land 3a',
                        'promised land 3b', 'promised land 3c'),
                       ('promised land 3c star',))]

giluwe_experiments = [(('identical-twin',),
                       # ('identical-twin b', 'identical-twin c'),
                       ('identical-twin star 1', 'identical-twin star 2')),
                      (('first guess bias 1', 'first guess bias 2'),
                       ('first guess rmse 1', 'first guess rmse 2')),
                      (('promised land 1a', 'promised land 1b',
                        'promised land 1c'),
                       ('promised land 2a', 'promised land 2b',
                        'promised land 2c'),
                       ('promised land 3a prime', 'promised land 3a',
                        'promised land 3b', 'promised land 3c'),
                       ('promised land 3c star 1', 'promised land 3c star 2'))]

for case, experiments in zip([test_cases.Giluwe, test_cases.Borden],
                             [giluwe_experiments, borden_experiments]):

    box_width = 0.5
    small_spacer = 0.1
    medium_spacer = 0.3
    large_spacer = 0.3
    available_fill_colors = ['sienna', 'olivedrab', 'mediumpurple',
                             'cornflowerblue']
    fill_colors = [[], []]
    known_reg_parameters = {}

    bed_differences = []
    surf_differences = []
    ice_mask = np.load(os.path.join(basedir, case.name,
                                    'ref_ice_mask.npy'))
    ice_mask.dtype = np.bool
    position = small_spacer

    all_positions = [[], []]
    all_widths = [[], box_width]
    labels = []
    hline_positions = []
    for exp_group in experiments:
        for exp_subgroup in exp_group:
            for exp in exp_subgroup:
                dl = load_pickle(os.path.join(basedir, case.name,
                                              exp,
                                              'data_logger.pkl'))
                inv_settings = load_pickle(
                    os.path.join(basedir, case.name, exp,
                                 'inversion_settings.pkl'))
                reg_parameters = str(inv_settings['reg_parameters'][:5])
                if not reg_parameters in known_reg_parameters:
                    known_reg_parameters[reg_parameters] = \
                        available_fill_colors[len(known_reg_parameters)]
                fill_colors[1].append(known_reg_parameters[reg_parameters])

                bed_diff = (dl.beds[-1] - dl.true_bed)[ice_mask]
                bed_differences.append(bed_diff)
                surf_diff = (dl.surfs[-1] - dl.ref_surf)[ice_mask]
                if os.path.exists(os.path.join(basedir, case.name,
                                               exp,
                                               'dem_noise.npy')):
                    surf_differences.append(
                        np.load(os.path.join(basedir, case.name, exp,
                                             'dem_noise.npy'))[ice_mask])
                    surf_differences.append(surf_diff)
                    all_widths[0].append(0.5 * box_width)
                    all_widths[0].append(0.5 * box_width)
                    all_positions[0].append(position + 0.25 * box_width)
                    all_positions[0].append(position + 0.75 * box_width)
                    fill_colors[0].append(known_reg_parameters[reg_parameters])
                    fill_colors[0].append(known_reg_parameters[reg_parameters])
                else:
                    surf_differences.append(surf_diff)
                    all_widths[0].append(box_width)
                    all_positions[0].append(position + 0.5 * box_width)
                    fill_colors[0].append(known_reg_parameters[reg_parameters])

                labels.append(get_experiment_name2(exp))
                # all_widths[0].append(box_width)
                all_positions[1].append(position + 0.5 * box_width)
                position += small_spacer + box_width
            position += medium_spacer
        hline_positions.append(position
                               + 0.5 * (
                                           large_spacer - medium_spacer - small_spacer))
        position += large_spacer

    diff_collections = [surf_differences, bed_differences]

    flierprops = dict(marker='.', markerfacecolor='blue', markersize=5,
                      linestyle='none')
    fig, ax_arr = plt.subplots(nrows=len(diff_collections), figsize=(9, 13))
    i = 0
    for diff_collection, ax, positions, box_widths, colors in zip(
            diff_collections, ax_arr, all_positions, all_widths, fill_colors):
        print(i);
        i = i + 1
        ax.set_xticks(all_positions[1])
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(labelleft=True, labelright=True)
        ax.xaxis.set_ticks_position('both')
        for hline_pos in hline_positions[:-1]:
            ax.axvline(hline_pos, color='gray', linestyle='dashed',
                       linewidth='0.5')
        ax.axhline(0, color='black', linestyle='--', linewidth=0.75)
        if diff_collection is bed_differences:
            # ax.axhline(-30, color='darkgreen', linestyle='--', linewidth=0.75)
            # ax.axhline(30, color='darkgreen', linestyle='--', linewidth=0.75)
            ax.axhspan(-30, 30, alpha=0.2, color='gray')
        print(len(diff_collection))
        print(len(positions))
        bplot = ax.boxplot(diff_collection, positions=positions,
                           widths=box_widths,
                           showfliers=True, flierprops=flierprops,
                           patch_artist=True,
                           whis=[5, 95])  # make whiskers show +- 95%

        for col, patch in zip(colors, bplot['boxes']):
            patch.set_facecolor(col)
        ax.tick_params(direction='in')

    if len(diff_collections) > 1:
        # ax_arr[0].set_xticklabels(labels, rotation=-30, ha='right')
        ax_arr[0].xaxis.set_ticks_position('top')
        ax_arr[0].xaxis.set_label_position('top')
        ax_arr[0].xaxis.set_ticks_position('both')
        ax_arr[0].tick_params(labeltop=False, labelbottom=False)
        ax_arr[0].set_ylabel('surface elevation error (m)')

    ax_arr[-1].set_xticklabels(labels, rotation=30, ha='right')
    ax_arr[-1].xaxis.set_label_position('bottom')
    custom_lines = [  # [Line2D([0], [0], alpha=0.0, visible=False)] + [
        Line2D([0], [0], color=known_reg_parameters[rp], lw=5) for
        rp in known_reg_parameters]
    tA = r'$\bf{A}$'
    tB = r'$\bf{B}$'
    tC = r'$\bf{C}$'
    # loc = 'upper center'
    # if case is test_cases.Borden:
    loc = 'lower center'
    leg = ax_arr[0].legend(custom_lines, [tA, tB, tC], ncol=4,
                           title='reg. parameter set:', loc=loc,
                           frameon=False, facecolor='white')
    ax_arr[-1].set_ylabel('bed elevation error (m)')
    if case is test_cases.Borden:
        ax_arr[-1].set_ylim([-380, None])
        print('Min_bed_error:')
        print(np.min(bed_differences))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, case.name + '_boxplot.pdf'))
    # plt.show()
