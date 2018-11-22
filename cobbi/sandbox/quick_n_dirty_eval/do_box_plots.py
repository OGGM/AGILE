import numpy as np
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from cobbi.core.data_logging import load_pickle
from cobbi.sandbox.quick_n_dirty_eval.experiment_naming_engine import \
    get_experiment_name2

from oggm import cfg;
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from oggm import cfg;

from cobbi.core.data_logging import load_pickle
from cobbi.sandbox.quick_n_dirty_eval.experiment_naming_engine import \
    get_experiment_name2

cfg.initialize()
import os

basedir = '/media/philipp/Daten/erstabgabe'

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
                       ('promised land 3c star ',))]

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

box_width = 0.5
small_spacer = 0.1
medium_spacer = 0.3
large_spacer = 0.3
available_fill_colors = ['sienna', 'olivedrab', 'mediumpurple',
                         'cornflowerblue']
fill_colors = []
known_reg_parameters = {}

bed_differences = []
surf_differences = []
ice_mask = np.load(os.path.join(basedir, 'Borden Peninsula',
                                'ref_ice_mask.npy'))
ice_mask.dtype = np.bool
position = small_spacer

all_positions = []
labels = []
hline_positions = []
for exp_group in borden_experiments:
    for exp_subgroup in exp_group:
        for exp in exp_subgroup:
            dl = load_pickle(os.path.join(basedir, 'Borden Peninsula',
                                          exp,
                                          'data_logger.pkl'))
            inv_settings = load_pickle(
                os.path.join(basedir, 'Borden Peninsula', exp,
                             'inversion_settings.pkl'))
            reg_parameters = str(inv_settings['reg_parameters'][:5])
            print(reg_parameters)
            if not reg_parameters in known_reg_parameters:
                known_reg_parameters[reg_parameters] = \
                    available_fill_colors[len(known_reg_parameters)]
            fill_colors.append(known_reg_parameters[reg_parameters])
            print(fill_colors[-1])

            bed_diff = (dl.beds[-1] - dl.true_bed)[ice_mask]
            bed_differences.append(bed_diff)
            surf_diff = (dl.surfs[-1] - dl.ref_surf)[ice_mask]
            surf_differences.append(surf_diff)
            labels.append(get_experiment_name2(exp))
            all_positions.append(position + 0.5 * box_width)
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
for diff_collection, ax in zip(diff_collections, ax_arr):
    ax.set_xticks(all_positions)
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
    bplot = ax.boxplot(diff_collection, positions=all_positions,
                       widths=0.6,
                       showfliers=True, flierprops=flierprops,
                       patch_artist=True)
    for col, patch in zip(fill_colors, bplot['boxes']):
        patch.set_facecolor(col)
    ax.tick_params(direction='in')

if len(diff_collections) > 1:
    # ax_arr[0].set_xticklabels(labels, rotation=-30, ha='right')
    ax_arr[0].xaxis.set_ticks_position('top')
    ax_arr[0].xaxis.set_label_position('top')
    ax_arr[0].xaxis.set_ticks_position('both')
    ax_arr[0].tick_params(labeltop=False, labelbottom=False)

ax_arr[-1].set_xticklabels(labels, rotation=30, ha='right')
ax_arr[-1].xaxis.set_label_position('bottom')
custom_lines = [  # [Line2D([0], [0], alpha=0.0, visible=False)] + [
    Line2D([0], [0], color=known_reg_parameters[rp], lw=5) for
    rp in known_reg_parameters]
tA = r'$\bf{A}$'
tB = r'$\bf{B}$'
tC = r'$\bf{C}$'
leg = ax.legend(custom_lines, [tA, tB, tC], ncol=4,
                title='reg. parameter set:', loc='lower center',
                frameon=False, facecolor='white')

plt.tight_layout()
plt.savefig(os.path.join(basedir, 'borden_boxplot.pdf'))
plt.show()
