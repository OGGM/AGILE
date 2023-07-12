import numpy as np
import pandas as pd
import os
# from scipy.stats.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import BoundaryNorm, ListedColormap
from agile2d.core.data_logging import load_pickle
from agile2d.core.visualization import MidpointNormalize
from agile2d.sandbox.quick_n_dirty_eval.experiment_naming_engine import \
    get_experiment_name2
from agile2d.core.utils import NonRGIGlacierDirectory
from agile2d.core.test_cases import Borden, Giluwe
from oggm import cfg;

cfg.initialize()

basedir = '/home/philipp/final/'
output_dir = '/home/philipp/final/plots'
file_extension = 'pdf'
flierprops = dict(marker='.', markerfacecolor='blue', markersize=5,
                  linestyle='none')

available_fill_colors = ['sienna', 'olivedrab', 'mediumpurple',
                         'cornflowerblue']

case = Borden
gdir = NonRGIGlacierDirectory(case, basedir)
ref_ice_mask = np.load(gdir.get_filepath('ref_ice_mask'))
df = pd.read_pickle(os.path.join(basedir,
                                 '{:s} dataframe.pkl'.format(case.name)))
fg_rmse_df = df.loc[df['experimentgroup'] == 'fg rmse'].copy()
del df
# mask too large rmse giving trouble in Giluwe
if case is Giluwe:
    xlim_upper = 61
    indexlim_upper = 61
    fg_rmse_df = fg_rmse_df.loc[fg_rmse_df['subgroupindex'] < xlim_upper - 1]
if case is Borden:
    xlim_upper = 82
    indexlim_upper = 73

sgrps = fg_rmse_df.groupby(['experimentsubgroup'])

fig, ax_arr = plt.subplots(ncols=3, figsize=(9, 6), sharey=True)
for i, ax, subgroup in zip(list(range(ax_arr.shape[0])), ax_arr, sgrps):
    key, fg_rmse_grp = subgroup
    percentiles = {}

    fg_rmse = fg_rmse_grp['firstguessrmse'].values.tolist()
    fg_percentile_index = np.array([0] + fg_rmse + [xlim_upper])
    for perc in [5, 25, 75, 95]:
        fg_percentile = fg_rmse_grp[
            'firstguess_{:d}_percentile'.format(perc)
        ].values.tolist()
        fg_percentile = np.array(
            [fg_percentile[0]] + fg_percentile + [fg_percentile[-1]])
        percentiles[perc] = fg_percentile
    # fg_percentiles_index = np.arange(0, percentiles[perc].size)
    # fg_rmse_index = np.arange(0, indexlim_upper)
    ax.fill_between(fg_percentile_index, percentiles[95], percentiles[5],
                    alpha=0.15, color='gray')
    ax.fill_between(fg_percentile_index, percentiles[75], percentiles[25],
                    alpha=0.5, color='darkgray')
    # ax.fill_between(fg_rmse_index, fg_rmse_index, -fg_rmse_index,
    #                alpha=0.2, color='gray')
    ax.axhline(0, color='k')
    bed_errs = fg_rmse_grp['optimizedbederror'].map(
        lambda x: x[ref_ice_mask]).values
    positions = fg_rmse
    bplot = ax.boxplot(bed_errs, positions=positions, showfliers=True,
                       whis=[5, 95], flierprops=flierprops,
                       patch_artist=True, widths=3.5)
    for patch in bplot['boxes']:
        patch.set_facecolor(available_fill_colors[i])
    x_ticks = np.arange(0, xlim_upper, 5)
    ax.set_xticks(x_ticks)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(labelleft=False, labelright=False, labelbottom=True,
                   direction='in')

    ax.set_xlabel('First guess RMSE (m)')
    ax.set_xlim([1, xlim_upper])
    ax.set_xticklabels(['' if x % 10 > 0 else str(x)
                        for x in x_ticks])
    ax.tick_params(labelbottom=True)

ax_arr[0].set_ylabel('Bed elevation error (m)')
ax_arr[0].tick_params(labelleft=True)
ax_arr[-1].tick_params(labelright=True)
legend_elements = [Patch(facecolor=available_fill_colors[i], edgecolor='k',
                         label=str(i + 1))
                   for i in range(len(sgrps))]
loc = 'upper center'
leg = ax_arr[1].legend(handles=legend_elements,
                       # labels=[str(i) for i in range(len(custom_lines))],
                       ncol=3, title='first guess noise pattern', loc=loc,
                       frameon=True, facecolor='white', fancybox=False,
                       framealpha=0.5, edgecolor='k')

sample_size = bed_errs[0].size
ax_arr[0].text(0.05, 0.93, 'n = {:d}'.format(sample_size),
               horizontalalignment='left',
               verticalalignment='top',
               transform=ax_arr[0].transAxes, weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(
    output_dir, 'fg_rmse_{:s}_boxplot_bed_errors.{:s}'.format(
        case.name, file_extension)))
    # plt.show()

bed_errs = fg_rmse_df['optimizedbederror'].map(
    lambda x: x[ref_ice_mask]).values
surf_errs = fg_rmse_df['optimizedsurferror'].map(
    lambda x: x[ref_ice_mask]).values
n = 21
cmap = plt.get_cmap('RdBu_r')
norm = BoundaryNorm(np.linspace(-1, 1, n, endpoint=True), n)
separate_at = []
for sgrp in sgrps:
    separate_at.append(sgrp[1].shape[0])
x_ticklabels = []
base_positions = []
last_position = 0
for sep in separate_at:
    x_ticklabels = x_ticklabels + list(range(10, sep * 5 + 1, 20))
for sep in np.cumsum(separate_at):
    base_positions = base_positions + np.arange(last_position + 1,
                                                sep, 4).tolist()
    last_position = sep

separate_at = np.cumsum(separate_at[:-1]) - 0.5
for quantity, errs in zip(['bed', 'surf'], [bed_errs, surf_errs]):
    fig, ax = plt.subplots()  #figsize=(9, 6))
    cmap = ListedColormap(cmap(np.linspace(0, 1, n, endpoint=True)))
    corr_matrix = np.corrcoef(np.vstack(errs))
    mat = ax.matshow(corr_matrix, cmap=cmap, vmin=corr_matrix.min(),
                     vmax=corr_matrix.max())
    for sep in separate_at:
        ax.axhline(sep, color='w')
        ax.axvline(sep, color='w')

    # x_ticklabels = [str(round(i))
    #                for j in [1, 2, 3]
    #                for i in base_positions]
    ax.set_xticks(base_positions)
    ax.set_yticks(base_positions)
    ax.set_xticklabels(x_ticklabels)
    ax.set_yticklabels(x_ticklabels)
    # ax.set_xlabel('First guess RMSE (m)')
    ax.set_ylabel('Desired first guess RMSE (m)')

    for k in range(1, 4):
        fig.text(0.22 * k - 0.055, 0.02, 'noise pattern {:d}'.format(k))

        fig.text(0.8, 1.16 - 0.295 * k, 'noise pattern {:d}'.format(k),
                 rotation=90)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.tick_params(labeltop=False, labelbottom=True)
    ax.tick_params(top=True, bottom=False, left=True, right=False)
    bounds = np.linspace(-1, 1, n, endpoint=True)
    bounds_step = bounds[1] - bounds[0]
    bounds = bounds[
        np.logical_and(bounds + bounds_step >= corr_matrix.min(),
                       bounds - bounds_step <= corr_matrix.max())]
    cbar = plt.colorbar(mat, boundaries=bounds)
    cbar.set_clim(-1, 1)
    cbar.set_label('Correlation coefficient')
    cbar.outline.set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, 'fg_rmse_{:s}_boxplot_{:s}_error_corr_matrix.{'
                    ':s}'.format(
            case.name, quantity, file_extension)))

fig, ax = plt.subplots()
fg_rmse_df.plot(x='experiment', y='iterations', kind='bar', ax=ax, rot=90)
ax.set_xlabel('')
ax.set_ylabel('Iterations #')
plt.tight_layout()
plt.savefig(os.path.join(output_dir,
                         '{:s}_{:s}_iterations.{:s}'.format(
                             'fg_rmse', case.name, file_extension)))

"""
# Side by side plots
fig, ax_arr = plt.subplots(ncols=len(sgrps), sharey='all')
for (key, fg_rmse_grp), ax in zip(sgrps, ax_arr):
    fg_rmse = fg_rmse_grp['firstguessrmse'].values.tolist()
    #fg_rmse = np.array([fg_rmse[0]] + fg_rmse + [fg_rmse[-1]])
    fg_rmse_index = np.arange(0, 61)
    ax.fill_between(fg_rmse_index, fg_rmse_index, -fg_rmse_index,
                    alpha=0.2, color='gray')
    ax.axhline(0, color='k')
    bed_errs = fg_rmse_grp['optimizedbederror'].map(
        lambda x: x[ref_ice_mask]).values
    positions = fg_rmse
    bplot = ax.boxplot(bed_errs, positions=positions, showfliers=True,
                       whis=[5,95], flierprops=flierprops,
                       patch_artist=True, widths=3.5)
    for patch in bplot['boxes']:
        patch.set_facecolor('sienna')
    x_ticks = np.arange(0, 61, 5)
    ax.set_xticks(x_ticks)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(labelleft=False, labelright=False, labelbottom=False,
                   direction='in')

    ax.set_xlim([1, 58])
    ax.set_xticklabels(['' if x%10 > 0 else str(x)
                        for x in x_ticks])
    ax.tick_params(labelbottom=True)

ax_arr[0].tick_params(labelleft=True)
ax_arr[0].set_ylabel('Bed elevation error (m)')
ax_arr[1].set_xlabel('First guess rmse (m)')
ax_arr[-1].tick_params(labelright=True)
plt.tight_layout()
#plt.savefig(os.path.join(output_dir,
#                         '{:s}_{:s}_boxplot_bed_errors.{:s}'.format(
#                             key, case.name, file_extension)))
plt.show()
"""
