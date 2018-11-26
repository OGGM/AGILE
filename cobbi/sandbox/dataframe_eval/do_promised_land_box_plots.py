import numpy as np
import pandas as pd
import os
# from scipy.stats.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import BoundaryNorm, ListedColormap
from cobbi.core.data_logging import load_pickle
from cobbi.core.visualization import MidpointNormalize
from cobbi.sandbox.quick_n_dirty_eval.experiment_naming_engine import \
    get_experiment_name2
from cobbi.core.utils import NonRGIGlacierDirectory
from cobbi.core.test_cases import Borden, Giluwe
from oggm import cfg;

cfg.initialize()

basedir = '/home/philipp/final/'
output_dir = '/home/philipp/final/plots'
file_extension = 'pdf'
flierprops = dict(marker='.', markerfacecolor='blue', markersize=5,
                  linestyle='none')

case = Borden
gdir = NonRGIGlacierDirectory(case, basedir)
ref_ice_mask = np.load(gdir.get_filepath('ref_ice_mask'))
df = pd.read_pickle(os.path.join(basedir,
                                 '{:s} dataframe.pkl'.format(case.name)))
promised_land_df = df.loc[df['experimentgroup'] == 'promised land'].copy()
del df
# mask too large rmse giving trouble in Giluwe
# if case is Giluwe:
#    fg_rmse_df = fg_rmse_df.loc[fg_rmse_df['subgroupindex'] < 57]

sgrps = promised_land_df.groupby(['experimentsubgroup'])
base_positions = np.array([2, 4, 6, 8, 10, 12])
width = 0.5
spacer = 0.05
separate_at = [0, 3, 5, 7, 9, 11, 14]
available_fill_colors = ['sienna', 'olivedrab', 'mediumpurple',
                         'cornflowerblue']
# ==== boxplot bed ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
for sep in separate_at:
    ax.axvline(sep, color='k', linestyle='--', linewidth=0.75)
ax.axhline(0, color='gray', linestyle='dotted', linewidth=0.75)
for i, subgroup in enumerate(sgrps):
    key, promised_land_grp = subgroup
    # fg_rmse = promised_land_grp['firstguessrmse'].values.tolist()
    # fg_rmse = np.array([fg_rmse[0]] + fg_rmse + [fg_rmse[-1]])
    # fg_rmse_index = np.arange(0, 61)
    # ax.fill_between(fg_rmse_index, fg_rmse_index, -fg_rmse_index,
    #                alpha=0.2, color='gray')
    # ax.axhline(0, color='k')

    bed_errs = promised_land_grp['optimizedbederror'].map(
        lambda x: x[ref_ice_mask]).values
    positions = base_positions + (width + spacer) * (i - 1)
    bplot = ax.boxplot(bed_errs, positions=positions, showfliers=True,
                       whis=[5, 95], flierprops=flierprops,
                       patch_artist=True, widths=width)
    for patch in bplot['boxes']:
        patch.set_facecolor(available_fill_colors[i])
# x_ticks = np.arange(0, 61, 5)
ax.set_xticks(base_positions)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('none')
ax.tick_params(labelleft=True, labelright=False, labelbottom=True,
               direction='in')

ax.set_ylabel('Bed elevation error (m)')
ax.set_xlabel('First guess RMSE (m)')
ax.set_xlim([1., 13.])
ax.set_xticklabels(base_positions)
ax.tick_params(labelbottom=True)

legend_elements = [Patch(facecolor=available_fill_colors[i], edgecolor='k',
                         label=str(i + 1))
                   for i in range(len(sgrps))]
loc = 'upper center'
leg = ax.legend(handles=legend_elements,
                # labels=[str(i) for i in range(len(custom_lines))],
                ncol=3, title='surface noise pattern', loc=loc,
                frameon=True, facecolor='white', fancybox=False,
                framealpha=1.0, edgecolor='k')

plt.tight_layout()
plt.savefig(os.path.join(
    output_dir, 'promised_land_{:s}_boxplot_bed_errors.{:s}'.format(
        case.name, file_extension)))

# ==== boxplot surf ===========================================================
fig, ax = plt.subplots(figsize=(8, 5))
for sep in separate_at:
    ax.axvline(sep, color='k', linestyle='--', linewidth=0.75)
ax.axhline(0, color='gray', linestyle='dotted', linewidth=0.75)
for rmse in base_positions:
    ax.fill_between([(rmse - 1), (rmse + 1)], [rmse, rmse], [-rmse, -rmse],
                    color='gray', alpha=0.2)
for i, subgroup in enumerate(sgrps):
    key, promised_land_grp = subgroup
    # fg_rmse = promised_land_grp['firstguessrmse'].values.tolist()
    # fg_rmse = np.array([fg_rmse[0]] + fg_rmse + [fg_rmse[-1]])
    # fg_rmse_index = np.arange(0, 61)
    # ax.fill_between(fg_rmse_index, fg_rmse_index, -fg_rmse_index,
    #                alpha=0.2, color='gray')
    # ax.axhline(0, color='k')

    surf_errs = promised_land_grp['optimizedsurferror'].map(
        lambda x: x[ref_ice_mask]).values
    positions = base_positions + (width + spacer) * (i - 1)
    bplot = ax.boxplot(surf_errs, positions=positions, showfliers=True,
                       whis=[5, 95], flierprops=flierprops,
                       patch_artist=True, widths=width)
    for patch in bplot['boxes']:
        patch.set_facecolor(available_fill_colors[i])
# x_ticks = np.arange(0, 61, 5)
ax.set_xticks(base_positions)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('none')
ax.tick_params(labelleft=True, labelright=True, labelbottom=True,
               direction='in')

ax.set_ylabel('Surface elevation error (m)')
ax.set_xlabel('First guess RMSE (m)')
ax.set_xlim([1., 13.])
ax.set_xticklabels(base_positions)
ax.tick_params(labelbottom=True)

legend_elements = [Patch(facecolor=available_fill_colors[i], edgecolor='k',
                         label=str(i + 1))
                   for i in range(len(sgrps))]
loc = 'upper center'
leg = ax.legend(handles=legend_elements,
                # labels=[str(i) for i in range(len(custom_lines))],
                ncol=3, title='surface noise pattern', loc=loc,
                frameon=True, facecolor='white', fancybox=False,
                framealpha=1.0, edgecolor='k')

plt.tight_layout()
plt.savefig(os.path.join(
    output_dir, 'promised_land_{:s}_boxplot_surf_errors.{:s}'.format(
        case.name, file_extension)))

# ==== iteration plot =========================================================
fig, ax = plt.subplots()
for sep in separate_at:
    ax.axvline(sep, color='k', linestyle='--', linewidth=0.75)
for i, subgroup in enumerate(sgrps):
    key, promised_land_grp = subgroup
    positions = base_positions + (width + spacer) * (i - 1)
    ax.bar(x=positions, height=promised_land_grp['iterations'],
           width=width, color=available_fill_colors[i])

ax.set_xticks(base_positions)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('none')
ax.set_xlabel('Surface noise RMSE (m)')
ax.set_ylabel('Iterations #')
ax.set_xlim([1., 13.])
loc = 'upper left'
leg = ax.legend(handles=legend_elements,
                ncol=3, title='surface noise pattern', loc=loc,
                frameon=True, facecolor='white', fancybox=False,
                framealpha=1.0, edgecolor='k')
plt.tight_layout()
plt.savefig(os.path.join(output_dir,
                         '{:s}_{:s}_iterations.{:s}'.format(
                             'promised_land', case.name, file_extension)))

# ==== surf correlation line plot =============================================
fig, ax = plt.subplots(figsize=(4.5, 3))
# for sep in separate_at:
#    ax.axvline(sep, color='k', linestyle='--', linewidth=0.75)
# ax.axhline(0, color='gray', linestyle='dotted', linewidth=0.75)
# for rmse in base_positions:
#    ax.fill_between([(rmse - 1), (rmse + 1)], [rmse, rmse], [-rmse, -rmse],
#                    color='gray', alpha=0.2)
for i, subgroup in enumerate(sgrps):
    key, promised_land_grp = subgroup
    surf_errs = promised_land_grp['optimizedsurferror'].map(
        lambda x: x[ref_ice_mask]).values
    surf_noise = promised_land_grp['surfacenoise'].map(
        lambda x: x[ref_ice_mask]).values
    rmse = base_positions
    corr = []
    for j in range(len(rmse)):
        corr.append(np.corrcoef(surf_noise[j], surf_errs[j])[0, 1])
    ax.plot(rmse, corr, color=available_fill_colors[i],
            label=str(i + 1), marker='D')
ax.set_ylabel('Correlation coefficient')
ax.set_xlabel('Surface noise RMSE(m)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir,
                         '{:s}_{:s}_surf_noise_err_corr.{:s}'.format(
                             'promised_land', case.name, file_extension)))

# ==== correlation matrix plots ===============================================
n = 21
cmap = plt.get_cmap('RdBu_r')
norm = BoundaryNorm(np.linspace(-1, 1, n, endpoint=True), n)

# for i, subgroup in enumerate(sgrps):
# key, promised_land_grp = subgroup
bed_errs = promised_land_df['optimizedbederror'].map(
    lambda x: x[ref_ice_mask]).values
surf_errs = promised_land_df['optimizedsurferror'].map(
    lambda x: x[ref_ice_mask]).values
separate_at = []
for sgrp in sgrps:
    separate_at.append(sgrp[1].shape[0])
separate_at = np.cumsum(separate_at[:-1]) - 0.5
for quantity, errs in zip(['bed', 'surf'], [bed_errs, surf_errs]):
    fig, ax = plt.subplots()
    cmap = ListedColormap(cmap(np.linspace(0, 1, n, endpoint=True)))
    corr_matrix = np.corrcoef(np.vstack(errs))
    mat = ax.matshow(corr_matrix, cmap=cmap, vmin=corr_matrix.min(),
                     vmax=corr_matrix.max())
    x_ticklabels = [str(round(i)) for i in base_positions]
    for k in range(len(sgrps) - 1):
        x_ticklabels = x_ticklabels + x_ticklabels
    for sep in separate_at:
        ax.axhline(sep, color='w')
        ax.axvline(sep, color='w')
    ax.set_xticks(np.arange(bed_errs.size)[::2])
    ax.set_yticks(np.arange(bed_errs.size)[::2])
    ax.set_xticklabels(x_ticklabels[::2])
    ax.set_yticklabels(x_ticklabels[::2])
    # ax.set_xlabel('Surface noise RMSE (m)')
    ax.set_ylabel('Surface noise RMSE (m)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
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
        output_dir,
        'promised_land_{:s}_{:s}_{:s}_error_corr_matrix.{:s}'.format(
            key, case.name, quantity, file_extension)))
