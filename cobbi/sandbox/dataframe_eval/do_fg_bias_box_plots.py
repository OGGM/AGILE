import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
import os

basedir = '/home/philipp/final/'
output_dir = '/home/philipp/final/plots'
file_extension = 'pdf'
flierprops = dict(marker='.', markerfacecolor='blue', markersize=5,
                  linestyle='none')

case = Giluwe
gdir = NonRGIGlacierDirectory(case, basedir)
ref_ice_mask = np.load(gdir.get_filepath('ref_ice_mask'))
df = pd.read_pickle(os.path.join(basedir,
                                 '{:s} dataframe.pkl'.format(case.name)))
print(df.columns.to_series().groupby(df.dtypes).groups)
fg_bias_df = df.loc[df['experimentgroup'] == 'fg bias'].copy()
del df
for key, fg_bias_grp in fg_bias_df.groupby(['experimentsubgroup']):
    fig, ax = plt.subplots()
    fg_rmse = fg_bias_grp['firstguessrmse'].values.tolist()
    fg_rmse = np.array([fg_rmse[0]] + fg_rmse + [fg_rmse[-1]])
    fg_rmse_index = np.arange(0, fg_rmse.size)
    ax.fill_between(fg_rmse_index, fg_rmse, -fg_rmse, alpha=0.2, color='gray')
    bed_errs = fg_bias_grp['optimizedbederror'].map(
        lambda x: x[ref_ice_mask]).values
    bplot = ax.boxplot(bed_errs, showfliers=True, whis=[5, 95],
                       flierprops=flierprops, patch_artist=True, widths=0.8)
    for patch in bplot['boxes']:
        patch.set_facecolor('olivedrab')
    x_ticklabels = fg_bias_grp['subgroupindex'].map(
        lambda x: '' if round(x) % 10 > 0 else str(round(x))
    ).values
    ax.set_xticklabels(x_ticklabels)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(labelleft=True, labelright=True, direction='in')
    ax.set_xlabel('First guess bias (m)')
    ax.set_ylabel('Bed elevation error (m)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
                             '{:s}_{:s}_boxplot_bed_errors.{:s}'.format(
                                 key, case.name, file_extension)))

    fig, ax = plt.subplots()
    surf_errs = fg_bias_grp['optimizedsurferror'].map(
        lambda x: x[ref_ice_mask]).values
    bplot = ax.boxplot(surf_errs, showfliers=True, whis=[5, 95],
                       flierprops=flierprops, patch_artist=True, widths=0.8)
    for patch in bplot['boxes']:
        patch.set_facecolor('cornflowerblue')
    x_ticklabels = fg_bias_grp['subgroupindex'].map(
        lambda x: '' if round(x) % 10 > 0 else str(round(x))
    ).values
    ax.set_xticklabels(x_ticklabels)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(labelleft=True, labelright=True, direction='in')
    ax.set_xlabel('First guess bias (m)')
    ax.set_ylabel('Surface elevation error (m)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
                             '{:s}_{:s}_boxplot_surf_errors.{:s}'.format(
                                 key, case.name, file_extension)))

    fig, ax = plt.subplots()
    fg_bias_grp.plot(x='firstguessbias', y='iterations', kind='bar', ax=ax)
    ax.set_xticklabels(x_ticklabels)
    ax.set_xlabel('First guess bias (m)')
    ax.set_ylabel('Iterations #')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
                             '{:s}_{:s}_iterations.{:s}'.format(
                                 key, case.name, file_extension)))

    del fg_bias_grp

    # norm = MidpointNormalize(midpoint=0., vmin=-1,
    #                         vmax=1)
    n = 21
    cmap = plt.get_cmap('RdBu_r')
    norm = BoundaryNorm(np.linspace(-1, 1, n, endpoint=True), n)
    for quantity, errs in zip(['bed', 'surf'], [bed_errs, surf_errs]):
        fig, ax = plt.subplots()
        cmap = ListedColormap(cmap(np.linspace(0, 1, n, endpoint=True)))
        corr_matrix = np.corrcoef(np.vstack(errs))
        mat = ax.matshow(corr_matrix, cmap=cmap, vmin=corr_matrix.min(),
                         vmax=corr_matrix.max())
        ax.set_xticks(np.arange(bed_errs.size)[::2])
        ax.set_yticks(np.arange(bed_errs.size)[::2])
        ax.set_xticklabels(x_ticklabels[::2])
        ax.set_yticklabels(x_ticklabels[::2])
        # ax.set_xlabel('First guess bias (m)')
        ax.set_ylabel('First guess bias (m)')
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
            output_dir, '{:s}_{:s}_boxplot_{:s}_error_corr_matrix.{:s}'.format(
                key, case.name, quantity, file_extension)))
