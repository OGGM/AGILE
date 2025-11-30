# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.5
#   kernelspec:
#     display_name: Python [conda env:agile_env]
#     language: python
#     name: conda-env-agile_env-py
# ---

import sys
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
import importlib.util
import os
import pickle
from oggm import utils
import copy
import seaborn as sns

# # add my functions and data paths

output_folder = os.environ["FIGURES_DIR"]

# ## functions

# +
# importing tools for analysis
paths_to_functions = [
    os.path.join(os.path.abspath("../"), 'tools_for_analysis'),
    os.path.abspath("../../../../"),
]

for path in paths_to_functions:
    sys.path.append(path)

from df_analysis import filter_dataframe, sort_dataframe, return_settings
from get_reference_glaciers import get_fl_at_year, fp_gdirs
from plot_statistics import plot_stats_per_iteration, plot_stat_metrics
from plot_along_fl import add_bed_h_and_volume_along_fl
# -

sys.path.append(os.path.abspath("../../"))
from glaciers_for_idealized_experiments import experiment_glaciers

# ## experiment settings

experiment_dir = os.environ["OUTPUT_DIR"]

from experiment_file_fg_oggm import experiment_options

# # Set working settings for plot

lam = 'lam9'
obs = 'obs6'
reg = 'reg0'

# # Define colors

# +
iterations = np.arange(0, 21, 5)

# Define a colormap and extract colors
cmap = matplotlib.colormaps['viridis']
colors = [cmap(i) for i in np.linspace(0, 1, len(iterations) + 1)]

# Override the color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
# -

# # Create plot Cost function and stats per iteration (fg_oggm and fg_glabtop)

main_results_dir = os.environ["OUTPUT_DIR"]


def get_ds_oggm(glacier, glacier_state):
    fp_pkl = os.path.join(
        main_results_dir,
        f'{glacier}_{glacier_state}_full_run_fg_oggm_{lam}_{obs}_{reg}.pkl')
    with open(fp_pkl, 'rb') as handle:
        ds_oggm = pickle.load(handle)
    return ds_oggm


def get_ds_glabtop(glacier, glacier_state):
    fp_pkl = os.path.join(
        main_results_dir,
        f'{glacier}_{glacier_state}_full_run_fg_glabtop_{lam}_{obs}_{reg}.pkl')
    with open(fp_pkl, 'rb') as handle:
        ds_glabtop = pickle.load(handle)
    return ds_glabtop


# # plotting function for single glacier and fg

def plot_single_glacier_along_fl(axs, glacier, glacier_state, fg_method,
                                 iterations=np.arange(0, 21, 5),
                                 scale_by_length=True
                                ):

    if fg_method == 'oggm':
        ds = get_ds_oggm(glacier, glacier_state)
    elif fg_method == 'glabtop':
        ds = get_ds_glabtop(glacier, glacier_state)
    else:
        raise ValueError(fg_method)

    fg_title = {
        'oggm': 'OGGM',
        'glabtop': 'GlabTop',
    }

    add_bed_h_and_volume_along_fl(axs, ds,
                               glacier=glacier,
                               glacier_state=glacier_state,
                               iterations=iterations,
                               title=f'{glacier} {glacier_state}, First guess {fg_title[fg_method]}',
                               scale_by_length=scale_by_length,
                              )

    for ax in axs:
        ax.grid('on')


# # plot showing detailed evolution of bed_h and volume along flowline

# ## Aletsch retreating

# +
glacier = 'Aletsch'
glacier_state = 'retreating'
fg_method = 'oggm'

panel_labels = ['a)', 'b)', 'c)']

output_filename = f'{glacier}_{glacier_state}_fg_{fg_method}_diff_along_fl'

fig, axs = plt.subplots(3, 1, figsize=(5,4))#, sharex=True)

plot_single_glacier_along_fl(axs, glacier, glacier_state, fg_method,
                             iterations=iterations)

for panel_label, ax in zip(panel_labels,
                               axs):
        ax.text(
            0.01, 0.98, panel_label,
            va='top',
            ha='left',
            transform=ax.transAxes,
            fontsize=11,
            zorder=10,
        )

fig.tight_layout(pad=0.3)

plt.savefig(os.path.join(output_folder, f'{output_filename}.png'),
                dpi=300, bbox_inches='tight')
# -


