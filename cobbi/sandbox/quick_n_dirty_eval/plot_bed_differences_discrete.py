import matplotlib.pyplot as plt
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from oggm import cfg
# import matplotlib.ticker as ticker
# import numpy as np
# from mpl_toolkits.axes_grid1 import make_axes_locatable

from cobbi.core import test_cases
from cobbi.core.data_logging import load_pickle
from cobbi.core.visualization import MidpointNormalize

# from cobbi.core.visualization import get_axes_coords
# from cobbi.core.visualization import plot_glacier_contours, imshow_ic
# from matplotlib.colors import ListedColormap
from cobbi.core.visualization import plot_differences_discrete_cmap
from cobbi.sandbox.quick_n_dirty_eval import experiment_naming_engine

cfg.initialize()

output_dir = '/home/philipp/final/plots/bed_diff/discrete'
basedir = '/home/philipp/final/'
file_extension = 'pdf'

figsize = (4.5, 3)

minval = -210
maxval = 210
min_max_val = max(abs(minval), abs(maxval))
cbar_steps = 22

# def add_colorbar(fig, ax, mappable, norm=None, boundaries=None,
#                  extend='neither'):
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     cbar = fig.colorbar(mappable, ax=ax, cax=cax, extend=extend,
#                         boundaries=boundaries)
#     cbar.outline.set_visible(False)
#     tick_locator = ticker.MaxNLocator(nbins=5)
#     cbar.locator = tick_locator
#     cbar.update_ticks()
#     #cbar.outline.set_linewidth(0.75)
#     return cbar
#
#
# def plot_differences_discrete_cmap(difference, filepath, case, cbar_min,
#                                    cbar_max, title=None, ice_mask=None,
#                                    bed_measurements=None, show_cbar=True,
#                                    norm=None, cmap='bwr', figsize=(4.5, 3),
#                                    cbar_label=None, existing_fig=None, n=21):
#     if type(cmap) is str:
#         cmap = plt.get_cmap(cmap)
#     cmap = ListedColormap(cmap(np.linspace(0, 1, n + 1)))
#     cbar_min_max = max(abs(cbar_min), abs(cbar_max))
#     bounds = np.linspace(-cbar_min_max, cbar_min_max, n)
#     bounds_step = bounds[1] - bounds[0]
#     bounds = bounds[
#         np.logical_and(bounds + bounds_step >= cbar_min,
#                        bounds - bounds_step <= cbar_max)]
#
#     fig = existing_fig
#     if existing_fig is None:
#         fig = plt.figure(figsize=figsize)
#
#     ax = fig.add_axes(get_axes_coords(case))
#     im_b = imshow_ic(ax, difference, case, cmap=cmap, ticks=False,
#                      norm=norm,
#                      vmin=cbar_min, vmax=cbar_max)
#     cbar = add_colorbar(fig, ax, im_b,
#                         norm=norm,
#                         boundaries=bounds,
#                         extend='neither')
#     cbar.set_label(cbar_label)
#     cbar.set_clim(-cbar_min_max, cbar_min_max)
#     if not show_cbar:
#         cbar.remove()
#     if title is not None:
#         ax.set_title(title)
#     if ice_mask is not None:
#         plot_glacier_contours(ax, ice_mask, case)
#     if bed_measurements is not None:
#         plot_glacier_contours(ax, ~bed_measurements.mask, case, colors='k',
#                               linestyles='solid', linewidths=[2.])
#     plt.savefig(filepath)
#     if existing_fig is None:
#        plt.close(fig)

for case in [test_cases.Giluwe, test_cases.Borden]:
    filepaths = glob.glob(os.path.join(basedir,
                                       '{:s}/*/data_logger.pkl'.format(
                                           case.name)))
    filepaths = sorted(filepaths)
    for path in filepaths:
        idir, temp = os.path.split(path)
        gdir, exp = os.path.split(idir)
        dl = load_pickle(path)
        exp_name = experiment_naming_engine.get_experiment_name2(exp)
        if exp_name is not None and len(dl.step_indices) > 0:
            ice_mask = np.load(os.path.join(gdir, 'ref_ice_mask.npy'))
            bed_measurements = None
            if exp_name.startswith('bed measurements'):
                bed_measurements = np.load(os.path.join(idir,
                                                        'bed_measurements.pkl'))

            diff_first_guess = dl.first_guessed_bed - dl.true_bed
            diff_optimized = dl.beds[-1] - dl.true_bed
            cbar_min = min(diff_first_guess.min(), diff_optimized.min())
            cbar_max = max(diff_first_guess.max(), diff_optimized.max())
            cbar_min_max = max(abs(cbar_min), abs(cbar_max))
            norm = MidpointNormalize(midpoint=0., vmin=-cbar_min_max,
                                     vmax=cbar_min_max)
            # my_cmap = sns.diverging_palette(240, 15, l=40, s=99, as_cmap=True)
            my_cmap = plt.get_cmap('seismic')
            plotpath = os.path.join(output_dir,
                                    '{:s}_{:s}_first_guess_bed_error.{'
                                    ':s}'.format(
                                        case.name,
                                        exp_name.replace('*', ' plus bed'),
                                        file_extension))
            plot_differences_discrete_cmap(
                diff_first_guess, plotpath, case, cbar_min=-min_max_val,
                cbar_max=min_max_val, ice_mask=ice_mask,
                bed_measurements=bed_measurements, show_cbar=False,
                norm=norm, cmap=my_cmap, figsize=figsize, n=cbar_steps,
                cbar_label='Bed elevation error (m)')

            plotpath = os.path.join(output_dir,
                                    '{:s}_{:s}_bed_error.{:s}'.format(
                                        case.name,
                                        exp_name.replace('*', ' plus bed'),
                                        file_extension))
            plot_differences_discrete_cmap(
                diff_optimized, plotpath, case, ice_mask=ice_mask,
                bed_measurements=bed_measurements, cbar_min=-min_max_val,
                cbar_max=min_max_val, show_cbar=True, norm=norm, cmap=my_cmap,
                figsize=figsize, n=cbar_steps,
                cbar_label='Bed elevation error (m)')
            # 'Bed errors case {:s}\n ''experiment {:s}'.format(case,exp_name),
            # exit()

            # Some more for bed measurements
            if bed_measurements is not None:
                no_meas_name = exp_name + ' without bed'
                meas_name = exp_name + ' with bed'
                no_meas_folder_name = 'promised land 3 10'
                dl_no_meas = load_pickle(os.path.join(basedir, case.name,
                                                      no_meas_folder_name,
                                                      'data_logger.pkl'))
                diff_no_meas = dl_no_meas.beds[-1] - dl_no_meas.true_bed
                cbar_min = min(diff_no_meas.min(), diff_optimized.min())
                cbar_max = max(diff_no_meas.max(), diff_optimized.max())
                cbar_min_max = max(abs(cbar_min), abs(cbar_max))
                norm = MidpointNormalize(midpoint=0., vmin=-cbar_min_max,
                                         vmax=cbar_min_max)

                plotpath = os.path.join(output_dir,
                                        '{:s}_{:s}_bed_error.{'
                                        ':s}'.format(
                                            case.name, no_meas_name,
                                            file_extension))
                plot_differences_discrete_cmap(
                    diff_no_meas, plotpath, case, cbar_min=-min_max_val,
                    cbar_max=min_max_val, ice_mask=ice_mask,
                    bed_measurements=bed_measurements, show_cbar=False,
                    norm=norm, cmap=my_cmap, figsize=figsize, n=cbar_steps,
                    cbar_label='Bed elevation error (m)')

                plotpath = os.path.join(output_dir,
                                        '{:s}_{:s}_bed_error.{:s}'.format(
                                            case.name, meas_name,
                                            file_extension))
                plot_differences_discrete_cmap(
                    diff_optimized, plotpath, case, cbar_min=-min_max_val,
                    cbar_max=min_max_val, ice_mask=ice_mask,
                    bed_measurements=bed_measurements, show_cbar=True,
                    norm=norm, cmap=my_cmap, figsize=figsize, n=cbar_steps,
                    cbar_label='Bed elevation error (m)')
