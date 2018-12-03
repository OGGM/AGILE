import matplotlib.pyplot as plt
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from oggm import cfg

from cobbi.core import test_cases
from cobbi.core.data_logging import load_pickle
from cobbi.core.visualization import MidpointNormalize, plot_bed_difference
from cobbi.sandbox.quick_n_dirty_eval import experiment_naming_engine

cfg.initialize()

output_dir = '/home/philipp/final/plots/bed_diff'
basedir = '/home/philipp/final/'
file_extension = 'pdf'


figsize = (4.5, 3)


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
            #my_cmap = sns.diverging_palette(240, 15, l=40, s=99, as_cmap=True)
            my_cmap = plt.get_cmap('seismic')
            plotpath = os.path.join(output_dir,
                                    '{:s}_{:s}_first_guess_bed_error.{'
                                    ':s}'.format(
                                        case.name,
                                        exp_name.replace('*', ' plus bed'),
                                        file_extension))
            plot_bed_difference(diff_first_guess, plotpath, case,
                                ice_mask=ice_mask,
                                bed_measurements=bed_measurements,
                                cbar_min=cbar_min, cbar_max=cbar_max,
                                show_cbar=False, norm=norm, cmap=my_cmap,
                                figsize=figsize)

            plotpath = os.path.join(output_dir,
                                    '{:s}_{:s}_bed_error.{:s}'.format(
                                        case.name,
                                        exp_name.replace('*', ' plus bed'),
                                        file_extension))
            plot_bed_difference(diff_optimized, plotpath, case,
                                ice_mask=ice_mask,
                                bed_measurements=bed_measurements,
                                cbar_min=cbar_min, cbar_max=cbar_max,
                                show_cbar=True, norm=norm, cmap=my_cmap,
                                figsize=figsize)
                #'Bed errors case {:s}\n ''experiment {:s}'.format(case,exp_name),
            #exit()

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
                plot_bed_difference(diff_no_meas, plotpath, case,
                                    ice_mask=ice_mask,
                                    bed_measurements=bed_measurements,
                                    cbar_min=cbar_min, cbar_max=cbar_max,
                                    show_cbar=False, norm=norm, cmap=my_cmap)

                plotpath = os.path.join(output_dir,
                                        '{:s}_{:s}_bed_error.{:s}'.format(
                                            case.name, meas_name,
                                            file_extension))
                plot_bed_difference(diff_optimized, plotpath, case,
                                    ice_mask=ice_mask,
                                    bed_measurements=bed_measurements,
                                    cbar_min=cbar_min, cbar_max=cbar_max,
                                    show_cbar=True, norm=norm, cmap=my_cmap)