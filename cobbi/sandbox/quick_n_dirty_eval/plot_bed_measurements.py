import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import numpy as np
import glob
import os
from cobbi.core.data_logging import load_pickle
from cobbi.sandbox.quick_n_dirty_eval import experiment_naming_engine
from cobbi.core.visualization import MidpointNormalize, truncate_colormap,\
    imshow_ic, plot_glacier_contours, add_colorbar, get_axes_coords
from cobbi.core import test_cases
from oggm import cfg


cfg.initialize()

output_dir = '/media/philipp/Daten/Dokumente/Studium/Master/Masterarbeit' \
           '/Thesis/figs/bed_measurements'
basedir = '/home/philipp/thesis/'
file_extension = 'pdf'


figsize = (4.5, 3)

def plot_bed_measurement(measurement_noise, filepath, case, cbar_min,
                        cbar_max, title=None, ice_mask=None,
                        show_cbar=True, norm=None, cmap='bwr'):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(get_axes_coords(case))
    im_b = imshow_ic(ax, measurement_noise, case, cmap=cmap, ticks=False,
                     #norm=norm,  # norm turns out to be superfluent if
                     # using clim on coolorbar
                     vmin=cbar_min, vmax=cbar_max)
    cbar = add_colorbar(fig, ax, im_b, norm=norm, extend='neither')
    cbar.set_label('bed measurement error  (m)')
    cbar_min_max = max(abs(cbar_min), abs(cbar_max))
    cbar.set_clim(-cbar_min_max, cbar_min_max)
    if not show_cbar:
        cbar.remove()
    if title is not None:
        ax.set_title(title)
    if ice_mask is not None:
        plot_glacier_contours(ax, ice_mask, case)
    plot_glacier_contours(ax, ~measurement_noise.mask, case, colors='k',
                          linestyles='solid', linewidths=[1.])
    plt.savefig(filepath)
    plt.close(fig)


for case in [test_cases.Giluwe, test_cases.Borden]:
    filepaths = glob.glob(os.path.join(basedir,
                                       '*/{:s}/*/data_logger.pkl'.format(
                                           case.name)))
    filepaths = sorted(filepaths)
    for path in filepaths:
        idir, temp = os.path.split(path)
        gdir, exp = os.path.split(idir)
        dl = load_pickle(path)
        exp_name = experiment_naming_engine.get_experiment_name(exp)
        if exp_name is not None:

            if exp_name.endswith('plus bed'):
                ice_mask = np.load(os.path.join(gdir, 'ref_ice_mask.npy'))
                bed_measurements = np.load(os.path.join(idir,
                                                       'bed_measurements.pkl'))

                measurement_noise = bed_measurements - dl.true_bed
                np.ma.set_fill_value(measurement_noise, np.nan)
                cbar_min = measurement_noise.min()
                cbar_max = measurement_noise.max()
                cbar_min_max = max(abs(cbar_min), abs(cbar_max))
                norm = MidpointNormalize(midpoint=0., vmin=-cbar_min_max,
                                         vmax=cbar_min_max,
                                         mask=measurement_noise.mask)
                #my_cmap = sns.diverging_palette(240, 15, l=40, s=99, as_cmap=True)
                my_cmap = plt.get_cmap('PRGn')

                plotpath = os.path.join(output_dir,
                                        '{:s}_{:s}_measurement_noise.{:s}'.format(
                                            case.name, exp_name, file_extension))
                plot_bed_measurement(measurement_noise, plotpath, case,
                                    ice_mask=ice_mask,
                                    cbar_min=cbar_min, cbar_max=cbar_max,
                                    show_cbar=True, norm=norm, cmap=my_cmap)
                    #'Bed errors case {:s}\n ''experiment {:s}'.format(case,exp_name),
                #exit()