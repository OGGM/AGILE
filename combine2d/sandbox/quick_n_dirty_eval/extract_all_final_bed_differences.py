import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import glob
import os
from combine2d.core.data_logging import load_pickle
from combine2d.core.visualization import MidpointNormalize, plot_glacier_contours
from combine2d.sandbox.quick_n_dirty_eval import experiment_naming_engine
from combine2d.sandbox.quick_n_dirty_eval \
    .get_surrounding_line_for_bed_measurements import \
    get_surrounding_line
from oggm import cfg

cfg.initialize()

basedir = '/home/philipp/thesis/'
filepaths = glob.glob(os.path.join(basedir, '*/*/*/data_logger.pkl'))
plot_dir = '/home/philipp/plots/inv_bed_differences/'

def plot_final_bed_difference(data_logger, filepath, title, ice_mask=None,
                              bed_measurements=None):
    return plot_bed_difference(data_logger.beds[-1] - data_logger.true_bed,
                               filepath, title, ice_mask=ice_mask,
                               bed_measurements=bed_measurements)

def plot_bed_difference(bed_difference, filepath, title, cbar_min=None,
                        cbar_max=None, ice_mask=None,
                        bed_measurements=None):
    if cbar_min is None:
        cbar_min = bed_difference.min()
    if cbar_max is None:
        cbar_max = bed_difference.max()
    fig, ax = plt.subplots()
    im_b = ax.imshow(bed_difference, cmap='RdBu_r',
                      norm=MidpointNormalize(midpoint=0., vmin=cbar_min,
                                             vmax=cbar_max))

    cbar = plt.colorbar(im_b)
    cbar.set_label('$\Delta$bed  height  (m)')
    ax.set_title(title)
    if ice_mask is not None:
        plot_glacier_contours(ax, ice_mask, linestyles='dashed',
                              linewidths=[0.75])
    if bed_measurements is not None:
        plot_glacier_contours(ax, ~bed_measurements.mask, colors='k',
                              linestyles='solid', linewidths=[1.])
    plt.savefig(filepath)
    plt.clf()



for path in filepaths:
    idir, temp = os.path.split(path)
    gdir, exp = os.path.split(idir)
    temp, case = os.path.split(gdir)
    dl = load_pickle(path)
    exp_name = experiment_naming_engine.get_experiment_name(exp)
    if exp_name is not None:
        ice_mask = np.load(os.path.join(gdir, 'ref_ice_mask.npy'))
        bed_measurements = None
        if exp_name.endswith('plus bed'):
            bed_measurements = np.load(os.path.join(idir,
                                                   'bed_measurements.pkl'))
        plotpath = os.path.join(plot_dir,
                                '{:s}_{:s}_bed_error.png'.format(case,
                                                                 exp_name))
        plot_final_bed_difference(
            dl, plotpath, 'Bed errors case {:s}\n '
                          'experiment {:s}'.format(case,exp_name),
            ice_mask=ice_mask, bed_measurements=bed_measurements)
