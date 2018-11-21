import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from cobbi.core.data_logging import load_pickle
from cobbi.core.visualization import MidpointNormalize, plot_glacier_contours
from cobbi.sandbox.quick_n_dirty_eval import experiment_naming_engine
from oggm import cfg

cfg.initialize()

basedir = '/home/philipp/thesis/'
filepaths = glob.glob(os.path.join(basedir, '*/*/*/data_logger.pkl'))
plot_dir = '/home/philipp/plots/inv_surf_differences/'

def plot_final_surface_difference(data_logger, filepath, title,
                        cbar_min=None,
                        cbar_max=None, ice_mask=None):
    return plot_surface_difference(data_logger.surfs[-1] -
                                   data_logger.ref_surf,
                               filepath, title, cbar_min, cbar_max)

def plot_surface_difference(surface_difference, filepath, title,
                        cbar_min=None,
                        cbar_max=None, ice_mask=None):
    if cbar_min is None:
        cbar_min = surface_difference.min()
    if cbar_max is None:
        cbar_max = surface_difference.max()
    fig, ax = plt.subplots()
    im_b = ax.imshow(surface_difference, cmap='RdBu_r',
                      norm=MidpointNormalize(midpoint=0., vmin=cbar_min,
                                             vmax=cbar_max))
    cbar = fig.colorbar(im_b)
    cbar.set_label('$\Delta$surface  height  (m)')
    ax.set_title(title)
    if ice_mask is not None:
        plot_glacier_contours(ax, ice_mask, linestyles='dashed',
                              linewidths=[0.75])
    plt.savefig(filepath)
    plt.clf()


for path in filepaths:
    idir, temp = os.path.split(path)
    gdir, exp = os.path.split(idir)
    temp, case = os.path.split(gdir)
    dl = load_pickle(path)
    exp_name = experiment_naming_engine.get_experiment_name(exp)
    if exp_name is not None:
        #if os.path.exists()
        cbar_min = None
        cbar_max = None

        surf_err = dl.surfs[-1] - dl.ref_surf
        cbar_min = surf_err.min()
        cbar_max = surf_err.max()

        ice_mask = np.load(os.path.join(gdir, 'ref_ice_mask.npy'))

        if os.path.exists(os.path.join(idir, 'dem_noise.npy')):
            dem_noise = np.load(os.path.join(idir, 'dem_noise.npy'))
            cbar_min = min(cbar_min, dem_noise.min())
            cbar_max = max(cbar_max, dem_noise.max())
            title = 'Surface perturbations case {:s}\nexperiment {:s}'.format(
                case, exp_name)
            plotpath = os.path.join(plot_dir,
                                    '{:s}_{:s}_surface_perturbations'
                                    '.png'.format(case,exp_name))
            plot_surface_difference(dem_noise, plotpath, title,
                                    cbar_min, cbar_max, ice_mask=ice_mask)

        title = 'Surface errors case {:s}\nexperiment {:s}'.format(case,
                                                                   exp_name)
        plotpath = os.path.join(plot_dir,
                                '{:s}_{:s}_surface_error.png'.format(case,
                                                                 exp_name))

        plot_surface_difference(surf_err, plotpath, title,
                                cbar_min, cbar_max, ice_mask=ice_mask)
