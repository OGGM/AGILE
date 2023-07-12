import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
from matplotlib.ticker import NullLocator
import matplotlib as mpl
import numpy as np
import glob
import os
from agile2d.core.data_logging import load_pickle
from agile2d.sandbox.quick_n_dirty_eval import experiment_naming_engine
from agile2d.core.visualization import MidpointNormalize, truncate_colormap, \
    imshow_ic, plot_glacier_contours, add_colorbar, get_axes_coords, \
    plot_differences_discrete_cmap
from agile2d.core import test_cases
from oggm import cfg

cfg.initialize()

output_dir = '/home/philipp/final/plots/noise'
basedir = '/home/philipp/final/'
file_extension = 'pdf'

figsize = (4.5, 3)

cbar_min = -37.5
cbar_max = 37.5
cbar_min_max = max(abs(cbar_min), abs(cbar_max))
cbar_steps = 16

for case in [test_cases.Giluwe, test_cases.Borden]:
    filepaths = glob.glob(os.path.join(basedir,
                                       '{:s}/*/dem_noise.npy'.format(
                                           case.name)))
    filepaths = sorted(filepaths)
    for path in filepaths:
        idir, temp = os.path.split(path)
        gdir, exp = os.path.split(idir)
        dl = load_pickle(os.path.join(idir, 'data_logger.pkl'))
        exp_name = experiment_naming_engine.get_experiment_name2(exp)
        dem_noise = np.load(path)
        if exp_name is not None and len(dl.step_indices) > 0:
            print(exp_name + ' ' + case.name)
            print(dem_noise.min())
            print(dem_noise.max())

            ice_mask = np.load(os.path.join(gdir, 'ref_ice_mask.npy'))

            norm = MidpointNormalize(midpoint=0., vmin=-cbar_min_max,
                                     vmax=cbar_min_max)
            # my_cmap = sns.diverging_palette(240, 15, l=40, s=99, as_cmap=True)
            my_cmap = plt.get_cmap('PuOr_r')
            plotpath = os.path.join(output_dir,
                                    '{:s}_{:s}_dem_noise.{'
                                    ':s}'.format(
                                        case.name,
                                        exp_name,
                                        file_extension))
            plot_differences_discrete_cmap(
                dem_noise, plotpath, case, ice_mask=ice_mask,
                cbar_min=cbar_min, cbar_max=cbar_max, show_cbar=False,
                norm=norm, cmap=my_cmap, figsize=figsize, n=cbar_steps)

cmap = ListedColormap(
    my_cmap(np.linspace(0, 1, cbar_steps - 1, endpoint=True)))
cbar_min_max = max(abs(cbar_min), abs(cbar_max))
bounds = np.linspace(-cbar_min_max, cbar_min_max, cbar_steps)
bounds_step = bounds[1] - bounds[0]
bounds = bounds[
    np.logical_and(bounds + bounds_step >= cbar_min,
                   bounds - bounds_step <= cbar_max)]

fig, ax = plt.subplots(figsize=(figsize[0] * 1.5, figsize[1] * 0.4))
cbar = mpl.colorbar.ColorbarBase(
    ax=ax, cmap=cmap, norm=norm, extend='neither', boundaries=bounds,
    orientation='horizontal')
cbar.outline.set_visible(False)
tick_locator = ticker.MaxNLocator(nbins=7)
cbar.locator = tick_locator
cbar.update_ticks()
cbar.set_label('Perlin noise (m)')

# cbar.outline.set_linewidth(0.75)
plt.tight_layout()
plotpath = os.path.join(output_dir,
                        'cbar_dem_noise.{:s}'.format(file_extension))
plt.savefig(plotpath)
