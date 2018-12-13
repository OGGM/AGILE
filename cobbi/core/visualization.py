import os
import shutil

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
import numpy as np
import salem
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from cobbi.core.arithmetics import compute_inner_mask
from cobbi.core.cost_function import get_costs_arr
# fontprops = fm.FontProperties(size=18)
from cobbi.core.data_logging import load_pickle


class MidpointNormalize(colors.Normalize):
    # see: https://matplotlib.org/users/colormapnorms.html#custom-normalization-two-linear-ranges
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False,
                 mask=None):
        self.midpoint = midpoint
        self.mask = mask
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    #see: https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_gradient(filepath, gradient, title, ref_shape=None,
                  cbar_min=None, cbar_max=None):

        if cbar_min is None:
            cbar_min = min(-1e-4,np.min(gradient))
        if cbar_max is None:
            cbar_max = max(1e-4, np.max(gradient))
        if ref_shape is not None:
            gradient = gradient.reshape(ref_shape)

        fig = plt.figure()
        im = plt.imshow(gradient,
                        norm=MidpointNormalize(midpoint=0.,
                                               vmin=cbar_min,
                                               vmax=cbar_max),
                        cmap='RdBu')

        plt.title(title)
        cbar = plt.colorbar(im)
        cbar.set_label('Gradient of cost-function (m$^{-1}$)')
        plt.savefig(filepath)
        plt.close(fig)


def plot_glacier_contours(ax, ice_mask, case, resolution_enhance=1e1,
                          colors='gray', linestyles='dashed',
                          linewidths=[0.75]):
    func = lambda x, y: ice_mask[int(y), int(x)]
    g = np.vectorize(func)

    x = np.linspace(0, ice_mask.shape[1], ice_mask.shape[1] * resolution_enhance)
    y = np.linspace(0, ice_mask.shape[0], ice_mask.shape[0] * resolution_enhance)
    X, Y = np.meshgrid(x[:-1], y[:-1])
    Z = g(X[:-1], Y[:-1])
    if case.name == 'Giluwe':
           extent=[0-0.45, x[:-1].max()-0.5,
                   0-0.3, y[:-1].max()-0.4]
    elif case.name == 'Borden Peninsula':
        extent = [0 - 0.45, x[:-1].max() - 0.45,
                  0 - 0.3, y[:-1].max() - 0.48]
    elif case.name == 'Borden Peninsula HR':
        extent = [0 - 0.45, x[:-1].max() - 0.45,
                  0 - 0.3, y[:-1].max() - 0.48]
    elif case.name == 'Trikora':
           extent=[0-0.45, x[:-1].max()-0.5,
                   0-0.3, y[:-1].max()-0.4]

    ax.contour(Z[::-1], [0.5], colors=colors, linewidths=linewidths,
               extent=extent, linestyles=linestyles)


def plot_first_guess():
    f = plt.figure()
    #im_b = plt.imshow(first_bed_guess, cmap=new_cmap)
    #cbar = plt.colorbar(im_b)
    #cbar.set_label('Bed  height A.S.L (m)')
    #plt.title(
    #    'First Bed Guess of case {:s}, dx={:d}m'.format(case.name, case.dx))
    #fname = '{:s}_first_guess.png'.format(case.name)
    #plt.savefig(filepath)
    plt.clf()

def imshow_ic(ax, arr, case, cmap=None, ice_mask=None, ticks=True,
              norm=None, vmin=None, vmax=None):
    y, x = arr.shape
    im = ax.imshow(arr[::-1, :], cmap=cmap, origin='lower', norm=norm,
                   vmin=vmin, vmax=vmax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    #if case.name == 'Giluwe':
    #xlocs = np.arange(arr.shape[1])[::10]
    #xlabels = ['{:d}'.format(l) for l in xlocs]
    #ylocs = np.arange(arr.shape[1])[::10]
    #ylabels = ['{:d}'.format(l) for l in ylocs[::-1]]
    #ax.set_xticks(xlocs)
    #ax.set_xticklabels(xlabels)
    #ax.set_yticks(ylocs)
    #ax.set_yticklabels(ylabels)
    if ticks is True:
        @ticker.FuncFormatter
        def major_formatter(x, pos):
            return '{:g}'.format(x * case.dx * 1e-3)
        ax.get_xaxis().set_major_locator(ticker.MultipleLocator(8))
        ax.get_xaxis().set_major_formatter(major_formatter)
        ax.get_yaxis().set_major_locator(ticker.MultipleLocator(8))
        ax.get_yaxis().set_major_formatter(major_formatter)

        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')

    elif ticks is False:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    elif ticks == 'scalebar':
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        if case.name == 'Giluwe':
            pixels = 8
        elif case.name == 'Borden Peninsula':
            pixels = 10
        elif case.name == 'Borden Peninsula HR':
            pixels = 10
        elif case.name == 'Trikora':
            pixels = 20
        scalebar = AnchoredSizeBar(ax.transData, pixels,
                                   '{:g}km'.format(pixels * case.dx * 1e-3),
                                   1, frameon=True, pad=0.5, size_vertical=0.3,
                                   borderpad=1.5)
        ax.add_artist(scalebar)


    if ice_mask is not None:
        plot_glacier_contours(ax, ice_mask, case, linestyles='dashed',
                              linewidths=[0.75])
    return im

def get_extent(arr, case):
    y, x = arr.shape
    dx = case.dx
    xlim = 0.5 * x / dx
    ylim = 0.5 * y / dx
    return np.array([-xlim, xlim, -ylim, ylim])

def add_colorbar(fig, ax, mappable, norm=None, boundaries=None,
                 extend='neither'):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, ax=ax, cax=cax, extend=extend,
                        boundaries=boundaries)
    cbar.outline.set_visible(False)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    #cbar.outline.set_linewidth(0.75)
    return cbar

def get_axes_coords(case):
    if case.name == 'Giluwe':
        return [0., 0.015, 0.75, 0.97]  # left, bottom, width, height
    elif case.name == 'Borden Peninsula':
        return [0., 0.015, 0.75, 0.97]  # left, bottom, width, height
    elif case.name == 'Borden Peninsula HR':
        return [0., 0.015, 0.75, 0.97]  # left, bottom, width, height
    elif case.name == 'Trikora':
        return [0., 0.015, 0.75, 0.97]  # left, bottom, width, height


def plot_bed_difference(bed_difference, filepath, case, cbar_min,
                        cbar_max, title=None,
                        ice_mask=None, bed_measurements=None,
                        show_cbar=True, norm=None, cmap='bwr',
                        figsize=(4.5, 3), cbar_label='bed elevation error (m)',
                        existing_fig=None):
    plot_differences(bed_difference, filepath, case, cbar_min, cbar_max,
                     title, ice_mask, bed_measurements, show_cbar, norm,
                     cmap, figsize, cbar_label=cbar_label,
                     existing_fig=existing_fig)


def plot_differences(difference, filepath, case, cbar_min, cbar_max,
                     title=None, ice_mask=None, bed_measurements=None,
                     show_cbar=True, norm=None, cmap='bwr', figsize=(4.5, 3),
                     cbar_label=None, existing_fig=None):
    fig = existing_fig
    if existing_fig is None:
        fig = plt.figure(figsize=figsize)

    ax = fig.add_axes(get_axes_coords(case))
    im_b = imshow_ic(ax, difference, case, cmap=cmap, ticks=False,
                     norm=norm, vmin=cbar_min, vmax=cbar_max)
    cbar = add_colorbar(fig, ax, im_b, norm=norm, extend='neither')
    cbar.set_label(cbar_label)
    cbar_min_max = max(abs(cbar_min), abs(cbar_max))
    cbar.set_clim(-cbar_min_max, cbar_min_max)
    if not show_cbar:
        cbar.remove()
    if title is not None:
        ax.set_title(title)
    if ice_mask is not None:
        plot_glacier_contours(ax, ice_mask, case)
    if bed_measurements is not None:
        plot_glacier_contours(ax, ~bed_measurements.mask, case, colors='k',
                              linestyles='solid', linewidths=[2.])
    plt.savefig(filepath)
    if existing_fig is None:
        plt.close(fig)


def plot_surf_difference(surf_difference, filepath, case, cbar_min,
                         cbar_max, title=None,
                         ice_mask=None, bed_measurements=None,
                         show_cbar=True, norm=None, cmap='PuOr_r',
                         figsize=(4.5, 3),
                         cbar_label='surf elevation error (m)',
                         existing_fig=None):

    plot_differences(surf_difference, filepath, case, cbar_min, cbar_max,
                     title, ice_mask, bed_measurements, show_cbar, norm,
                     cmap, figsize, cbar_label=cbar_label,
                     existing_fig=existing_fig)


def get_subdir_filepath(gdir, subdir, filepath):
    my_dir, my_file = os.path.split(gdir.get_filepath(filepath))
    return os.path.join(my_dir, subdir, my_file)


def plot_iterative_behaviour(gdir, subdir, figsize=(4.5, 3),
                             file_extension='png', reset=False):
    fig = plt.figure(figsize=figsize)
    case = gdir.case
    ref_surf = salem.GeoTiff(gdir.get_filepath('ref_dem')).get_vardata()
    inv_settings = load_pickle(get_subdir_filepath(gdir, subdir,
                               'inversion_settings'))
    noise = 0.
    if os.path.exists(get_subdir_filepath(gdir, subdir, 'dem_noise')):
        noise = np.load(get_subdir_filepath(gdir, subdir,'dem_noise'))
    noisy_ref_surf = ref_surf + noise
    ref_ice_mask = np.load(gdir.get_filepath('ref_ice_mask'))
    ref_inner_mask = compute_inner_mask(ref_ice_mask, full_array=True)

    dl = load_pickle(get_subdir_filepath(gdir, subdir, 'data_logger'))

    reg_parameters = inv_settings['reg_parameters']
    interesting_costs = []
    cost_names = []
    for l, lamb in enumerate(reg_parameters):
        if lamb != 0:
            interesting_costs.append(l)
            cost_names.append('J{:d}'.format(l))

    interesting_costs.append(-1)
    cost_names.append('surf_misfit')
    # make sure all directories exist:
    plot_dir = os.path.join(gdir.dir, subdir, 'plot')
    if reset:
        if os.path.exists(plot_dir):
            shutil.rmtree(plot_dir)
    #if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    os.makedirs(os.path.join(plot_dir, 'bed_error'))
    os.makedirs(os.path.join(plot_dir, 'surf_error'))
    os.makedirs(os.path.join(plot_dir, 'summed_cost'))
    os.makedirs(os.path.join(plot_dir, 'gradient'))
    for c_name in cost_names:
        os.makedirs(os.path.join(plot_dir, c_name))
    dl.plot_rmses(plot_dir)
    dl.plot_c_terms(plot_dir)

    for i in dl.step_indices:
        plot_iterative_step(dl, i, interesting_costs, cost_names, plot_dir,
                            case, ref_ice_mask, ref_inner_mask,
                            noisy_ref_surf, reg_parameters, file_extension,
                            existing_fig=fig)

    plt.close(fig)


def plot_iterative_step(dl, i, interesting_costs, cost_names, plot_dir, case,
                        ref_ice_mask, ref_inner_mask,
                        noisy_ref_surf, reg_parameters,
                        file_extension='png', existing_fig=None):
    base_plotpath = 'iteration{:03d}_{:s}.{:s}'.format(i, '{:s}',
                                                       file_extension)
    model_surf = dl.surfs[i]
    guessed_bed = dl.beds[i]
    surf_diff = model_surf - dl.ref_surf
    bed_diff = guessed_bed - dl.true_bed
    model_ice_mask = (model_surf - guessed_bed) > 0
    model_inner_mask = compute_inner_mask(model_ice_mask, full_array=True)
    costs_arr = get_costs_arr(reg_parameters, noisy_ref_surf, ref_ice_mask,
                              ref_inner_mask, guessed_bed, model_surf,
                              model_ice_mask, model_inner_mask, case.dx)

    cmap_bed_diff = plt.get_cmap('seismic')
    cmap_surf_diff = plt.get_cmap('PuOr_r')
    cmap_gradient = plt.get_cmap('BrBG')
    cmap_list = [sns.diverging_palette(240, 0, l=40, s=99, as_cmap=True),
                 sns.diverging_palette(240, 42.5, l=40, s=99, as_cmap=True),
                 sns.diverging_palette(240, 85, l=40, s=99, as_cmap=True),
                 sns.diverging_palette(240, 127.5, l=40, s=99, as_cmap=True),
                 sns.diverging_palette(240, 170, l=40, s=99, as_cmap=True),
                 sns.diverging_palette(240, 212.5, l=40, s=99, as_cmap=True),
                 sns.diverging_palette(240, 255, l=40, s=99, as_cmap=True)]

    cbar_min = bed_diff.min()
    cbar_max = bed_diff.max()
    cbar_min_max = max(abs(cbar_min), abs(cbar_max))
    norm = MidpointNormalize(midpoint=0., vmin=-cbar_min_max,
                             vmax=cbar_min_max)
    plotpath = os.path.join(plot_dir, 'bed_error',
                            base_plotpath.format('bed_error'))
    plot_bed_difference(bed_diff, plotpath, case,
                        ice_mask=ref_ice_mask,
                        bed_measurements=None,
                        cbar_min=cbar_min, cbar_max=cbar_max,
                        show_cbar=True, norm=norm, cmap=cmap_bed_diff,
                        existing_fig=existing_fig)
    existing_fig.clear()

    cbar_min = surf_diff.min()
    cbar_max = surf_diff.max()
    cbar_min_max = max(abs(cbar_min), abs(cbar_max))
    norm = MidpointNormalize(midpoint=0., vmin=-cbar_min_max,
                             vmax=cbar_min_max)
    plotpath = os.path.join(plot_dir, 'surf_error',
                            base_plotpath.format('surf_error'))
    plot_surf_difference(surf_diff, plotpath, case,
                         ice_mask=ref_ice_mask,
                         bed_measurements=None,
                         cbar_min=cbar_min, cbar_max=cbar_max,
                         show_cbar=True, norm=norm, cmap=cmap_surf_diff,
                         existing_fig=existing_fig)
    existing_fig.clear()

    cbar_min = dl.grads[i].min()
    cbar_max = dl.grads[i].max()
    cbar_min_max = max(abs(cbar_min), abs(cbar_max))
    norm = MidpointNormalize(midpoint=0., vmin=-cbar_min_max,
                             vmax=cbar_min_max)
    plotpath = os.path.join(plot_dir, 'gradient',
                            base_plotpath.format('gradient'))
    plot_differences(dl.grads[i].reshape(surf_diff.shape), plotpath, case,
                     ice_mask=ref_ice_mask,
                     bed_measurements=None, cbar_min=cbar_min,
                     cbar_max=cbar_max, show_cbar=True, norm=norm,
                     cmap=cmap_gradient,
                     cbar_label='gradient of cost function (m$^{-1}$)',
                     existing_fig=existing_fig)
    existing_fig.clear()

    for j in interesting_costs:
        cbar_min = 0
        cbar_max = costs_arr[j].max()
        cbar_min_max = max(abs(cbar_min), abs(cbar_max))
        norm = MidpointNormalize(midpoint=0., vmin=-cbar_min_max,
                                 vmax=cbar_min_max)
        plotpath = os.path.join(plot_dir, cost_names[j],
                                base_plotpath.format(cost_names[j]))
        plot_differences(costs_arr[j], plotpath, case, ice_mask=ref_ice_mask,
                         bed_measurements=None, cbar_min=cbar_min,
                         cbar_max=cbar_max, show_cbar=True, norm=norm,
                         cmap=cmap_list[j],
                         cbar_label='{:s} (m$^2$)'.format(cost_names[j]),
                         existing_fig=existing_fig)
        existing_fig.clear()

    summed_costs = np.sum(costs_arr, axis=0)
    cbar_min = 0
    cbar_max = summed_costs.max()
    cbar_min_max = max(abs(cbar_min), abs(cbar_max))
    #norm = MidpointNormalize(midpoint=0., vmin=-cbar_min_max,
    #                         vmax=cbar_min_max)
    plotpath = os.path.join(plot_dir, 'summed_cost',
                            base_plotpath.format('summed_cost'))
    plot_differences(summed_costs, plotpath, case, ice_mask=ref_ice_mask,
                     bed_measurements=None, cbar_min=cbar_min,
                     cbar_max=cbar_max, show_cbar=True, norm=None,
                     cmap='gist_heat_r',
                     cbar_label='summed cost (m$^2$)')
    existing_fig.clear()
    # TODO: bed_measurements, ...


def plot_differences_discrete_cmap(difference, filepath, case, cbar_min,
                                   cbar_max, title=None, ice_mask=None,
                                   bed_measurements=None, show_cbar=True,
                                   norm=None, cmap='bwr', figsize=(4.5, 3),
                                   cbar_label=None, existing_fig=None, n=21):
    if type(cmap) is str:
        cmap = plt.get_cmap(cmap)
    cmap = ListedColormap(cmap(np.linspace(0, 1, n - 1, endpoint=True)))
    cbar_min_max = max(abs(cbar_min), abs(cbar_max))
    bounds = np.linspace(-cbar_min_max, cbar_min_max, n)
    bounds_step = bounds[1] - bounds[0]
    bounds = bounds[
        np.logical_and(bounds + bounds_step >= cbar_min,
                       bounds - bounds_step <= cbar_max)]

    fig = existing_fig
    if existing_fig is None:
        fig = plt.figure(figsize=figsize)

    ax = fig.add_axes(get_axes_coords(case))
    im_b = imshow_ic(ax, difference, case, cmap=cmap, ticks=False,
                     norm=norm,
                     vmin=cbar_min, vmax=cbar_max)
    cbar = add_colorbar(fig, ax, im_b,
                        norm=norm,
                        boundaries=bounds,
                        extend='neither')
    cbar.set_label(cbar_label)
    cbar.set_clim(-cbar_min_max, cbar_min_max)
    if not show_cbar:
        cbar.remove()
    if title is not None:
        ax.set_title(title)
    if ice_mask is not None:
        plot_glacier_contours(ax, ice_mask, case)
    if bed_measurements is not None:
        plot_glacier_contours(ax, ~bed_measurements.mask, case, colors='k',
                              linestyles='solid', linewidths=[2.])
    plt.savefig(filepath)
    if existing_fig is None:
        plt.close(fig)