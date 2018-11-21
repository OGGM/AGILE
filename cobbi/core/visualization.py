import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
fontprops = fm.FontProperties(size=18)

#from cobbi.utils.optimization import LCurveTest

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
        #value[~np.isnan(value)] = np.nan # Allow masks
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

def add_colorbar(fig, ax, mappable, norm=None, extend='neither'):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, ax=ax, cax=cax, extend=extend)
    cbar.outline.set_visible(False)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    #cbar.outline.set_linewidth(0.75)
    return cbar

def get_axes_coords(case):
    if case.name == 'Giluwe':
        return [0., 0.05, 0.9, 0.9]  # left, bottom, width, height
    elif case.name == 'Borden Peninsula':
        return [0., 0.015, 0.85, 0.97]  # left, bottom, width,
        # height


def plot_bed_difference(bed_difference, filepath, case, cbar_min,
                        cbar_max, title=None,
                        ice_mask=None, bed_measurements=None,
                        show_cbar=True, norm=None, cmap='bwr',
                        figsize=(4.5, 3), cbar_label='bed elevation error (m)'):
    plot_differences(bed_difference, filepath, case, cbar_min, cbar_max,
                     title, ice_mask, bed_measurements, show_cbar, norm,
                     cmap, figsize, cbar_label=cbar_label)

def plot_differences(difference, filepath, case, cbar_min, cbar_max,
                     title=None, ice_mask=None, bed_measurements=None,
                     show_cbar=True, norm=None, cmap='bwr', figsize=(4.5, 3),
                     cbar_label=None):
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
    plt.close(fig)


def plot_surf_difference(surf_difference, filepath, case, cbar_min,
                         cbar_max, title=None,
                         ice_mask=None, bed_measurements=None,
                         show_cbar=True, norm=None, cmap='PuOr_r',
                         figsize=(4.5, 3),
                         cbar_label='surf elevation error (m)'):

    plot_differences(surf_difference, filepath, case, cbar_min, cbar_max,
                     title, ice_mask, bed_measurements, show_cbar, norm,
                     cmap, figsize, cbar_label=cbar_label)