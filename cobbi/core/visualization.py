import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
#from cobbi.utils.optimization import LCurveTest

class MidpointNormalize(colors.Normalize):
    # see: https://matplotlib.org/users/colormapnorms.html#custom-normalization-two-linear-ranges
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


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