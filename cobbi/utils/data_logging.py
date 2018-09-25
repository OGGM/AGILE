import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
import gzip
from oggm import cfg


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


class DataLogger(object):

    def __init__(self, ref_bed, ref_surf, first_guess_bed):
        self.reset(ref_bed, ref_surf, first_guess_bed)

    def reset(self, ref_bed, ref_surf, first_guess_bed):
        self.grads = []
        self.costs = []
        self.c_terms = []
        self.surfs = []
        self.beds = []
        self.step_indices = []
        self.lambdas = np.zeros(9)
        self.ref_bed = ref_bed
        self.ref_surf = ref_surf
        self.first_guess_bed = first_guess_bed

    def filter_data_from_optimization(self):
        # Filter all "exploratory" model runs to only get "real" iteration steps
        my_grads = []
        my_costs = []
        my_c_terms = []
        my_beds = []
        my_surfs = []
        for i in self.step_indices:
            my_grads.append(self.grads[i])
            my_costs.append(self.costs[i])
            my_beds.append(self.beds[i])
            my_surfs.append(self.surfs[i])
            my_c_terms.append(self.c_terms[i])
        self.grads = my_grads
        self.costs = my_costs
        self.beds = my_beds
        self.surfs = my_surfs
        self.c_terms = my_c_terms
        self.step_indices = np.arange(len(self.step_indices))

    def get_bed_differences(self):
        return np.array(self.beds - self.ref_bed)

    def get_bed_rmses(self):
        return np.sqrt(np.mean(self.get_bed_differences() ** 2, axis=(1, 2)))

    def get_surf_differences(self):
        return np.array(self.surfs - self.ref_surf)

    def get_surf_rmses(self):
        return np.sqrt(np.mean(self.get_surf_differences() ** 2, axis=(1, 2)))

    def plot_all(self, dir):
        plt.ioff()
        self.plot_costs(dir)
        self.plot_c_terms(dir)
        self.plot_rmses(dir)
        self.plot_bed_differences(dir)
        self.plot_surf_differences(dir)
        self.plot_grads(dir, self.ref_bed.shape)
        plt.ion()

    def plot_costs(self, basedir):
        plt.figure()
        plt.semilogy(self.costs)
        plt.xlabel('function call #')
        plt.ylabel('Cost')
        plt.savefig(basedir + 'cost.pdf')
        plt.clf()

    def plot_c_terms(self, basedir):
        plt.figure()
        data = np.array(self.c_terms)
        for i in range(self.lambdas.size):
            plt.semilogy(data[:, i], label='Reg {:d}'.format(i))
        plt.semilogy(data[:, -1], label='Bare cost')
        plt.xlabel('function call #')
        plt.ylabel('Cost')
        plt.legend()
        plt.savefig(basedir + 'c_terms.pdf')
        plt.clf()

    def plot_image(self, data, max_val, title, filename):
        plt.figure()
        im = plt.imshow(data, norm=MidpointNormalize(midpoint=0., vmin=-max_val,
                                                     vmax=max_val), cmap='RdBu')
        plt.title(title)
        plt.colorbar(im)
        plt.savefig(filename)
        plt.clf()

    def plot_rmses(self, basedir):
        plt.figure()
        plt.semilogy(self.get_bed_rmses(), label='Bed')
        plt.semilogy(self.get_surf_rmses(), label='Surface')
        plt.xlabel('function call #')
        plt.ylabel('RMSE')
        plt.legend()
        plt.savefig(basedir + 'bed_surf_rmse.pdf')
        plt.clf()

    def plot_bed_differences(self, basedir, cbar_range=50):
        bed_differences = self.get_bed_differences()
        for i in range(bed_differences.shape[0]):
            self.plot_image(bed_differences[i, :, :], cbar_range,
                            'Bed difference #{:d}, |max_diff|={:g}'.format(
                                i, np.max(np.abs(bed_differences[i]))),
                                basedir + 'bed_diff{:d}.png'.format(i))

    def plot_surf_differences(self, basedir, cbar_range=50):
        surf_differences = self.get_surf_differences()
        for i in range(surf_differences.shape[0]):
            self.plot_image(surf_differences[i, :, :], cbar_range,
                            'Surface difference #{:d}, |max_diff|={:g}'.format(
                                i, np.max(np.abs(surf_differences[i]))),
                                basedir + 'surf_diff{:d}.png'.format(i))

    def plot_grads(self, basedir, ref_shape, cbar_range=None):
        cbar_min_max = cbar_range
        for i in range(len(self.grads)):
            if cbar_range is None:
                cbar_min_max = np.max(np.abs(self.grads[i]))
            self.plot_image(self.grads[i].reshape(ref_shape), cbar_min_max,
                            'Gradient #{:d}'.format(i),
                            basedir + 'grad{:d}.png'.format(i))


def load_pickle(filepath, use_compression=None):
    """Reads a pickle located in the directory.

    Parameters
    ----------
    filepath : str
        filepath
    use_compression : bool
        whether or not the file ws compressed. Default is to use
        cfg.PARAMS['use_compression'] for this (recommended)

    Returns
    -------
    An object read from the pickle
    """
    use_comp = (use_compression if use_compression is not None
                else cfg.PARAMS['use_compression'])
    _open = gzip.open if use_comp else open
    with _open(filepath, 'rb') as f:
        out = pickle.load(f)
    return out

def write_pickle(var, filepath, use_compression=None):
    """ Writes a variable to a pickle on disk.

    Parameters
    ----------
    var : object
        the variable to write to disk
    filepath : str
        filepath
    use_compression : bool
        whether or not the file ws compressed. Default is to use
        cfg.PARAMS['use_compression'] for this (recommended)
    """
    use_comp = (use_compression if use_compression is not None
                else cfg.PARAMS['use_compression'])
    _open = gzip.open if use_comp else open
    with _open(filepath, 'wb') as f:
        pickle.dump(var, f, protocol=-1)
