import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
import gzip
from oggm import cfg
import os


# from agile.utils.optimization import LCurveTest

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

    def __init__(self, true_bed, exact_surf, ref_surf, first_guessed_bed):
        self.reset(true_bed, exact_surf, ref_surf, first_guessed_bed)

    def __init__(self, inv_dir):
        self.reset(inv_dir.true_bed,
                   inv_dir.ref_surf,
                   inv_dir.ref_surf,
                   inv_dir.first_guessed_bed)
        self.case = inv_dir.inv_settings['case']
        self.solver = inv_dir.inv_settings['solver']
        self.minimize_options = inv_dir.inv_settings['minimize_options']

    def reset(self, true_bed, exact_surf, ref_surf, first_guessed_bed):
        self.grads = []
        self.costs = []
        self.c_terms = []
        self.surfs = []
        self.beds = []
        self.step_indices = []
        self.lambdas = np.zeros(6)
        self.true_bed = true_bed
        self.exact_surf = exact_surf
        self.ref_surf = ref_surf
        self.first_guessed_bed = first_guessed_bed
        self.solver = None
        self.minimize_options = None
        self.case = None

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
        return np.array(self.beds - self.true_bed)

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
        self.plot_grads(dir, self.true_bed.shape)
        plt.ion()

    def plot_costs(self, basedir):
        fig = plt.figure()
        plt.semilogy(self.costs)
        plt.xlabel('Iteration #')
        plt.ylabel('Cost')
        plt.savefig(os.path.join(basedir, 'cost.pdf'))
        plt.close(fig)

    def plot_c_terms(self, basedir):
        fig = plt.figure()
        data = np.array(self.c_terms)
        for i in range(self.lambdas.size):
            plt.semilogy(data[:, i], label='Reg {:d}'.format(i))
        plt.semilogy(data[:, -1], label='Bare cost', color='k')
        plt.xlabel('Iteration #')
        plt.ylabel('Cost')
        plt.legend()
        plt.savefig(os.path.join(basedir, 'c_terms.pdf'))
        plt.close(fig)

    def plot_image(self, data, max_val, title, filename):
        fig = plt.figure()
        im = plt.imshow(data, norm=MidpointNormalize(midpoint=0., vmin=-max_val,
                                                     vmax=max_val), cmap='RdBu')
        plt.title(title)
        plt.colorbar(im)
        plt.savefig(filename)
        plt.close(fig)

    def plot_rmses(self, basedir):
        fig = plt.figure()
        plt.semilogy(self.get_bed_rmses()[self.step_indices], label='Bed')
        plt.semilogy(self.get_surf_rmses()[self.step_indices], label='Surface')
        plt.xlabel('Iteration #')
        plt.ylabel('RMSE')
        plt.legend()
        plt.savefig(os.path.join(basedir, 'bed_surf_rmse.pdf'))
        plt.close(fig)

    def plot_bed_differences(self, basedir, cbar_range=200):
        bed_differences = self.get_bed_differences()
        for i in self.step_indices:
            fig = plt.figure()
            im = plt.imshow(bed_differences[i, :, :],
                            norm=MidpointNormalize(midpoint=0.,
                                                   vmin=-cbar_range,
                                                   vmax=cbar_range),
                            cmap='RdBu')
            plt.title('Bed difference #{:d}, max|Δb|={:g} m'.format(
                                i, np.max(np.abs(bed_differences[i]))))
            cbar = plt.colorbar(im)
            cbar.set_label('Δb (m)')
            plt.savefig(os.path.join(basedir, 'bed_diff{:d}.png'.format(i)))
            plt.close(fig)

    def plot_surf_differences(self, basedir, cbar_range=50):
        surf_differences = self.get_surf_differences()
        for i in self.step_indices:
            #self.plot_image(surf_differences[i, :, :], cbar_range,
            #                'Surface difference #{:d}, |max_diff|={:g}'.format(
            #                    i, np.max(np.abs(surf_differences[i]))),
            #                    basedir + 'surf_diff{:d}.png'.format(i))
            fig = plt.figure()
            im = plt.imshow(surf_differences[i, :, :],
                            norm=MidpointNormalize(midpoint=0.,
                                                   vmin=-cbar_range,
                                                   vmax=cbar_range),
                            cmap='RdBu')
            plt.title('Surface difference #{:d}, max|Δs|={:g} m'.format(
                                i, np.max(np.abs(surf_differences[i]))))
            cbar = plt.colorbar(im)
            cbar.set_label('Δs (m)')
            plt.savefig(os.path.join(basedir, 'surf_diff{:d}.png'.format(i)))
            plt.close(fig)

    def plot_grads(self, basedir, ref_shape, cbar_range=None):
        cbar_min_max = cbar_range
        for i in self.step_indices:
            if cbar_range is None:
                cbar_min_max = np.max(np.abs(self.grads[i]))
            #self.plot_image(self.grads[i].reshape(ref_shape), cbar_min_max,
            #                'Gradient #{:d}'.format(i),
            #                basedir + 'grad{:d}.png'.format(i))

            fig = plt.figure()
            im = plt.imshow(self.grads[i].reshape(ref_shape),
                            norm=MidpointNormalize(midpoint=0.,
                                                   vmin=-cbar_min_max,
                                                   vmax=cbar_min_max),
                            cmap='RdBu')
            plt.title('Gradient #{:d}'.format(i))
            cbar = plt.colorbar(im)
            cbar.set_label('Gradient of cost-function (m$^{-1}$)')
            plt.savefig(os.path.join(basedir, 'grad{:d}.png'.format(i)))
            plt.close(fig)


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
