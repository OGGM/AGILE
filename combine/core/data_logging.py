import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import holoviews as hv
from holoviews import opts
import pickle
import gzip
from combine.core.arithmetics import RMSE, mean_BIAS
from oggm import cfg
import os


# from combine.utils.optimization import LCurveTest

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

    def __init__(self, true_bed, exact_surf, ref_surf, first_guessed_bed,
                 true_shape=None, first_guessed_shape=None):
        self.reset(true_bed, exact_surf, ref_surf, first_guessed_bed,
                   true_shape, first_guessed_shape)

    # def __init__(self, inv_dir):
    #     self.reset(inv_dir.true_bed,
    #                inv_dir.ref_surf,
    #                inv_dir.ref_surf,
    #                inv_dir.first_guessed_bed)
    #     self.case = inv_dir.inv_settings['case']
    #     self.solver = inv_dir.inv_settings['solver']
    #     self.minimize_options = inv_dir.inv_settings['minimize_options']

    def reset(self, true_bed, exact_surf, ref_surf, first_guessed_bed,
              true_shape, first_guessed_shape):
        self.grads = []
        self.costs = []
        self.c_terms = []
        self.surfs = []
        self.beds = []
        self.step_indices = []
        self.widths = []
        self.shapes = []
        self.fct_calls = [0]
        self.lambdas = np.zeros(6)
        self.true_bed = true_bed
        self.exact_surf = exact_surf
        self.ref_surf = ref_surf
        self.first_guessed_bed = first_guessed_bed
        self.true_shape = true_shape
        self.first_guessed_shape = first_guessed_shape
        self.solver = None
        self.minimize_options = None
        self.case = None

    def callback_fct(self, x0):
        i = len(self.costs) - 1
        if i >= 0:
            self.step_indices.append(i)
            b = self.true_bed
            shape = self.true_shape
            log_entry = '''
            ----------------------------------------------
            Function Call: {func_call:d}
            Iteration: {iteration:d}
            Cost: {cost:g}
            Bed RMSE: {bed_rmse:g}
            Bed Bias: {bed_bias:g}
            Bed Max_diff: {bed_maxdiff:g}
            Surface RMSE: {surf_rmse:g}
            Surface Max_diff: {surf_maxdiff:g}
            '''
            myargs = {
                'func_call': i,
                'iteration': len(self.step_indices),
                'cost': self.costs[i],
                'bed_rmse': RMSE(self.beds[i], b),
                'bed_bias': mean_BIAS(self.beds[i], b),
                'bed_maxdiff': np.max(np.abs(self.beds[i] - b)),
                'surf_rmse': RMSE(self.surfs[i], self.ref_surf),
                'surf_maxdiff': np.max(np.abs(self.surfs[i] - self.ref_surf))
            }

            if len(self.shapes) > 0:
                log_entry += '''Shape RMSE: {shape_rmse:g}
            Shape Bias: {shape_bias:g}
            Shape Max_diff: {shape_maxdiff:g}
            '''
                myargs['shape_rmse'] = RMSE(self.shapes[i], shape)
                myargs['shape_bias'] = mean_BIAS(self.shapes[i], shape)
                myargs['shape_maxdiff'] = np.max(np.abs(self.shapes[i] -
                                                        shape))

            log_entry = log_entry.format(**myargs)
            print(log_entry)


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


class DataLogger_bed_h_and_shape(object):

    def __init__(self, true_bed, exact_surf, ref_surf, first_guessed_bed,
                 true_shape=None, first_guessed_shape=None, opti_var='bed_h'):
        self.reset(true_bed, exact_surf, ref_surf, first_guessed_bed,
                   true_shape, first_guessed_shape, opti_var)

    # def __init__(self, inv_dir):
    #     self.reset(inv_dir.true_bed,
    #                inv_dir.ref_surf,
    #                inv_dir.ref_surf,
    #                inv_dir.first_guessed_bed)
    #     self.case = inv_dir.inv_settings['case']
    #     self.solver = inv_dir.inv_settings['solver']
    #     self.minimize_options = inv_dir.inv_settings['minimize_options']

    def reset(self, true_bed, exact_surf, ref_surf, first_guessed_bed,
              true_shape, first_guessed_shape, opti_var):
        # variables for main loop
        self.main_iterations = []

        # variable for function calls
        self.fct_calls = [0]

        # help variables to switch where grad and costs should be saved
        self.in_bed_h_opti = False
        self.in_shape_opti = False
        self.in_bed_h_and_shape_opti = False

        # variables for plot
        self.geometry = {}
        self.measurements = {}

        # variables for bed_h optimisation
        self.bed_h_grads = []
        self.bed_h_costs = []
        self.bed_h_c_terms = []
        self.bed_h_surfs = []
        self.bed_h_widths = []
        self.bed_h_thicks = []
        self.beds = []
        self.step_indices_bed_h = []
        self.current_step_indices_bed_h = []

        # variables for shape optimisation
        self.shape_grads = []
        self.shape_grads_smoothed = []
        self.shape_costs = []
        self.shape_c_terms = []
        self.shape_surfs = []
        self.shape_widths = []
        self.shape_thicks = []
        self.shapes = []
        self.step_indices_shape = []
        self.current_step_indices_shape = []

        # variables for bed_h and shape optimisation
        self.grads = []
        self.costs = []
        self.c_terms = []
        self.surfs = []
        self.beds = []
        self.step_indices = []
        self.widths = []
        self.shapes = []
        self.thicks = []
        self.step_indices = []

        # some other things to calculate some statistics
        self.lambdas = np.zeros(6)
        self.true_bed = true_bed
        self.exact_surf = exact_surf
        self.ref_surf = ref_surf
        self.first_guessed_bed = first_guessed_bed
        self.true_shape = true_shape
        self.first_guessed_shape = first_guessed_shape
        self.opti_var = opti_var
        self.solver = None
        self.minimize_options = None
        self.case = None

    def callback_fct(self, x0):
        if self.in_bed_h_opti:
            i = len(self.bed_h_costs) - 1
            if i >= 0:
                self.step_indices_bed_h.append(i)
                self.current_step_indices_bed_h.append(i)
                b = self.true_bed
                text = '''
    Iteration bed height optimisation: {iteration:d}
    Total Function calls: {fct_call:d}
    Cost: {cost:g}
    Bed RMSE: {bed_rmse:g}
    Bed Bias: {bed_bias:g}
    Bed Max_diff: {bed_max_diff:g}'''

                args = {'iteration': len(self.current_step_indices_bed_h),
                        'fct_call': self.fct_calls[-1],
                        'cost': self.bed_h_costs[i],
                        'bed_rmse': RMSE(self.beds[i], b),
                        'bed_bias': mean_BIAS(self.beds[i], b),
                        'bed_max_diff': np.max(np.abs(self.beds[i] - b))}

                print(text.format(**args))
            self.in_bed_h_opti = False

        elif self.in_shape_opti:
            i = len(self.shape_costs) - 1
            if i >= 0:
                self.step_indices_shape.append(i)
                self.current_step_indices_shape.append(i)
                shape = self.true_shape
                text = '''
    Iteration bed shape optimisation: {iteration:d}
    Total Function calls: {fct_call:d}
    Cost: {cost:g}
    Shape RMSE: {shape_rmse:g}
    Shape Bias: {shape_bias:g}
    Shape Max_diff: {shape_max_diff:g}'''

                args = {'iteration': len(self.current_step_indices_shape),
                        'fct_call': self.fct_calls[-1],
                        'cost': self.shape_costs[i],
                        'shape_rmse': RMSE(self.shapes[i], shape),
                        'shape_bias': mean_BIAS(self.shapes[i], shape),
                        'shape_max_diff': np.max(np.abs(self.shapes[i] -
                                                        shape))}

                print(text.format(**args))
            self.in_bed_h_opti = False
        elif self.in_bed_h_and_shape_opti:
            i = len(self.costs) - 1
            if i >= 0:
                self.step_indices.append(i)
                b = self.true_bed
                shape = self.true_shape
                log_entry = '''
                ----------------------------------------------
                Function Call: {func_call:d}
                Iteration: {iteration:d}
                Cost: {cost:g}
                Bed RMSE: {bed_rmse:g}
                Bed Bias: {bed_bias:g}
                Bed Max_diff: {bed_maxdiff:g}
                Shape RMSE: {shape_rmse:g}
                Shape Bias: {shape_bias:g}
                Shape Max_diff: {shape_maxdiff:g}
                Surface RMSE: {surf_rmse:g}
                Surface Max_diff: {surf_maxdiff:g}
                '''
                myargs = {
                    'func_call': i,
                    'iteration': len(self.step_indices),
                    'cost': self.costs[i],
                    'bed_rmse': RMSE(self.beds[i], b),
                    'bed_bias': mean_BIAS(self.beds[i], b),
                    'bed_maxdiff': np.max(np.abs(self.beds[i] - b)),
                    'shape_rmse': RMSE(self.shapes[i], shape),
                    'shape_bias': mean_BIAS(self.shapes[i], shape),
                    'shape_maxdiff': np.max(np.abs(self.shapes[i] - shape)),
                    'surf_rmse': RMSE(self.surfs[i], self.ref_surf),
                    'surf_maxdiff': np.max(np.abs(self.surfs[i] -
                                                  self.ref_surf))
                }

                log_entry = log_entry.format(**myargs)
                print(log_entry)
            self.in_bed_h_and_shape_opti = False

    def main_iteration_callback(self):
        text = '''
--------------------------------------------------
Main Iteration number {iteration:d}:'''
        arg = {'iteration': self.main_iterations[-1]}
        print(text.format(**arg))

    def filter_data_from_optimization(self, opti_var):
        # Filter all "exploratory" model runs to only get "real" iteration
        # steps
        my_grads = []
        my_costs = []
        my_c_terms = []
        my_beds = []
        my_surfs = []
        my_widths = []
        my_shapes = []
        my_thicks = []

        if opti_var == 'bed_h':
            for i in self.step_indices_bed_h:
                my_grads.append(self.bed_h_grads[i])
                my_costs.append(self.bed_h_costs[i])
                my_beds.append(self.beds[i])
                my_surfs.append(self.bed_h_surfs[i])
                my_c_terms.append(self.bed_h_c_terms[i])
                my_widths.append(self.bed_h_widths[i])
                my_thicks.append(self.bed_h_thicks[i])

            self.bed_h_grads = my_grads
            self.bed_h_costs = my_costs
            self.beds = my_beds
            self.bed_h_surfs = my_surfs
            self.bed_h_c_terms = my_c_terms
            self.bed_h_widths = my_widths
            self.bed_h_thicks = my_thicks
            self.step_indices_bed_h = np.arange(len(self.step_indices_bed_h))

        elif opti_var == 'shape':
            for i in self.step_indices_shape:
                my_grads.append(self.shape_grads[i])
                my_costs.append(self.shape_costs[i])
                my_shapes.append(self.shapes[i])
                my_surfs.append(self.shape_surfs[i])
                my_c_terms.append(self.shape_c_terms[i])
                my_widths.append(self.shape_widths[i])
                my_thicks.append(self.shape_thicks[i])

            self.shape_grads = my_grads
            self.shape_costs = my_costs
            self.shapes = my_shapes
            self.shape_surfs = my_surfs
            self.shape_c_terms = my_c_terms
            self.shape_widths = my_widths
            self.shape_thicks = my_thicks
            self.step_indices_shape = np.arange(len(self.step_indices_shape))

        elif opti_var == 'bed_h and shape':
            for i in self.step_indices:
                my_grads.append(self.grads[i])
                my_costs.append(self.costs[i])
                my_beds.append(self.beds[i])
                my_surfs.append(self.surfs[i])
                my_c_terms.append(self.c_terms[i])
                my_widths.append(self.widths[i])
                my_shapes.append(self.shapes[i])
                my_thicks.append(self.thicks[i])
            self.grads = my_grads
            self.costs = my_costs
            self.beds = my_beds
            self.surfs = my_surfs
            self.c_terms = my_c_terms
            self.widths = my_widths
            self.shapes = my_shapes
            self.thicks = my_thicks
            self.step_indices = np.arange(len(self.step_indices))

    def get_bed_differences(self):
        return np.array(self.beds - self.true_bed)

    def get_bed_rmses(self):
        return np.sqrt(np.mean(self.get_bed_differences() ** 2, axis=(1, 2)))

    def get_surf_differences(self):
        return np.array(self.surfs - self.ref_surf)

    def get_surf_rmses(self):
        return np.sqrt(np.mean(self.get_surf_differences() ** 2, axis=(1, 2)))

    def plot_iterations(self,
                        opti_var='bed_h',
                        plot_width=600,
                        plot_height=400):
        if opti_var == 'bed_h':
            x = self.geometry['distance_along_glacier']
            estimated_beds = np.zeros(
                (len(self.beds), len(x)))
            estimated_beds[:, self.measurements['ice_mask']] = self.beds
            estimated_beds[:, ~self.measurements['ice_mask']] =\
                self.measurements['bed_known']
            dictionary = {int(i): hv.Curve(
                (x,
                 y - self.measurements['bed_all']),
                'distance',
                'diff bed_h',
                label='estimated')
                          for i, y in enumerate(estimated_beds)}

            first_guess = np.zeros(len(x))
            first_guess[self.measurements['ice_mask']] = self.first_guessed_bed
            first_guess[~self.measurements['ice_mask']] =\
                self.measurements['bed_known']

            first_guess_bed = hv.Curve(
                (x,
                 first_guess -
                 self.measurements['bed_all']),
                'distance',
                'diff bed_h',
                label='first guess')

            zero_line_bed = hv.Curve(
                (x,
                 np.zeros(len(x))),
                'distance',
                'diff bed_h',
                ).opts(line_color='black')

            bed_h_plot = (zero_line_bed *
                          first_guess_bed *
                          hv.HoloMap(dictionary,
                                     kdims='Iteration')
                          ).opts(width=plot_width,
                                 height=plot_height)

            return bed_h_plot.opts(opts.Curve(line_width=3))

        elif opti_var == 'shape':
            x = self.geometry['distance_along_glacier']

            estimated_shapes = np.zeros(
                (len(self.shapes), len(x)))
            estimated_shapes[:, self.measurements['ice_mask']] = self.shapes
            estimated_shapes[:, ~self.measurements['ice_mask']] =\
                self.measurements['shape_known']
            dictionary = {int(i): hv.Curve(
                (x,
                 y - self.measurements['shape_all']),
                'distance',
                'diff shape',
                label='estimated')
                          for i, y in enumerate(estimated_shapes)}

            first_guess = np.zeros(len(x))
            first_guess[self.measurements['ice_mask']] =\
                self.first_guessed_shape
            first_guess[~self.measurements['ice_mask']] =\
                self.measurements['shape_known']

            first_guess_shape = hv.Curve(
                (x,
                 first_guess -
                 self.measurements['shape_all']),
                'distance',
                'diff shape',
                label='first guess')
            zero_line_shape = hv.Curve(
                (x,
                 np.zeros(len(x))),
                'distance',
                'diff shape',
                ).opts(line_color='black')
            shape_plot = (zero_line_shape *
                          first_guess_shape *
                          hv.HoloMap(dictionary,
                                     kdims='Iteration')
                          ).opts(width=plot_width,
                                 height=plot_height)
            return shape_plot.opts(opts.Curve(line_width=3))

        elif opti_var == 'bed_h and shape':
            self.filter_data_from_optimization(opti_var)
            x = self.geometry['distance_along_glacier']

            first_guess_b = np.zeros(len(x))
            first_guess_b[self.measurements['ice_mask']] =\
                self.first_guessed_bed
            first_guess_b[~self.measurements['ice_mask']] =\
                self.measurements['bed_known']
            first_guess_bed = hv.Curve(
                (x,
                 first_guess_b -
                 self.measurements['bed_all']),
                'distance',
                'diff bed_h',
                label='first guess')

            zero_line_bed = hv.Curve(
                (x,
                 np.zeros(len(x))),
                'distance',
                'diff bed_h',
                ).opts(line_color='black')

            first_guess_s = np.zeros(len(x))
            first_guess_s[self.measurements['ice_mask']] =\
                self.first_guessed_shape
            first_guess_s[~self.measurements['ice_mask']] =\
                self.measurements['shape_known']

            first_guess_shape = hv.Curve(
                (x,
                 first_guess_s -
                 self.measurements['shape_all']),
                'distance',
                'diff shape',
                label='first guess')

            zero_line_shape = hv.Curve(
                (x,
                 np.zeros(len(x))),
                'distance',
                'diff shape',
                ).opts(line_color='black')

            estimated_beds = np.zeros(
                (len(self.beds), len(x)))
            estimated_beds[:, self.measurements['ice_mask']] = self.beds
            estimated_beds[:, ~self.measurements['ice_mask']] =\
                self.measurements['bed_known']

            estimated_shapes = np.zeros(
                (len(self.shapes), len(x)))
            estimated_shapes[:, self.measurements['ice_mask']] = self.shapes
            estimated_shapes[:, ~self.measurements['ice_mask']] =\
                self.measurements['shape_known']

            dictionary = {int(i): ((zero_line_bed * first_guess_bed *
                                    hv.Curve(
                                        (x,
                                         bed -
                                         self.measurements['bed_all']),
                                        'distance',
                                        'diff bed_h',
                                        label='estimated')
                                    ).opts(xaxis='top',
                                           width=plot_width,
                                           height=int(plot_height / 2))
                                   + (zero_line_shape * first_guess_shape *
                                      hv.Curve(
                                        (x,
                                         shape -
                                         self.measurements['shape_all']),
                                        'distance',
                                        'diff shape',
                                        label='estimated')
                                      ).opts(width=plot_width,
                                             height=int(plot_height / 2))
                                   ).cols(1).opts(opts.Curve(line_width=3))
                          for i, (bed, shape) in enumerate(zip(estimated_beds,
                                                               estimated_shapes
                                                               ))}

            return hv.HoloMap(dictionary, kdims='Iteration').collate()

    def plot_glacier_bed(self):
        glacier_bed = hv.Curve((self.geometry['distance_along_glacier'],
                                self.geometry['bed_h']),
                               'distance',
                               'height',
                               label='glacier bed'
                               ).opts(line_color='black',
                                      width=600,
                                      height=400)
        
        glacier_sfc = hv.Curve((self.geometry['distance_along_glacier'],
                                self.measurements['sfc_h']),
                               'distance',
                               'height',
                               label='glacier surface'
                               ).opts(line_color='blue',
                                      width=600,
                                      height=400)

        return (glacier_bed * glacier_sfc).opts(opts.Curve(line_width=3))

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
