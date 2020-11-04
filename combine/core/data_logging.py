import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import holoviews as hv
from holoviews import opts
import pickle
import gzip
from combine.core.arithmetics import RMSE, mean_BIAS, max_dif
from oggm import cfg
import os


# from combine.utils.optimization import LCurveTest


class DataLogger(object):

    def __init__(self, bed_geometry, opti_parameter, two_parameter_option,
                 main_iterations_separeted, geometry, measurements,
                 first_guess, reg_parameters, used_bed_h_geometry,
                 used_along_glacier_geometry, minimize_options, solver,
                 glacier_state):
        # first save all initial data for idealized experiment
        self.bed_geometry = bed_geometry
        self.opti_parameter = opti_parameter
        self.geometry = geometry
        self.measurements = measurements
        self.reg_parameters = reg_parameters
        self.geometry_bed_h = used_bed_h_geometry
        self.along_glacier_geometry = used_along_glacier_geometry
        self.glacier_state = glacier_state

        # define some variables needed for all bed_geometries
        self.grads = np.empty((0, self.geometry['nx']))
        self.costs = np.empty((0, 1))
        self.c_terms = np.empty((0, len(self.reg_parameters)))
        self.surfs = np.empty((0, self.geometry['nx']))
        self.true_surfs = measurements['sfc_h']
        self.widths = np.empty((0, self.geometry['nx']))
        self.true_widths = measurements['widths']
        self.solver = solver
        self.minimize_options = minimize_options
        self.current_main_iterations = None
        self.computing_time = None
        self.fct_calls = np.array([0])
        # save optimisation variable of current iteration
        self.opti_var_iteration = np.empty((0, 1))
        # save step indices for filtering out steps only needed by the
        # minimisation algorithm
        self.step_indices = np.empty((0, 1))

        # help variable for two optimisation variables
        two_opti_parameter_options = ['separated', 'at once']

        # create variables according to the bed_geometry and opti_parameter
        if bed_geometry == 'rectangular':
            if opti_parameter == 'bed_h':
                self.opti_var_1 = 'bed_h'
                self.opti_var_2 = None

            else:
                raise ValueError('Unknown optimisation parameter for '
                                 'rectangular!')

        elif bed_geometry == 'parabolic':
            if opti_parameter == 'bed_h':
                self.opti_var_1 = 'bed_h'
                self.opti_var_2 = None

            elif opti_parameter == 'bed_shape':
                self.opti_var_1 = 'bed_shape'
                self.opti_var_2 = None

            elif opti_parameter == 'bed_h and bed_shape':
                self.opti_var_1 = 'bed_h'
                self.opti_var_2 = 'bed_shape'

                if two_parameter_option not in two_opti_parameter_options:
                    raise ValueError('Unknown optimisation option for two '
                                     'parameters!')
                self.two_parameter_option = two_parameter_option
                self.main_iterations_separeted = main_iterations_separeted

                # variable to keep track of the main iterations
                self.main_iterations = np.empty((0, 1))
                # variable to keep track in which main iteration the algorithm
                # is at the current minimisation
                self.current_main_iterations = np.empty((0, 1))

            else:
                raise ValueError('Unknown optimisation parameter for '
                                 'parabolic!')

        elif bed_geometry == 'trapezoidol':
            if opti_parameter == 'bed_h':
                self.opti_var_1 = 'bed_h'
                self.opti_var_2 = None

            elif opti_parameter == 'w0':
                self.opti_var_1 = 'w0'
                self.opti_var_2 = None

            elif opti_parameter == 'bed_h and w0':
                self.opti_var_1 = 'bed_h'
                self.opti_var_2 = 'w0'

                if two_parameter_option not in two_opti_parameter_options:
                    raise ValueError('Unknown optimisation option for two '
                                     'parameters!')
                self.two_parameter_option = two_parameter_option
                self.main_iterations_separeted = main_iterations_separeted

                # variable to keep track of the main iterations
                self.main_iterations = np.empty((0, 1))
                # variable to keep track in which main iteration the algorithm
                # is at the current miniisation
                self.current_main_iterations = np.empty((0, 1))

            else:
                raise ValueError('Unknown optimisation parameter for '
                                 'trapezoidol!')

        else:
            raise ValueError('Unknown bed geometry!')

        # save the true and the first guess for optimisation variable 1
        self.true_opti_var_1 = self.geometry[self.opti_var_1]
        self.first_guessed_opti_var_1 = self.first_guess[self.opti_var_1]
        # variable for saving the iteration output
        self.guessed_opti_var_1 = np.empty((0, self.geometry['nx']))

        # check if second optimisation variable is needed
        if self.opti_var_2 is not None:
            # safe the true and the first guess for optimisation variable 2
            self.true_opti_var_2 = self.geometry[self.opti_var_2]
            self.first_guessed_opti_var_2 = self.first_guess[self.opti_var_2]
            # variable for saving the iteration output
            self.guessed_opti_var_2 = np.empty((0, self.geometry['nx']))

        # create info Text for callback_fct
        self.info_text = '''
    Iteration: {iteration:d}
    Optimisation Variable: {opti_var}
    Total Function calls: {fct_call:d}
    Cost: {cost:g}
    {opti_var_1} RMSE: {opti_var_1_rmse:g}
    {opti_var_1} Bias: {opti_var_1_bias:g}
    {opti_var_1} Max_diff: {opti_var_1_max_dif:g}'''
        if self.opti_var_2 is not None:
            self.info_text += '''
    {opti_var_2} RMSE: {opti_var_2_rmse:g}
    {opti_var_2} Bias: {opti_var_2_bias:g}
    {opti_var_2} Max_diff: {opti_var_2_max_dif:g}
    '''

        # define filename to save output at the end
        if bed_geometry == 'rectangular':
            self.filename = 'rec_'
        elif bed_geometry == 'parabolic':
            self.filename = 'par_'
        elif bed_geometry == 'trapezoidol':
            self.filename = 'tra_'
        else:
            raise ValueError('Unknown bed geometry!')

        if self.geometry_bed_h == 'linear':
            self.filename += 'line_'
        elif self.geometry_bed_h == 'cliff':
            self.filename += 'clif_'
        elif self.geometry_bed_h == 'random':
            self.filename += 'rand_'
        else:
            raise ValueError('Unknown bed height geometry!')

        if self.along_glacier_geometry == 'constant':
            self.filename += 'cons_'
        elif self.along_glacier_geometry == 'random':
            self.filename += 'rand_'
        else:
            raise ValueError('Unknown along glacier geometry!')

        if self.glacier_state == 'equilibrium':
            self.filename += 'equ_'
        elif self.glacier_state == 'advancing':
            self.filename += 'adv_'
        elif self.glacier_state == 'retreating':
            self.filename += 'ret_'
        else:
            raise ValueError('Unknown glacier state!')

        self.filename += self.opti_var_1

        if self.opti_var_2 is not None:
            self.filename += '_and_'
            self.filename += self.opti_var_2
            self.filename += '_'
            self.filename += self.two_opti_parameters

        self.filename += '.nc'

    def callback_fct(self, x0):
        i = len(self.costs) - 1
        # make sure that there are some results to show (if no calculation was
        # perfomed there is nothing to show)
        if i >= 0:
            # save the current index for the later cleaning of the data
            self.step_indices = np.append(self.step_indices, i)
            # define the arguments for the shown text
            args = {'iteration': len(self.step_indices),
                    'opti_var': self.opti_var_iteration[-1],
                    'fct_call': self.fct_calls[-1],
                    'cost': self.costs[-1],
                    'opti_var_1_rmse': RMSE(self.guessed_opti_var_1[-1],
                                            self.true_opti_var_1),
                    'opti_var_1_bias': mean_BIAS(self.guessed_opti_var_1[-1],
                                                 self.true_opti_var_1),
                    'opti_var_1_max_dif': max_dif(self.guessed_opti_var_1[-1],
                                                  self.true_opti_var_1)
                    }

            # include arguments for second optimisation variable
            if self.opti_var_2 is not None:
                args.update(
                    {'opti_var_2_rmse': RMSE(self.guessed_opti_var_2[-1],
                                             self.true_opti_var_2),
                     'opti_var_2_bias': mean_BIAS(self.guessed_opti_var_2[-1],
                                                  self.true_opti_var_2),
                     'opti_var_2_max_dif': max_dif(self.guessed_opti_var_2[-1],
                                                   self.true_opti_var_2)
                     })

            # show text
            print(self.info_text.format(**args))

    def main_iteration_callback(self):
        text = '''
--------------------------------------------------
Main Iteration number {iteration:d}:'''
        arg = {'iteration': self.main_iterations[-1]}
        print(text.format(**arg))

    def filter_data_from_optimization(self, opti_var):
        # Filter all "exploratory" model runs to only get "real" iteration
        # steps
        index = self.step_indices
        self.grads = self.grads[index]
        self.costs = self.costs[index]
        self.c_terms = self.c_terms[index]
        self.surfs = self.surfs[index]
        self.widths = self.widths[index]
        self.fct_calls = self.fct_calls[index]
        self.opti_var_iteration = self.opti_var_iteration[index]
        self.guessed_opti_var_1 = self.guessed_opti_var_1[index]
        if self.opti_var_2 is not None:
            self.guessed_opti_var_2 = self.guessed_opti_var_2[index]
        if self.main_iterations.size != 0:
            self.current_main_iterations = self.current_main_iterations[index]

    def create_and_save_dataset(self):
        dataset = xr.Dataset(
            data_vars={
                'true_' + self.opti_var_1:
                    (['points_with_ice'],
                     self.true_opti_var_1),
                'first_guessed_' + self.opti_var_1:
                    (['points_with_ice'],
                     self.first_guessed_opti_var_1),
                'guessed_' + self.opti_var_1:
                    (['nr_of_iteration', 'points_with_ice'],
                     self.guessed_opti_var_1),
                'cost':
                    (['nr_of_iteration'],
                     self.costs),
                'cost_terms':
                    (['nr_of_iteration', 'nr_of_reg_parameter'],
                     self.c_terms),
                'gradients':
                    (['nr_of_iteration', 'points_with_ice'],
                     self.grads),
                'function_calls':
                    (['nr_of_iteration'],
                     self.fct_calls),
                'optimisation_variable':
                    (['nr_of_iteration'],
                     self.opti_var_iteration),
                'ice_mask':
                    (['total_distance'],
                     self.measurements['ice_mask']),
                'true_surface_h':
                    (['total_distance'],
                     self.true_surfs),
                'surface_h':
                    (['nr_of_iteration', 'total_distance'],
                     self.surfs),
                'true_widths':
                    (['total_distance'],
                     self.true_widths),
                'widths':
                    (['nr_of_iteration', 'total_distance'],
                     self.widths)
            },
            coords={
                'total_distance': self.geometry['distance_along_glacier'],
                'points_with_ice': np.arange(len(self.true_opti_var_1)),
                'nr_of_iteration': np.arange(len(self.step_indices)) + 1,
                'nr_of_reg_parameter': np.arange(len(self.reg_parameters))
            },
            attrs={
                'reg_parameters': self.reg_parameters,
                'glacier_state': self.glacier_state,
                'geometry': self.geometry,
                'measurements': self.measurements,
                'geometry_of_bed_h': self.geometry_bed_h,
                'along_glacier_geometry': self.along_glacier_geometry,
                'solver': self.solver,
                'minimize_options': self.minimize_options,
                'computing_time': self.computing_time
            })

        # check if there is a second optimisation variable
        if self.opti_var_2 is not None:
            # save additional data from second optimisation variable
            dataset['true_' + self.opti_var_2] = \
                (['points_with_ice'],
                 self.true_opti_var_2)
            dataset['first_guessed_' + self.opti_var_2] = \
                (['points_with_ice'],
                 self.first_guessed_opti_var_2)
            dataset['guessed_' + self.opti_var_2] = \
                (['nr_of_iteration', 'points_with_ice'],
                 self.guessed_opti_var_2)
            dataset.attrs['optimisation of two variables'] = \
                self.two_opti_parameters

        # save current main iteration (only when optimisaton for two parameters
        # is separated)
        if self.main_iterations.size != 0:
            dataset['current_main_iterations'] = \
                (['nr_of_iteration'],
                 self.current_main_iterations)
            dataset.attrs['max number of main iteration'] = \
                self.main_iterations_separeted

        # save dataset as netcdf
        dataset.to_netcdf(self.filename)

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

    def plot_glacier_shape(self):
        glacier_shape = hv.Curve((self.geometry['distance_along_glacier'],
                                  self.geometry['bed_shapes']),
                                 'distance',
                                 'shape factor',
                                 label='glacier shape factor'
                                 ).opts(line_color='black',
                                        width=600,
                                        height=400)

        return (glacier_shape).opts(opts.Curve(line_width=3))
