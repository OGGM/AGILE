import torch
import numpy as np
import xarray as xr
import holoviews as hv
from holoviews import opts
from combine.core.arithmetics import RMSE, mean_BIAS, max_dif, \
    MaxCalculationTimeReached
import os
hv.extension('matplotlib')

# from combine.utils.optimization import LCurveTest


class DataLogger(object):

    def __init__(self, bed_geometry, opti_parameter, two_parameter_option,
                 main_iterations_separeted, geometry, measurements,
                 first_guess, fg_sfc_h, fg_widths, reg_parameters,
                 used_bed_h_geometry, used_along_glacier_geometry,
                 minimize_options, max_time_minimize, solver, glacier_state,
                 mb_opts, grad_scaling, filename_suffix, task_id):
        # first save all initial data for idealized experiment
        self.bed_geometry = bed_geometry
        self.opti_parameter = opti_parameter
        self.geometry = geometry
        self.measurements = measurements
        self.first_guess = first_guess
        self.fg_sfc_h = fg_sfc_h
        self.fg_widths = fg_widths
        self.reg_parameters = reg_parameters
        self.geometry_bed_h = used_bed_h_geometry
        self.along_glacier_geometry = used_along_glacier_geometry
        self.glacier_state = glacier_state
        self.ice_mask = self.measurements['ice_mask']
        self.mb_opts = mb_opts
        self.two_parameter_option = two_parameter_option
        self.solver = solver
        self.minimize_options = minimize_options
        self.max_time_minimize = max_time_minimize
        self.main_iterations_separeted = main_iterations_separeted
        self.filename_suffix = filename_suffix
        self.grad_scaling = grad_scaling

        # task_id only needed for cluster computation
        self.task_id = task_id

        # define some variables needed for all bed_geometries
        self.costs = np.empty((0, 1))
        self.c_terms = np.empty((0, len(self.reg_parameters)))
        self.sfc_h = np.empty((0, self.geometry['nx']))
        self.true_sfc_h = self.measurements['sfc_h']
        self.widths = np.empty((0, self.geometry['nx']))
        self.true_widths = self.measurements['widths']
        self.time_needed = np.empty((0, 1))

        # variable to keep track in which main iteration the algorithm is at
        # the current minimisation, only needed for separated optimisation
        self.current_main_iterations = np.empty((0, 1))

        # variable to keep track of time
        self.start_time = None
        self.total_computing_time = 'no time recorded'
        self.fct_calls = np.array([0])
        # save optimisation variable of current iteration
        self.opti_var_iteration = np.empty((0, 1))
        # save step indices for filtering out steps only needed by the
        # minimisation algorithm
        self.step_indices = np.empty((0, 1), dtype=np.int_)
        # variable to keep track of the main iterations,
        # only needed for separated optimisation of two parameters
        self.main_iterations = np.empty((0, 1), dtype=np.int_)

        # help variable for two optimisation variables
        two_opti_parameter_options = ['separated', 'at_once', 'calculated']

        # create variables according to the bed_geometry and opti_parameter
        if bed_geometry == 'rectangular':
            if opti_parameter == 'bed_h':
                self.opti_var_1 = 'bed_h'
                self.opti_var_2 = None
                self.geometry_var = 'widths'

            else:
                raise ValueError('Unknown optimisation parameter for '
                                 'rectangular!')

        elif bed_geometry == 'parabolic':
            if opti_parameter == 'bed_h':
                self.opti_var_1 = 'bed_h'
                self.opti_var_2 = None
                self.geometry_var = 'bed_shape'

            elif opti_parameter == 'bed_shape':
                self.opti_var_1 = 'bed_shape'
                self.opti_var_2 = None
                self.geometry_var = 'bed_h'

            elif opti_parameter == 'bed_h and bed_shape':
                self.opti_var_1 = 'bed_h'
                self.opti_var_2 = 'bed_shape'

                if two_parameter_option not in two_opti_parameter_options:
                    raise ValueError('Unknown optimisation option for two '
                                     'parameters!')

            else:
                raise ValueError('Unknown optimisation parameter for '
                                 'parabolic!')

        elif bed_geometry == 'trapezoidal':
            if opti_parameter == 'bed_h':
                self.opti_var_1 = 'bed_h'
                self.opti_var_2 = None
                self.geometry_var = 'w0'

            elif opti_parameter == 'w0':
                self.opti_var_1 = 'w0'
                self.opti_var_2 = None
                self.geometry_var = 'bed_h'

            elif opti_parameter == 'bed_h and w0':
                self.opti_var_1 = 'bed_h'
                self.opti_var_2 = 'w0'

                if two_parameter_option not in two_opti_parameter_options:
                    raise ValueError('Unknown optimisation option for two '
                                     'parameters!')

            else:
                raise ValueError('Unknown optimisation parameter for '
                                 'trapezoidal!')

        else:
            raise ValueError('Unknown bed geometry!')

        # save the true and the first guess for optimisation variable 1
        self.true_opti_var_1 = self.geometry[self.opti_var_1][self.ice_mask]
        self.first_guessed_opti_var_1 = self.first_guess[self.opti_var_1]
        self.known_opti_var_1 = self.geometry[self.opti_var_1][~self.ice_mask]
        # variable for saving the iteration output
        self.guessed_opti_var_1 = np.empty((0, self.geometry['nx']))
        # variable for gradients
        self.grads_opti_var_1 = np.empty((0, self.geometry['nx']))
        # variable to save last message of minimisation algorithm
        self.message_opti_var_1 = ''

        # check if second optimisation variable is needed
        if self.opti_var_2 is not None:
            # safe the true and the first guess for optimisation variable 2
            self.true_opti_var_2 = \
                self.geometry[self.opti_var_2][self.ice_mask]
            self.first_guessed_opti_var_2 = self.first_guess[self.opti_var_2]
            self.known_opti_var_2 = \
                self.geometry[self.opti_var_2][~self.ice_mask]
            # variable for saving the iteration output
            self.guessed_opti_var_2 = np.empty((0, self.geometry['nx']))
            # variable for gradients
            self.grads_opti_var_2 = np.empty((0, self.geometry['nx']))
            # variable to save last message of minimisation algorithm
            self.message_opti_var_2 = ''

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
        elif bed_geometry == 'trapezoidal':
            self.filename = 'tra_'
        else:
            raise ValueError('Unknown bed geometry!')

        if self.geometry_bed_h == 'linear':
            self.filename += 'line_'
        elif self.geometry_bed_h == 'cliff':
            self.filename += 'clif_'
        elif self.geometry_bed_h == 'random':
            self.filename += 'rand_'
        elif self.geometry_bed_h == 'HEF':
            self.filename += 'HEF_'
        else:
            raise ValueError('Unknown bed height geometry!')

        if self.along_glacier_geometry == 'constant':
            self.filename += 'cons_'
        elif self.along_glacier_geometry == 'random':
            self.filename += 'rand_'
        elif self.along_glacier_geometry == 'wide_top':
            self.filename += 'wide_'
        elif self.along_glacier_geometry == 'HEF':
            self.filename += ''
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
            self.filename += self.two_parameter_option

    def save_data_in_datalogger(self, var, data):
        if type(data) == torch.Tensor:
            data = data.detach().numpy().astype(np.float64)
        elif type(data) != np.ndarray:
            data = np.array(data)

        current_var = getattr(self, var)
        new_var = np.reshape(np.append(current_var, data), (-1, data.size))
        setattr(self, var, new_var)

    def callback_fct(self, x0):
        i = len(self.costs) - 1
        # make sure that there are some results to show (if no calculation was
        # perfomed there is nothing to show)
        if i >= 0:
            # save the current index for the later cleaning of the data
            self.step_indices = np.append(self.step_indices, i)
            # define the arguments for the shown text
            args = {'iteration': len(self.step_indices),
                    'opti_var': self.opti_var_iteration[-1][0],
                    'fct_call': self.fct_calls[-1],
                    'cost': self.costs[-1][0],
                    'opti_var_1': self.opti_var_1,
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
                    {'opti_var_2': self.opti_var_2,
                     'opti_var_2_rmse': RMSE(self.guessed_opti_var_2[-1],
                                             self.true_opti_var_2),
                     'opti_var_2_bias': mean_BIAS(self.guessed_opti_var_2[-1],
                                                  self.true_opti_var_2),
                     'opti_var_2_max_dif': max_dif(self.guessed_opti_var_2[-1],
                                                   self.true_opti_var_2)
                     })

            # show text
            print(self.info_text.format(**args))

            if self.max_time_minimize is not None:
                # raise an exception if maximum calculation time is reached
                if self.time_needed[-1] > self.max_time_minimize:
                    raise MaxCalculationTimeReached()

    def main_iteration_callback(self):
        if self.two_parameter_option == 'separated':
            text = '''
--------------------------------------------------
Main Iteration number {iteration:d}:'''
            arg = {'iteration': self.main_iterations[-1]}
            print(text.format(**arg))

    def squeeze_generic(self, a, axes_to_keep=[0]):
        out_s = [s for i, s in enumerate(a.shape)
                 if i in axes_to_keep or s != 1]
        return a.reshape(out_s)

    def filter_data_from_optimization(self):
        # Filter all "exploratory" model runs to only get "real" iteration
        # steps
        index = self.step_indices
        self.grads_opti_var_1 = self.grads_opti_var_1[index]
        self.costs = self.squeeze_generic(self.costs[index])
        self.c_terms = self.c_terms[index]
        self.sfc_h = self.sfc_h[index]
        self.widths = self.widths[index]
        self.time_needed = self.squeeze_generic(self.time_needed[index])
        # + 1 in index because fct_calls starts with [0] and not empty
        self.fct_calls = self.squeeze_generic(self.fct_calls[index + 1])
        self.opti_var_iteration = self.squeeze_generic(
            self.opti_var_iteration[index])
        self.guessed_opti_var_1 = self.guessed_opti_var_1[index]
        if self.opti_var_2 is not None:
            self.guessed_opti_var_2 = self.guessed_opti_var_2[index]
            self.grads_opti_var_2 = self.grads_opti_var_2[index]
        if self.main_iterations.size is not None:
            self.current_main_iterations = self.squeeze_generic(
                self.current_main_iterations[index])

    def create_and_save_dataset(self):
        dataset = xr.Dataset(
            data_vars={
                'spinup_sfc_h':
                    (['total_distance'],
                     self.measurements['spinup_sfc']),
                'true_' + self.opti_var_1:
                    (['points_with_ice'],
                     self.true_opti_var_1),
                'first_guessed_' + self.opti_var_1:
                    (['points_with_ice'],
                     self.first_guessed_opti_var_1),
                'ice_mask':
                    (['total_distance'],
                     self.squeeze_generic(self.ice_mask)),
                'true_surface_h':
                    (['total_distance'],
                     self.squeeze_generic(self.true_sfc_h)),
                'first_guess_surface_h':
                    (['total_distance'],
                     self.squeeze_generic(self.fg_sfc_h)),
                'true_widths':
                    (['total_distance'],
                     self.squeeze_generic(self.true_widths)),
                'first_guess_widths':
                    (['total_distance'],
                     self.squeeze_generic(self.fg_widths)),
                'total_true_' + self.opti_var_1:
                    (['total_distance'],
                     self.geometry[self.opti_var_1])
            },
            coords={
                'total_distance': self.geometry['distance_along_glacier'],
                'points_with_ice': np.arange(len(self.true_opti_var_1))
            },
            attrs={
                'reg_parameters': self.reg_parameters,
                'glacier_state': self.glacier_state,
                'domain_points': self.geometry['nx'],
                'distance_between_points_m': self.geometry['map_dx'],
                'years_of_model_run': self.measurements['yrs_to_run'],
                'mb_ELA': self.mb_opts['ELA'],
                'mb_grad': self.mb_opts['grad'],
                'geometry_of_bed_h': self.geometry_bed_h,
                'along_glacier_geometry': self.along_glacier_geometry,
                'solver': self.solver,
                'total_computing_time': self.total_computing_time,
                'last minimisation message' + self.opti_var_1:
                    self.message_opti_var_1,
                'minimisation_possible': 'no',
                'max_number_of_main_iteration(separeted)':
                    self.main_iterations_separeted
            })

        if self.max_time_minimize is not None:
            dataset.attrs['max_time_minimize'] = self.max_time_minimize

        # check if there is a second optimisation variable
        if self.opti_var_2 is not None:
            # save additional data from second optimisation variable
            dataset['true_' + self.opti_var_2] = \
                (['points_with_ice'],
                 self.squeeze_generic(self.true_opti_var_2))
            dataset['first_guessed_' + self.opti_var_2] = \
                (['points_with_ice'],
                 self.squeeze_generic(self.first_guessed_opti_var_2))
            dataset['total_true_' + self.opti_var_2] = \
                (['total_distance'],
                 self.geometry[self.opti_var_2])
            dataset.attrs['optimisation of two variables'] = \
                self.two_parameter_option
            dataset.attrs['last minimisation message' + self.opti_var_2] =\
                self.message_opti_var_2

        # add options of minimisation function with prefix 'minimize_'
        for key in self.minimize_options:
            if key != 'disp':
                dataset.attrs['minimize_' + key] = \
                    self.minimize_options[key]

        # check if minimisation could find one improvement
        if np.arange(len(self.step_indices)).size != 0:
            dataset.coords['nr_of_iteration'] = \
                np.arange(len(self.step_indices)) + 1
            dataset.coords['nr_of_reg_parameter'] = \
                np.arange(len(self.reg_parameters))
            dataset['guessed_' + self.opti_var_1] = \
                (['nr_of_iteration', 'points_with_ice'],
                 self.guessed_opti_var_1)
            dataset['cost'] = \
                (['nr_of_iteration'],
                 self.costs)
            dataset['cost_terms'] = \
                (['nr_of_iteration', 'nr_of_reg_parameter'],
                 self.c_terms)
            dataset['gradients_' + self.opti_var_1] = \
                (['nr_of_iteration', 'points_with_ice'],
                 self.grads_opti_var_1)
            dataset['function_calls'] = \
                (['nr_of_iteration'],
                 self.fct_calls)
            dataset['optimisation_variable'] = \
                (['nr_of_iteration'],
                 self.opti_var_iteration)
            dataset['surface_h'] = \
                (['nr_of_iteration', 'total_distance'],
                 self.sfc_h)
            dataset['widths'] = \
                (['nr_of_iteration', 'total_distance'],
                 self.widths)
            dataset['computing_time'] = \
                (['nr_of_iteration'],
                 self.time_needed)
            dataset.attrs['minimisation_possible'] = 'yes'

            # check if there is a second optimisation variable
            if self.opti_var_2 is not None:
                # save additional data from second optimisation variable
                dataset['guessed_' + self.opti_var_2] = \
                    (['nr_of_iteration', 'points_with_ice'],
                     self.guessed_opti_var_2)
                dataset['gradients_' + self.opti_var_2] = \
                    (['nr_of_iteration', 'points_with_ice'],
                     self.grads_opti_var_2)

            # save current main iteration (only when optimisaton for two
            # parameters is separated)
            dataset['current_main_iterations'] = \
                (['nr_of_iteration'],
                 self.current_main_iterations)

            # save scaling of gradients if 'at_once'
            if self.two_parameter_option == 'at_once':
                dataset.attrs['bed_h gradient scaling'] = \
                    self.grad_scaling['bed_h']
                dataset.attrs['shape_var gradient scaling'] = \
                    self.grad_scaling['shape_var']

            # add to filename if maxiterations were used
            dataset.attrs['maxiter_reached'] = 'no'
            if (self.opti_var_2 is None) or \
               (self.two_parameter_option in ['at_once', 'calculated']):
                if (len(self.step_indices) ==
                   self.minimize_options['maxiter']):
                    # self.filename += '_maxiter'
                    # task_id only needed for cluster computation
                    # self.filename += ('_' + str(self.task_id))
                    dataset.attrs['maxiter_reached'] = 'yes'
            elif self.two_parameter_option == 'separated':
                if self.current_main_iterations[-1] == \
                   self.main_iterations_separeted:
                    # self.filename += '_maxiter'
                    # task_id only needed for cluster computation
                    # self.filename += ('_' + str(self.task_id))
                    dataset.attrs['maxiter_reached'] = 'yes'
            else:
                raise ValueError('Somthing went wrong by checking if maxiter'
                                 ' was used!')

        # add given filename suffix
        self.filename += self.filename_suffix

        # check if filename already exists, prevent overwriting
        list_of_files = os.listdir()
        if (self.filename + '.nc') in list_of_files:
            for file_nr in range(10):
                if (self.filename + '_' + str(file_nr) + '.nc') \
                   not in list_of_files:
                    self.filename += ('_' + str(file_nr))
                    break
                if file_nr == 9:
                    raise ValueError('There are to many files with the same '
                                     'name!')

        # save dataset as netcdf
        dataset.to_netcdf(self.filename + '.nc')

    def save_result_plot(self, plot_height=450, plot_width=800):
        x = self.geometry['distance_along_glacier']

        # start with first optimisation variable
        y1_final_guess = np.zeros(len(x))
        y1_first_guess = np.zeros(len(x))

        y1_final_guess[self.ice_mask] = self.guessed_opti_var_1[-1]
        y1_final_guess[~self.ice_mask] = self.known_opti_var_1

        y1_first_guess[self.ice_mask] = self.first_guessed_opti_var_1
        y1_first_guess[~self.ice_mask] = self.known_opti_var_1

        # if only one
        if self.opti_parameter in ['bed_h', 'bed_shape', 'w0']:
            y2_final_guess = None
            y2_first_guess = None

        elif self.opti_parameter in ['bed_h and bed_shape', 'bed_h and w0']:
            y2_final_guess = np.zeros(len(x))
            y2_first_guess = np.zeros(len(x))

            y2_final_guess[self.ice_mask] = self.guessed_opti_var_2[-1]
            y2_final_guess[~self.ice_mask] = self.known_opti_var_2

            y2_first_guess[self.ice_mask] = self.first_guessed_opti_var_2
            y2_first_guess[~self.ice_mask] = self.known_opti_var_2

        else:
            raise ValueError('Unknown optimisation parameter!')

        # plot for first optimisation variable
        diff_estimated_opti_1 = hv.Curve((x,
                                          y1_final_guess -
                                          self.geometry[self.opti_var_1]),
                                         'distance',
                                         'diff ' + self.opti_var_1,
                                         label=('diff estimated ' +
                                                self.opti_var_1))

        diff_first_guess_opti_1 = hv.Curve((x,
                                            y1_first_guess -
                                            self.geometry[self.opti_var_1]),
                                           'distance',
                                           'diff ' + self.opti_var_1,
                                           label=('diff first guess ' +
                                                  self.opti_var_1))

        zero_line_opti_1 = hv.Curve((x,
                                     np.zeros(len(x))),
                                    'distance',
                                    'diff ' + self.opti_var_1
                                    ).opts(color='black',
                                           show_grid=True)

        rmse_opti_1 = hv.Curve(([0], [0]),
                               'distance',
                               'diff ' + self.opti_var_1,
                               label='RMSE: {:.4f}'.format(
                                   RMSE(self.guessed_opti_var_1[-1],
                                        self.true_opti_var_1))
                               ).opts(visible=False)

        bias_opti_1 = hv.Curve(([0], [0]),
                               'distance',
                               'diff ' + self.opti_var_1,
                               label='BIAS: {:.4f}'.format(
                                   mean_BIAS(self.guessed_opti_var_1[-1],
                                             self.true_opti_var_1))
                               ).opts(visible=False)

        diff_opti_1 = hv.Curve(([0], [0]),
                               'distance',
                               'diff ' + self.opti_var_1,
                               label='DIFF: {:.4f}'.format(
                                   max_dif(self.guessed_opti_var_1[-1],
                                           self.true_opti_var_1))
                               ).opts(visible=False)

        opti_1_plot = (zero_line_opti_1 *
                       diff_first_guess_opti_1 *
                       diff_estimated_opti_1 *
                       rmse_opti_1 *
                       bias_opti_1 *
                       diff_opti_1
                       ).opts(xaxis='top',
                              legend_position='best')

        if y2_final_guess is not None:
            # plot for first optimisation variable
            diff_estimated_opti_2 = hv.Curve((x,
                                              y2_final_guess -
                                              self.geometry[self.opti_var_2]),
                                             'distance',
                                             'diff ' + self.opti_var_2,
                                             label='diff estimated ' +
                                             self.opti_var_2)

            diff_first_guess_opti_2 = hv.Curve((x,
                                                y2_first_guess -
                                                self.geometry[self.opti_var_2]
                                                ),
                                               'distance',
                                               'diff ' + self.opti_var_2,
                                               label='diff first guess ' +
                                               self.opti_var_2)

            zero_line_opti_2 = hv.Curve((x,
                                         np.zeros(len(x))),
                                        'distance',
                                        'diff ' + self.opti_var_2
                                        ).opts(color='black',
                                               show_grid=True)

            rmse_opti_2 = hv.Curve(([0], [0]),
                                   'distance',
                                   'diff ' + self.opti_var_2,
                                   label='RMSE: {:.4f}'.format(
                                       RMSE(self.guessed_opti_var_2[-1],
                                            self.true_opti_var_2))
                                   ).opts(visible=False)

            bias_opti_2 = hv.Curve(([0], [0]),
                                   'distance',
                                   'diff ' + self.opti_var_2,
                                   label='BIAS: {:.4f}'.format(
                                       mean_BIAS(self.guessed_opti_var_2[-1],
                                                 self.true_opti_var_2))
                                   ).opts(visible=False)

            diff_opti_2 = hv.Curve(([0], [0]),
                                   'distance',
                                   'diff ' + self.opti_var_2,
                                   label='DIFF: {:.4f}'.format(
                                       max_dif(self.guessed_opti_var_2[-1],
                                               self.true_opti_var_2))
                                   ).opts(visible=False)

            opti_2_plot = (zero_line_opti_2 *
                           diff_first_guess_opti_2 *
                           diff_estimated_opti_2 *
                           rmse_opti_2 *
                           bias_opti_2 *
                           diff_opti_2
                           ).opts(xaxis='bottom',
                                  legend_position='best')

            final_plot = (opti_1_plot.opts(aspect=4,
                                           fontsize={'ticks': 15,
                                                     'labels': 25,
                                                     'legend': 20,
                                                     'legend_title': 20},
                                           show_frame=True) +
                          opti_2_plot.opts(aspect=4,
                                           xaxis='bottom',
                                           show_grid=True,
                                           fontsize={'ticks': 15,
                                                     'labels': 25,
                                                     'legend': 20,
                                                     'legend_title': 20},
                                           show_frame=True)).cols(1)

            final_plot.opts(sublabel_format='',
                            tight=True,
                            fig_inches=20,
                            fontsize={'title': 27})

        else:
            final_plot = opti_1_plot.opts(aspect=2,
                                          fig_inches=20,
                                          show_frame=True,
                                          fontsize={'ticks': 15,
                                                    'labels': 25,
                                                    'legend': 20,
                                                    'legend_title': 20,
                                                    'title': 27})

        final_plot.opts(title=(self.filename + '\n reg_par = ' +
                               str(self.reg_parameters) + ', iterations: ' +
                               str(len(self.step_indices)) + ', fct_calls: ' +
                               str(self.fct_calls[-1])))
        final_plot.opts(opts.Curve(linewidth=3))

        hv.save(final_plot,
                self.filename,
                fmt='svg')

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
