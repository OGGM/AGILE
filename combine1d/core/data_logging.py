import torch
import numpy as np
import xarray as xr
import time
from combine1d.core.arithmetics import RMSE, mean_BIAS, max_dif
from combine1d.core.exception import MaxCalculationTimeReached
import os
import warnings
import copy
from oggm.core.flowline import FileModel


class DataLogger(object):

    def __init__(self, gdir, fls_init, inversion_input, climate_filename,
                 climate_filesuffix, output_filesuffix):

        self.gdir = gdir
        # first extract all needed data (is also a check that everything is there before starting)
        self.inversion_input = inversion_input
        self.obs_reg_parameters = inversion_input['obs_reg_parameters']
        self.regularisation_terms = inversion_input['regularisation_terms']
        self.control_vars = inversion_input['control_vars']
        self.observations = inversion_input['observations']
        self.mb_models_settings = inversion_input['mb_models_settings']
        self.min_w0_m = inversion_input['min_w0_m']
        self.spinup_options = inversion_input['spinup_options']
        self.solver = inversion_input['solver']
        self.minimize_options = inversion_input['minimize_options']
        self.max_time_minimize = inversion_input['max_time_minimize']

        # if cuda is wanted check if available
        if inversion_input['device'] == 'cuda':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu":
                warnings.warn('Cuda was selected for calculation but is not available! '
                              'Continue calculation on cpu!')
        else:
            self.device = "cpu"

        if inversion_input['torch_type'] == 'float':
            self.torch_type = torch.float
        elif inversion_input['torch_type'] == 'double':
            self.torch_type = torch.double
        else:
            raise ValueError('Unknown torch type ' + inversion_input['torch_type'] + '!')

        self.flowline_init = fls_init[0]
        self.ice_mask = np.where(self.flowline_init.thick > 10e-2, True, False)
        self.climate_filename = climate_filename
        self.climate_filesuffix = climate_filesuffix
        self.output_filesuffix = output_filesuffix

        # save time of initialisation, maybe reset later
        self.start_time = time.time()

        # define variables which are filled during the minimisation run
        self.costs = None
        self.c_terms = None
        self.time_needed = None
        self.flowlines = None
        self.end_time = None
        self.known_parameters = None
        self.len_unknown_parameter = None
        self.parameter_indices = None
        self.unknown_parameters = None
        self.fct_calls = np.array([0])
        self.step_indices = np.array([0])  # include zero to save outcome of first guess run

        self.filename = 'COMBINE_inversion_results'

        # create info Text for callback_fct TODO: think about showing the evolution of the c_terms
        self.info_text = '''

    Iteration: {iteration:d}
    Total Function calls: {fct_call:d}
    Needed Time: {time_needed:g}
    Cost: {cost:g}
    '''

    def add_true_bed(self):
        raise NotImplementedError('Not added yet!')

    def save_data_in_datalogger(self, var, data):
        if type(data) == torch.Tensor:
            data = data.detach().numpy().astype(np.float64)
        elif type(data) != np.ndarray:
            data = np.array(data)

        current_var = getattr(self, var)

        if current_var is None:
            new_var = data
        else:
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
                    'fct_call': self.fct_calls[-1],
                    'time_needed': self.time_needed[-1],
                    'cost': self.costs[-1][0],
                    }

            # show text
            print(self.info_text.format(**args))

            if self.max_time_minimize is not None:
                # raise an exception if maximum calculation time is reached
                if self.time_needed[-1] > self.max_time_minimize:
                    raise MaxCalculationTimeReached()

    def squeeze_generic(self, a, axes_to_keep=[0]):
        out_s = [s for i, s in enumerate(a.shape)
                 if i in axes_to_keep or s != 1]
        return a.reshape(out_s)

    def filter_data_from_optimization(self):
        # Filter all "exploratory" model runs to only get "real" iteration
        # steps
        index = self.step_indices
        self.costs = self.squeeze_generic(self.costs[index])
        self.c_terms = self.c_terms[index]
        self.time_needed = self.squeeze_generic(self.time_needed[index])
        self.flowline = self.flowline[index]
        self.unknown_parameters = self.unknown_parameters[index]
        self.fct_calls = self.squeeze_generic(self.fct_calls[index + 1])

    def create_and_save_dataset(self):
        if True:
            raise NotImplementedError('Need update!')

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
                     self.squeeze_generic(self.fg_run['sfc_h'])),
                'true_widths':
                    (['total_distance'],
                     self.squeeze_generic(self.true_widths)),
                'first_guess_widths':
                    (['total_distance'],
                     self.squeeze_generic(self.fg_run['widths'])),
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
                'max_number_of_main_iteration(iterative)':
                    self.main_iterations_iterative
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
            dataset.attrs['last minimisation message' + self.opti_var_2] = \
                self.message_opti_var_2

        # add options of minimisation function with prefix 'minimize_'
        for key in self.minimize_options:
            if key != 'disp':
                dataset.attrs['minimize_' + key] = \
                    self.minimize_options[key]

        if self.glacier_state == 'retreating with unknow spinup':
            dataset.coords['one_dimension'] = np.arange(1)
            dataset['true_spinup_ELA'] = \
                (['one_dimension'],
                 np.array([self.mb_opts['ELA'][0]]))
            dataset['true_spinup_sfc_h'] = \
                (['total_distance'],
                 self.measurements['true_spinup_sfc'])
            dataset['true_spinup_widths'] = \
                (['total_distance'],
                 self.measurements['true_spinup_widths'])
            dataset['first_guess_spinup_ELA'] = \
                (['one_dimension'],
                 np.array([self.first_guess['spinup_ELA']]))

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
            # parameters is iterative)
            dataset['current_main_iterations'] = \
                (['nr_of_iteration'],
                 self.current_main_iterations)

            if self.glacier_state == 'retreating with unknow spinup':
                dataset['spinup_ELA_guessed'] = \
                    (['nr_of_iteration'],
                     self.spinup_ELA_guessed)
                dataset['spinup_ELA_grad'] = \
                    (['nr_of_iteration'],
                     self.spinup_ELA_grad)
                dataset['spinup_sfc_h_guessed'] = \
                    (['nr_of_iteration', 'total_distance'],
                     self.spinup_sfc_h_guessed)
                dataset['spinup_widths_guessed'] = \
                    (['nr_of_iteration', 'total_distance'],
                     self.spinup_widths_guessed)

            # save scaling of gradients if 'explicit'
            if self.two_parameter_option == 'explicit':
                dataset.attrs['bed_h gradient scaling'] = \
                    self.grad_scaling['bed_h']
                dataset.attrs['shape_var gradient scaling'] = \
                    self.grad_scaling['shape_var']

            dataset.attrs['maxiter_reached'] = 'no'
            if (self.opti_var_2 is None) or \
                    (self.two_parameter_option in ['explicit', 'implicit']):
                if (len(self.step_indices) ==
                        self.minimize_options['maxiter']):
                    dataset.attrs['maxiter_reached'] = 'yes'
            elif self.two_parameter_option == 'iterative':
                if self.current_main_iterations[-1] == \
                        self.main_iterations_iterative:
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
        dataset.to_netcdf(self.filename + self.output_filesuffix + '.nc')


def initialise_DataLogger(gdir, inversion_input_filesuffix='_combine', init_model_filesuffix=None,
                          init_model_fls=None, climate_filename='climate_historical',
                          climate_filesuffix='', output_filesuffix='_combine'):
    '''
    extract information out of gdir and save in datalogger.

    Parameters
    ----------
    gdir

    Returns
    -------

    '''

    if init_model_filesuffix is not None:
        fp = gdir.get_filepath('model_geometry',
                               filesuffix=init_model_filesuffix)
        fmod = FileModel(fp)
        init_model_fls = fmod.fls

    if init_model_fls is None:
        fls_init = gdir.read_pickle('model_flowlines')
    else:
        fls_init = copy.deepcopy(init_model_fls)

    if len(fls_init) > 1:
        raise NotImplementedError('COMBINE only works with single flowlines!')

    # include check if inversion_input file exist
    if not os.path.isfile(gdir.get_filepath('inversion_input',
                                            filesuffix=inversion_input_filesuffix)):
        raise AttributeError('inversion_input' + inversion_input_filesuffix +
                             ' file not found!')
    inversion_input = gdir.read_pickle(filename='inversion_input',
                                       filesuffix=inversion_input_filesuffix)

    # here fill 'observations' with initial flowline values if needed
    rgi_yr = gdir.rgi_date

    def check_if_year_is_given(obs_check):
        if inversion_input['observations'][obs_check] == {}:
            inversion_input['observations'][obs_check] = {str(rgi_yr): []}
        return [key for key in inversion_input['observations'][obs_check].keys()][0]

    for obs in inversion_input['observations'].keys():
        if obs == 'fl_surface_h:m':
            yr_obs = check_if_year_is_given(obs)
            inversion_input['observations'][obs][yr_obs] = fls_init[0].surface_h
        if obs == 'fl_widths:m':
            yr_obs = check_if_year_is_given(obs)
            inversion_input['observations'][obs][yr_obs] = fls_init[0].widths_m
        if obs == 'fl_total_area:m2':
            yr_obs = check_if_year_is_given(obs)
            inversion_input['observations'][obs][yr_obs] = fls_init[0].area_m2
        if obs == 'fl_total_area:km2':
            yr_obs = check_if_year_is_given(obs)
            inversion_input['observations'][obs][yr_obs] = fls_init[0].area_km2

    data_logger = DataLogger(gdir, fls_init, inversion_input, climate_filename,
                             climate_filesuffix, output_filesuffix)

    return data_logger
