import pickle
import logging

import torch
import numpy as np
import xarray as xr
import time
from agile1d.core.arithmetics import RMSE, mean_BIAS, max_dif
from agile1d.core.exception import MaxCalculationTimeReached
from agile1d.core.flowline import FluxBasedModel, SemiImplicitModel
import os
import warnings
import copy
from oggm.core.flowline import FileModel, MixedBedFlowline

# Module logger
log = logging.getLogger(__name__)


class DataLogger(object):

    def __init__(self, gdir, fls_init, inversion_input, climate_filename,
                 climate_filesuffix, output_filesuffix, output_filepath):

        self.gdir = gdir
        # first extract all needed data (is also a check that everything is there before starting)
        self.inversion_input = inversion_input
        self.obs_scaling_parameters = inversion_input['obs_scaling_parameters']
        self.regularisation_terms = inversion_input['regularisation_terms']
        self.cost_lambda = inversion_input['cost_lambda']
        self.observations = inversion_input['observations']
        self.observations_for_scaling = inversion_input['observations_for_scaling']
        self.mb_models_settings = inversion_input['mb_models_settings']
        self.min_w0_m = inversion_input['min_w0_m']
        self.min_ice_thickness = inversion_input['min_ice_thickness']
        self.max_ice_thickness = inversion_input['max_ice_thickness']
        self.additional_ice_thickness = inversion_input['additional_ice_thickness']
        self.bed_h_bounds = inversion_input['bed_h_bounds']
        self.max_deviation_surface_h = inversion_input['max_deviation_surface_h']
        self.limits_lambda = inversion_input['limits_lambda']
        self.limits_height_shift_spinup = inversion_input['limits_height_shift_spinup']
        self.spinup_options = inversion_input['spinup_options']
        self.solver = inversion_input['solver']
        self.minimize_options = inversion_input['minimize_options']
        self.max_time_minimize = inversion_input['max_time_minimize']
        self.control_vars_characteristic_scale = \
            inversion_input['control_vars_characteristic_scale']

        if inversion_input['dynamic_model'] == 'flux_based':
            self.dynamic_model = FluxBasedModel
        elif inversion_input['dynamic_model'] == 'implicit':
            self.dynamic_model = SemiImplicitModel
        else:
            raise ValueError("Unknown dynamic model "
                             f"{inversion_input['dynamic_model']}! (Options "
                             f"are 'flux_based' or 'implicit'")

        if self.spinup_options is None:
            spinup_type = None
        # if spinup option is surface_h must be added as control variable
        elif 'surface_h' in list(self.spinup_options.keys()):
            spinup_type = 'surface_h'
            inversion_input['control_vars'].append(spinup_type)
        elif 'section' in list(self.spinup_options.keys()):
            spinup_type = 'section'
            inversion_input['control_vars'].append(spinup_type)
        elif 'height_shift' in list(self.spinup_options.keys()):
            spinup_type = 'height_shift_spinup'
            inversion_input['control_vars'].append(spinup_type)
        elif list(self.spinup_options.keys())[0] in ['perfect_sfc_h',
                                                     'perfect_thickness',
                                                     'perfect_section']:
            spinup_type = list(self.spinup_options.keys())[0]
            fls_true_init = gdir.read_pickle(
                'model_flowlines',
                filesuffix=self.spinup_options[spinup_type])[0]
            if spinup_type == 'perfect_sfc_h':
                self.perfect_spinup_value = fls_true_init.surface_h
            elif spinup_type == 'perfect_thickness':
                self.perfect_spinup_value = fls_true_init.thick
            elif spinup_type == 'perfect_section':
                self.perfect_spinup_value = fls_true_init.section
            else:
                raise NotImplementedError(f'{spinup_type}')
        else:
            raise NotImplementedError(f'{self.spinup_options.keys()}')

        if spinup_type is not None:
            if 'perfect_bed_h' in list(self.spinup_options.keys()):
                fls_true_init = gdir.read_pickle(
                    'model_flowlines',
                    filesuffix=self.spinup_options['perfect_bed_h'])[0]
                self.perfect_spinup_value = fls_true_init.bed_h
        self.spinup_type = spinup_type
        self.control_vars = inversion_input['control_vars']

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
        if isinstance(self.flowline_init, MixedBedFlowline):
            self.is_trapezoid = (self.flowline_init.is_trapezoid &
                                 ~self.flowline_init.is_rectangular)
            self.is_rectangular = self.flowline_init.is_rectangular
            self.is_parabolic = ~self.flowline_init.is_trapezoid
        else:
            raise TypeError('agile 1D only works with MixedBedFlowline!')

        self.climate_filename = climate_filename
        self.climate_filesuffix = climate_filesuffix
        self.output_filesuffix = output_filesuffix

        if output_filepath is None:
            self.output_filepath = self.gdir.dir
        else:
            self.output_filepath = output_filepath

        # save time of initialisation, maybe reset later
        self.start_time = time.time()

        # define variables which are filled during the minimisation run
        self.costs = None
        self.c_terms = None
        self.reg_terms = None
        self.c_terms_description = None
        self.time_needed = None
        self.grads = None
        self.flowlines = None
        self.initial_flux = None
        self.sfc_h_start = None
        self.section_start = None
        self.observations_mdl = None
        self.end_time = None
        self.known_parameters = None
        self.len_unknown_parameter = None
        self.parameter_indices = None
        self.unknown_parameters = None
        self.fct_calls = np.array([0])
        self.step_indices = np.array([0])  # include zero to save outcome of first guess run
        self.minimize_message = None
        self.minimize_status = None
        self.memory_error = False
        self.extra_bed_h = None
        self.first_guess = None

        self.filename = gdir.name + '_' + \
            inversion_input['experiment_description']

        # create info Text for callback_fct TODO: think about showing the evolution of the c_terms
        self.info_text = '''

    Experiment: {experiment}
    Iteration: {iteration:d}
    Total Function calls: {fct_call:d}
    Needed Time: {time_needed:.1f} s
    Cost: ({cost:.3e}) = J_obs ({J_obs:.3e}) + lam ({lambda:.1e}) * J_reg ({J_reg:.3e})
    J_obs = mean({c_terms})
    J_reg = mean({reg_terms})
    '''

    def add_true_bed(self):
        raise NotImplementedError('Not added yet!')

    def save_data_in_datalogger(self, var, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().to('cpu').numpy().astype(np.float64)
        elif not isinstance(data, np.ndarray):
            data = np.array(data)

        current_var = getattr(self, var)

        if current_var is None:
            new_var = np.reshape(data, (-1, data.size))
        else:
            new_var = np.reshape(np.append(current_var, data), (-1, data.size))

        setattr(self, var, new_var)

    def callback_fct(self, x0):
        log.debug('In callback function of minimisation (Successful Iteration)')
        i = len(self.costs) - 1
        # make sure that there are some results to show (if no calculation was
        # perfomed there is nothing to show)
        if i >= 0:
            # save the current index for the later cleaning of the data
            self.step_indices = np.append(self.step_indices, i)
            # create c_terms and reg_terms string
            c_terms_str = []
            reg_terms_str = []
            for key in self.c_terms_description[-1][0].keys():
                if key in ['J_obs', 'J_reg']:
                    continue
                elif key in ['smoothed_bed', 'smoothed_flux']:
                    reg_terms_str.append(
                        f'{key}: {self.c_terms_description[-1][0][key]:.2e}')
                else:
                    # I assume it is an observation with years
                    for year in self.c_terms_description[-1][0][key]:
                        c_terms_str.append(
                            f'{key} ({year}): '
                            f'{self.c_terms_description[-1][0][key][year]:.2e}'
                        )
            # define the arguments for the shown text
            args = {'experiment': self.filename,
                    'iteration': len(self.step_indices) - 1,
                    'fct_call': self.fct_calls[-1],
                    'time_needed': float(self.time_needed[-1]),
                    'cost': self.costs[-1][0],
                    'J_obs': self.c_terms_description[-1][0]['J_obs'],
                    'J_reg': self.c_terms_description[-1][0]['J_reg'],
                    'lambda': self.cost_lambda,
                    'c_terms': c_terms_str,
                    'reg_terms': reg_terms_str
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
        self.reg_terms = self.reg_terms[index]
        self.time_needed = self.squeeze_generic(self.time_needed[index])
        self.flowlines = self.squeeze_generic(self.flowlines[index])
        self.initial_flux = self.squeeze_generic(self.initial_flux[index])
        self.sfc_h_start = self.squeeze_generic(self.sfc_h_start[index])
        self.section_start = self.squeeze_generic(self.section_start[index])
        self.observations_mdl = self.squeeze_generic(self.observations_mdl[index])
        self.unknown_parameters = self.unknown_parameters[index]
        self.fct_calls = self.squeeze_generic(self.fct_calls[index + 1])
        self.c_terms_description = self.squeeze_generic(self.c_terms_description[index])
        self.grads = self.squeeze_generic(self.grads[index])

    def create_and_save_dataset(self):
        ds = xr.Dataset()

        ds.coords['x'] = np.arange(len(self.ice_mask))
        ds.coords['iteration'] = np.arange(len(self.step_indices))
        ds.coords['nr_cost_terms'] = np.arange(len(self.c_terms[-1]))
        ds.coords['nr_reg_terms'] = np.arange(len(self.reg_terms[-1]))
        ds.coords['nr_unknown_parameters'] = np.arange(len(self.unknown_parameters[-1]))

        ds['ice_mask'] = (['x'], self.ice_mask)
        ds['costs'] = (['iteration'], self.costs)
        ds['grads'] = (['iteration', 'nr_unknown_parameters'], self.grads)
        ds['flowlines'] = (['iteration'], self.flowlines)
        ds['initial_flux'] = (['iteration', 'x'], self.initial_flux)
        ds['sfc_h_start'] = (['iteration', 'x'], self.sfc_h_start)
        ds['section_start'] = (['iteration', 'x'], self.section_start)
        ds['observations_mdl'] = (['iteration'], self.observations_mdl)
        ds['c_terms'] = (['iteration', 'nr_cost_terms'], self.c_terms)
        ds['reg_terms'] = (['iteration', 'nr_reg_terms'], self.reg_terms)
        ds['c_terms_description'] = (['iteration'], self.c_terms_description)
        ds['time_needed'] = (['iteration'], self.time_needed)
        ds['fct_calls'] = (['iteration'], self.fct_calls)
        ds['unknown_parameters'] = (['iteration', 'nr_unknown_parameters'],
                                    self.unknown_parameters)

        ds.attrs['experiment_description'] = self.filename
        ds.attrs['climate_filename'] = (self.climate_filename +
                                        self.climate_filesuffix)
        ds.attrs['output_filesuffix'] = self.output_filesuffix
        ds.attrs['flowline_init'] = self.flowline_init
        ds.attrs['obs_scaling_parameters'] = self.obs_scaling_parameters
        ds.attrs['regularisation_terms'] = self.regularisation_terms
        ds.attrs['cost_lambda'] = self.cost_lambda
        ds.attrs['control_vars'] = self.control_vars
        ds.attrs['observations'] = self.observations
        ds.attrs['observations_for_scaling'] = self.observations_for_scaling
        ds.attrs['mb_models_settings'] = self.mb_models_settings
        ds.attrs['min_w0_m'] = self.min_w0_m
        ds.attrs['min_ice_thickness'] = self.min_ice_thickness
        ds.attrs['max_ice_thickness'] = self.max_ice_thickness
        ds.attrs['max_deviation_surface_h'] = self.max_deviation_surface_h
        ds.attrs['limits_lambda'] = self.limits_lambda
        ds.attrs['spinup_options'] = self.spinup_options
        ds.attrs['solver'] = self.solver
        ds.attrs['minimize_options'] = self.minimize_options
        ds.attrs['max_time_minimize'] = self.max_time_minimize
        ds.attrs['device'] = self.device
        ds.attrs['parameter_indices'] = self.parameter_indices
        ds.attrs['minimize_message'] = self.minimize_message
        ds.attrs['minimize_status'] = self.minimize_status
        ds.attrs['memory_error'] = self.memory_error
        ds.attrs['rgi_id'] = self.gdir.rgi_id

        # check if filename already exists, prevent overwriting
        path = self.gdir.dir
        list_of_files = os.listdir(path)
        if (self.filename + '.pkl') in list_of_files:
            for file_nr in range(50):
                if (self.filename + '_' + str(file_nr) + '.pkl') \
                        not in list_of_files:
                    self.filename += ('_' + str(file_nr))
                    break
                if file_nr == 49:
                    raise ValueError('There are to many files with the same '
                                     'name!')

        # save dataset as pickle
        out = os.path.join(self.output_filepath, self.filename + '.pkl')
        with open(out, 'wb') as handle:
            # Attention pickle only is guaranteed to work if the same xarray
            # version is used for opening as is used for writing!
            pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # also save version of xarray
        out = os.path.join(self.output_filepath, self.filename + '_xr_version.txt')
        with open(out, 'w') as f:
            f.write('The xarray version is needed to open the pickle file '
                    'correctly \n')
            f.write(f'xarray = {xr.__version__}')

        # to open it use:
        # with open('filename.pkl', 'rb') as handle:
        #     b = pickle.load(handle)


def initialise_DataLogger(gdir, inversion_input_filesuffix='_agile',
                          init_model_filesuffix=None,
                          init_model_fls='_trapezoidal',
                          climate_filename='climate_historical',
                          climate_filesuffix='', output_filesuffix='_agile',
                          output_filepath=None):
    '''
    extract information out of gdir and save in datalogger. TODO

    '''

    if init_model_filesuffix is not None:
        fp = gdir.get_filepath('model_geometry',
                               filesuffix=init_model_filesuffix)
        fmod = FileModel(fp)
        init_model_fls = fmod.fls

    if type(init_model_fls) is str:
        fls_init = gdir.read_pickle('model_flowlines',
                                    filesuffix=init_model_fls)
    else:
        fls_init = copy.deepcopy(init_model_fls)

    if len(fls_init) > 1:
        raise NotImplementedError('agile only works with single flowlines!')

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

    # save observations for scaling if wanted
    inversion_input['observations_for_scaling'] = None
    for reg_term in inversion_input['regularisation_terms']:
        if (reg_term in ['fl_surface_h_scale_1', 'fl_surface_h_scale_2',
                         'bed_h_grad_scale']):
            if ('fl_surface_h:m' not in inversion_input['observations'].keys()
                    and (reg_term in ['fl_surface_h_scale_1',
                                      'fl_surface_h_scale_2'])):
                raise NotImplementedError('fl_surface_h must be observations '
                                          'if scaling should be used!')
            if all(k in inversion_input['regularisation_terms'] for k in
                   ['fl_surface_h_scale_1', 'fl_surface_h_scale_2']):
                raise NotImplementedError('only one of the two scaling for '
                                          'fl_surface_h can be used at a time!')
            inversion_input['observations_for_scaling'] = {
                'fl_widths:m': fls_init[0].widths_m}
    if 'area_bed_h' in inversion_input['control_vars']:
        inversion_input['observations_for_scaling'] = {
            'fl_widths:m': fls_init[0].widths_m}

    if all(k in inversion_input['control_vars'] for k in ['bed_h',
                                                          'area_bed_h']):
        raise NotImplementedError("It is not possible to define 'bed_h' and "
                                  "'area_bed_h' as control variables "
                                  "simultaneously!")

    data_logger = DataLogger(gdir, fls_init, inversion_input, climate_filename,
                             climate_filesuffix, output_filesuffix,
                             output_filepath)

    return data_logger
