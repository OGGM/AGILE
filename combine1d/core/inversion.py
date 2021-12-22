# Built ins
import logging
import warnings
import time

# External libs
from scipy.optimize import minimize
import numpy as np

# Locals
from oggm import utils, cfg
from oggm import entity_task
from combine1d.core.cost_function import create_cost_fct
from combine1d.core.data_logging import initialise_DataLogger
from combine1d.core.exception import MaxCalculationTimeReached
from combine1d.core.first_guess import get_first_guess

# Module logger
log = logging.getLogger(__name__)


def get_default_inversion_settings(get_doc=False):

    inversion_settings = dict()
    inversion_settings_doc = dict()

    def add_setting():
        inversion_settings[_key] = _default
        inversion_settings_doc[_key] = _doc

    # define the elements of the inversion settings with default and doc
    _key = "control_vars"
    _doc = "Defines the control variables in an array which are changed during the inversion. " \
           "Options: 'bed_h', 'surface_h', 'lambdas', 'w0_m'. Default: ['bed_h']"
    _default = ['bed_h']
    add_setting()

    _key = "mb_models_settings"
    _doc = "Defines the used MassBalanceModels in a dictionary. The key is the name, the value " \
           "is again a dictionary with keys 'type' (options: 'constant') and 'years' (the " \
           "interval the model is valid in an numpy array). Caution first MassBalanceModel must " \
           "start at least one year before first given observation year!" \
           "Default: {'MB1': {'type': 'constant', 'years': np.array([1999, 2005])}," \
           "'MB2': {'type': 'constant', 'years': np.array([2005, 2010])}}"
    _default = {'MB1': {'type': 'constant', 'years': np.array([1999, 2005])},
                'MB2': {'type': 'constant', 'years': np.array([2005, 2010])}}
    add_setting()

    _key = "min_w0_m"
    _doc = "Minimum allowed bottom width for trapezoidal bed shapes in m. Default: 10."
    _default = 10.
    add_setting()

    _key = "min_ice_thickness"
    _doc = "Minimum allowed ice thickness in m, calculated from surface height" \
           "of initial flowline. " \
           "Default: 0.1"
    _default = 0.1
    add_setting()

    _key = "max_ice_thickness"
    _doc = "Maximum allowed ice thickness in m, calculated from surface height" \
           "of initial flowline. " \
           "Default: 1000"
    _default = 1000.
    add_setting()

    _key = "max_deviation_surface_h"
    _doc = "Maximum allowed deviation of surface_h in m, calculated from " \
           "surface height of initial flowline. " \
           "Default: 1000"
    _default = 1000.
    add_setting()

    _key = "limits_lambda"
    _doc = "Allowed limits for lambda (the angle of trapezoidal side wall)," \
           "0 -> 90°, 2 -> 45°. " \
           "Default: (0., 4.)"
    _default = (0., 4.)
    add_setting()

    _key = "observations"
    _doc = "Gives the observations which should be matched during the minimisation. Organised " \
           "in a dictionary with keys giving  'measurement:unit' (e.g. 'area:m2' or 'area:km2')," \
           " value again a " \
           "dict with keys " \
           "the year (e.g. '2003') or period (e.g.'2000-2010') and the measured " \
           "'values' " \
           "(the actual Measurement). Example for one entry: {'area:km2': {'2003': 12, " \
           "'2010': 8}}" \
           "Measurement options: " \
           "'fl_surface_h' (unit: 'm') the surface heights at each grid point with ice along " \
           "the initial flowline (extracted from initial flowline, if no time given RGI date);" \
           "'fl_widths' (unit: 'm') the surface widths at each grid point with ice along " \
           "the initial flowline (extracted from initial flowline, if no time given RGI date);" \
           "'fl_total_area' (unit: 'm2', 'km2') the total area of the given initial flowline " \
           "(extracted from initial flowline, if no time given RGI date); " \
           "'area' (unit: 'm2', 'km2') an additional total area measurement; " \
           "'dh' (unit: 'm') geodetic measurement ('time' is given as period, e.g. '2009-2015')." \
           "Default: {'fl_surface_h:m': {}, " \
           "'fl_widths:m': {}"
    _default = {'fl_surface_h:m': {},
                'fl_widths:m': {}}
    add_setting()

    _key = "obs_reg_parameters"
    _doc = "Defines the relative contribution to of the observations to the total cost value. " \
           "Could be given directly by using a dict with the same keys as 'observations'  " \
           "(e.g. {'fl_surface_h': 10., 'fl_widths_m': 1.}). Or one can use a dict with key " \
           "'scale' (e.g. {'scale': {'fl_surface_h': 10., 'fl_widths_m': 1.}}), this option " \
           "first express the individual mismatches in percent of the observation and then " \
           "multiple with the given numbers (from example above this means a mismatch of 1% " \
           "to 'fl_surface_h' is equally weighted as a 10% mismatch to 'fl_widths_m'). " \
           "Default: {'scale': {'fl_surface_h': 1., 'fl_widths_m': 1.}}"
    _default = {'scale': {'fl_surface_h:m': 1., 'fl_widths:m': 1.}}
    add_setting()

    _key = "regularisation_terms"
    _doc = "Defines which additional regularisation terms should be used (with their reg " \
           "parameter). Options: " \
           "'smoothed_bed' adds a term which prefer smoother beds. " \
           "Default: {'smoothed_bed': 1}"
    _default = {'smoothed_bed': 1.}
    add_setting()

    _key = "experiment_description"
    _doc = "Explanation of the current experiment. If a dataset is saved this" \
           "is the filename of the resulting nc datafile. " \
           "Default: 'COMBINE_inversion_results'"
    _default = 'COMBINE_inversion_results'
    add_setting()

    _key = "spinup_options"
    _doc = "Not implemented yet!"
    _default = None
    add_setting()

    _key = "minimize_options"
    _doc = "The options for scipy.optimize.minimize. For options check " \
           "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html. " \
           "Default: {'maxiter': 10, 'ftol': 1e-7, 'gtol': 1e-8, 'disp': True, 'maxcor': 50, " \
           "'maxls': 50}"
    _default = {'maxiter': 10, 'ftol': 1e-7, 'gtol': 1e-8, 'disp': True, 'maxcor': 50, 'maxls': 50}
    add_setting()

    _key = "solver"
    _doc = "Solver for scipy.optimize.minimize. Default: 'L-BFGS-B'"
    _default = 'L-BFGS-B'
    add_setting()

    _key = "max_time_minimize"
    _doc = "Defines a maximum time for the minimization after it is terminated in s. Default: None"
    _default = None
    add_setting()

    _key = "torch_type"
    _doc = "The used datatype for the torch tensors. Options: 'float', 'double. Default: 'double'"
    _default = 'double'
    add_setting()

    _key = "device"
    _doc = "The device of the computation, if 'cuda' it is checked if a GPU is available, " \
           "otherwise 'cpu' is selected. Options: 'cpu', 'cuda'. Default: 'cuda'."
    _default = 'cuda'
    add_setting()

    if get_doc:
        return inversion_settings, inversion_settings_doc

    return inversion_settings


@entity_task(log, writes=['inversion_input'])
def prepare_for_combine_inversion(gdir, inversion_settings=None, filesuffix='_combine'):
    """TODO
    """
    if inversion_settings is None:
        inversion_settings = get_default_inversion_settings(get_doc=False)

    # just writes the inversion settings to a file
    gdir.write_pickle(inversion_settings, filename='inversion_input', filesuffix=filesuffix)


def get_control_var_bounds(data_logger):
    bounds = np.zeros(data_logger.len_unknown_parameter, dtype='object')

    for var in data_logger.parameter_indices.keys():
        var_indices = data_logger.parameter_indices[var]
        if var == 'bed_h':
            fl = data_logger.flowline_init
            ice_mask = data_logger.ice_mask
            bounds[var_indices] = [(sfc_h - data_logger.max_ice_thickness,
                                    sfc_h + data_logger.min_ice_thickness)
                                   for sfc_h in fl.surface_h[ice_mask]]
        elif var == 'surface_h':
            fl = data_logger.flowline_init
            ice_mask = data_logger.ice_mask
            bounds[var_indices] = [(sfc_h - data_logger.max_deviation_surface_h,
                                    sfc_h + data_logger.max_deviation_surface_h)
                                   for sfc_h in fl.surface_h[ice_mask]]
        elif var == 'lambdas':
            bounds[var_indices] = [data_logger.limits_lambda]
        elif var == 'w0_m':
            fl = data_logger.flowline_init
            is_trapezoid = data_logger.is_trapezoid
            bounds[var_indices] = [(data_logger.min_w0_m, max_w0_m)
                                   for max_w0_m in fl.widths_m[is_trapezoid]]
        else:
            raise NotImplementedError(f'{var}')

    return bounds


@entity_task(log, writes=['model_flowlines'])
def combine_inversion(gdir, inversion_input_filesuffix='_combine', init_model_filesuffix=None,
                      init_model_fls=None, climate_filename='climate_historical',
                      climate_filesuffix='', output_filesuffix='_combine',
                      write_nc_dataset=True):
    """TODO
    """

    data_logger = initialise_DataLogger(gdir,
                                        inversion_input_filesuffix=inversion_input_filesuffix,
                                        init_model_filesuffix=init_model_filesuffix,
                                        init_model_fls=init_model_fls,
                                        climate_filename=climate_filename,
                                        climate_filesuffix=climate_filesuffix,
                                        output_filesuffix=output_filesuffix)

    cost_fct = create_cost_fct(data_logger)

    first_guess = get_first_guess(data_logger)

    bounds = get_control_var_bounds(data_logger)

    # continue here
    try:
        res = minimize(fun=cost_fct,
                       x0=first_guess,
                       method=data_logger.solver,
                       jac=True,
                       bounds=bounds,
                       options=data_logger.minimize_options,
                       callback=data_logger.callback_fct)
        data_logger.minimize_message = res.message
        data_logger.minimize_status = res.status
    except MaxCalculationTimeReached:
        data_logger.minimize_message = 'Maximum calculation time reached!'
        data_logger.minimize_status = 'max calc time reached'

    # filter out data used by minimize function for exploratory
    data_logger.filter_data_from_optimization()

    data_logger.end_time = time.time()

    if write_nc_dataset:
        # save results to netcdf file
        data_logger.create_and_save_dataset()

    gdir.write_pickle(data_logger.flowlines[-1], 'model_flowlines',
                      filesuffix=output_filesuffix)

    return data_logger
