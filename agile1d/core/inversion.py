# Built ins
import logging
import shutil
import warnings
import time
import copy

# External libs
from agile1d.core.dynamics import run_model_and_get_model_values
from agile1d.core.flowline import MixedBedFlowline
from scipy.optimize import minimize, Bounds
import numpy as np

# Locals
from oggm import utils, cfg, workflow, tasks, entity_task
from oggm.cfg import G
from agile1d.core.cost_function import create_cost_fct, initialise_mb_models, \
    initialise_flowline, do_height_shift_spinup
from agile1d.core.data_logging import initialise_DataLogger
from agile1d.core.exception import MaxCalculationTimeReached
from agile1d.core.first_guess import get_first_guess

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
    _doc = "Defines the control variables in an array which are changed during " \
           "the inversion. " \
           "Options: 'bed_h' (not good), 'lambdas', 'w0_m', 'area_bed_h'." \
           "Default: ['area_bed_h']"
    _default = ['area_bed_h']
    add_setting()

    _key = "control_vars_characteristic_scale"
    _doc = "Defines the control variables characteristic scale. This means " \
           "after the control variables are scaled the input to the " \
           "minimisation algorithm will be between [0, characteristic_scale]. " \
           "The idea is to avoid problems with the representation of small " \
           "numbers. " \
           "Default: 10."
    _default = 10.
    add_setting()

    _key = "mb_models_settings"
    _doc = "Defines the used MassBalanceModels in a dictionary. The key is the " \
           "name, the value is again a dictionary with keys 'type' (options: " \
           "'constant' or 'TIModel') and 'years' (the interval the model is valid in an " \
           "numpy array, if empty MB is valid for whole period). This interval " \
           "defines also the period of the model run. Caution first " \
           "MassBalanceModel must start at least " \
           "one year before first given observation year!" \
           "Default: {'MB': {'type': 'TIModel'," \
           "                 'years': np.array([1980, 2020])," \
           "                 'model_args': {}}}"
    _default = {'MB': {'type': 'TIModel',
                       'years': np.array([1980, 2020]),
                       'model_args': {}}}

    add_setting()

    _key = "dynamic_model"
    _doc = "Define which dynamic solver should be used. The options are " \
           "'flux_based', which is a reimplemented and simplified version " \
           "of OGGMs FluxBasedModel, or 'implicit', which is a implicit " \
           "solver only for a trapezoidal flowline. Default is 'implicit'. "
    _default = 'implicit'
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

    _key = "additional_ice_thickness"
    _doc = "The upper boundary of the ice thickness is calculated using the " \
           "approach from GlabTop (Linsbauer 2021) and this value is added " \
           "additionally for the definition of the upper boundary. " \
           "Default: 100, OUTDATED"
    _default = 100.
    add_setting()

    _key = "bed_h_bounds"
    _doc = "Define how large the boundaries for the bed_h are, in relation of " \
           "first guess thickness. (e.g. (0.4, 1.6) means the bed height can " \
           "be between 1.4*fg_thick and 0.2*fg_thick). " \
           "Default: (0.4, 1.6)"
    _default = (0.4, 1.6)
    add_setting()

    _key = "max_deviation_surface_h"
    _doc = "Maximum allowed deviation of surface_h in m, calculated from " \
           "surface height of initial flowline. " \
           "Default: 1000"
    _default = 1000.
    add_setting()

    _key = "limits_lambda"
    _doc = "Allowed limits for lambda (the angle of trapezoidal side wall)," \
           "0.1 -> ~87°, 2 -> 45°. " \
           "Default: (0.1, 4.)"
    _default = (0.1, 4.)
    add_setting()

    _key = "limits_height_shift_spinup"
    _doc = "Allowed limits for height_shift_spinup if used. " \
           "Default: (-1000., 1000.)"
    _default = (-1000., 1000.)
    add_setting()

    _key = "observations"
    _doc = "Gives the observations which should be matched during the " \
           "minimisation. Organised in a dictionary with keys giving  " \
           "'measurement:unit' (e.g. 'area:m2' or 'area:km2'), value again a " \
           "dict with keys the year (e.g. '2003') or period (e.g.'2000-2010') " \
           "and the measured 'values' (the actual Measurement). Example for " \
           "one entry: {'area:km2': {'2003': 12, '2010': 8}}" \
           "Measurement options: " \
           "'fl_surface_h' (unit: 'm') the surface heights at each grid point " \
           "with ice along the initial flowline (extracted from initial " \
           "flowline, if no time given RGI date);" \
           "'fl_widths' (unit: 'm') the surface widths at each grid point with " \
           "ice along the initial flowline (extracted from initial flowline, " \
           "if no time given RGI date);" \
           "'fl_total_area' (unit: 'm2', 'km2') the total area of the given " \
           "initial flowline (extracted from initial flowline, if no time " \
           "given RGI date); " \
           "'area' (unit: 'm2', 'km2') an additional total area measurement; " \
           "'dmdtda' (unit: 'kg m-2 yr-1', 'kg yr-1') geodetic measurement, if " \
           "unit is 'kg m-2 yr-1' the mean area of the start year and end " \
           "year of the given period is used; if unit is 'kg yr-1' it is " \
           "assumed a fixed area was used during the creation of the " \
           "geodetic mass-balance, in this case it is assumed the provided " \
           "'dmdtda' was already multiplied by the fixed area (e.g. " \
           "provided_dmdtda = original_dmdtda * RGI_area); the time " \
           "is given as period (e.g. 'dmdtda:kg m-2 yr-1': {'2009-2015': 20})." \
           "'us' (unit: myr-1) surface ice velocity (if no time given RGI date)" \
           "Default: {'fl_surface_h:m': {}, " \
           "'dmdtda:kg m-2 yr-1': {}"
    _default = {'fl_surface_h:m': {},
                'dmdtda:kg m-2 yr-1': {}}
    add_setting()

    _key = "obs_scaling_parameters"
    _doc = "Defines the relative contribution of the observations to the " \
           "total cost value. There are two options 'uncertainty' and 'scale'." \
           " If 'uncertainty' is given the reg parameter of an observations " \
           "equals '1 / uncertainty²'. The uncertainty " \
           "is prescribed in cost_function define_reg_parameters. The " \
           "uncertainty can be given as 'absolute' values or 'relative' to " \
           "the observed quantity. Or (NOT " \
           "RECOMMENDED) one can use a dict with key 'scale' (e.g. " \
           "{'scale': {'fl_surface_h': 10., 'fl_widths_m': 1.}}), this option " \
           "first express the individual mismatches in percent of the " \
           "observation and afterwards multiple with the given numbers (from " \
           "example " \
           "above this means a mismatch of 1% at 'fl_surface_h' is equally " \
           "weighted as a 10% mismatch at 'fl_widths_m'). " \
           "Default: {'uncertainty': {'fl_surface_h:m': 1.," \
           "'dmdtda:kg m-2 yr-1': 1.}}"
    _default = {'uncertainty': {'fl_surface_h:m': {'absolute': 10.},
                                'dmdtda:kg m-2 yr-1': {'absolute': 100.}}}
    add_setting()

    _key = "regularisation_terms"
    _doc = "Defines which additional regularisation terms should be used " \
           "(with their reg parameter). Scaling of " \
           "regularisation terms is also included during the creation." \
           "Options: " \
           "'smoothed_bed' adds a term which prefer smoother beds. " \
           "(NOT GOOD) 'fl_surface_h_scale_1' scale fl_surface_h in cost function with 1/wk. " \
           "(NOT GOOD) 'fl_surface_h_scale_2' scale fl_surface_h in cost function with " \
           "1/sum(wk) ()² * wk." \
           "'bed_h_grad_scale' scales the bed_h gradient with width." \
           "'smoothed_flux' prefer smoother initial flux." \
           "'distance_from_fg' use distance from fg to make some control " \
           "variables more likely to change use with {'control': factor, " \
           "'control1: factor2}. NOT SCALED PROBABLY AT THE MOMENT! " \
           "Default: {'smoothed_bed': 1}"
    _default = {'smoothed_bed': 1.}
    add_setting()

    _key = "cost_lambda"
    _doc = "Defines tha labmda of the cost function" \
           "(cost = Jobs + lambda * Jreg)" \
           "Default: 1"
    _default = 1.
    add_setting()

    _key = "experiment_description"
    _doc = "Explanation of the current experiment. If a dataset is saved this" \
           "is the filename of the resulting nc datafile. " \
           "Default: 'agile_inversion_results'"
    _default = 'agile_inversion_results'
    add_setting()

    _key = "spinup_options"
    _doc = "Options how to initialise each minimisation run. First Option is " \
           "'surface_h', which just let the minimisatio algorithm " \
           "change the initial surface_h (meaning the run starts probably with " \
           "an ice surface not in an dynamic consistant state!), you also must " \
           "provide options how the first guess surface_h is created " \
           "(e.g. {'surface_h': {'mb_model': {'type': 'constant'," \
           "'years': np.array([1980, 2000]), 't_bias': -2}}}). " \
           "Second Option is to include a short spinup with variable " \
           "height shift of the whole mb profile for an adaptive spinup " \
           "(e.g. {'height_shift': {'mb_model': " \
           "{'type': 'constant', 'years': np.array([1980, 2000]), " \
           "'fg_height_shift': -100," \
           "'spinup_length_yrs': 30}}}). " \
           "Third Option is more for idealized experiments. It means you " \
           "provide perfectly known initial surface height or initial ice " \
           "thickness or initial section (volume). You must provide the" \
           "filesuffix of the true initial flowline." \
           "e.g. {'perfect_sfc_h/perfect_thickness/perfect_section': " \
           "'filesuffix'}." \
           "Fourth Option is to use the initial cross section area as a " \
           "control variable (with constant dx also can be considered as " \
           " initial volume distribution), it is similar to 'surface_h' option " \
           "but also takes grid cells of different sizes into account. It " \
           "should be agiled with a regularisation term which smoothes the " \
           "initial ice flux. With 'extra_grid_points' you can define how " \
           "many grid points you want to allow an advance/retreat of the " \
           "glacier at the inital state." \
           "e.g. {'section':" \
           "         {'extra_grid_points': 10," \
           "          'limits': (0.6, 1.4)," \
           "          'fg_years': 1" \
           "          }" \
           "      }" \
           "Option 'perfect_bed_h' could be used for idealized experiments. " \
           "e.g. {'perfect_bed_h': filesuffix}" \
           "Default: {'height_shift':" \
           "    {'mb_model': {'type': 'constant'," \
           "                  'years': np.array([1980, 2000]), " \
           "                  'fg_height_shift': -100}," \
           "     'spinup_length_yrs': 20}}"
    _default = {'height_shift': {'mb_model': {'type': 'constant',
                                              'years': np.array([1980, 2000]),
                                              'fg_height_shift': -100},
                                 'spinup_length_yrs': 20}}
    add_setting()

    _key = "minimize_options"
    _doc = "The options for scipy.optimize.minimize. For options check " \
           "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html. " \
           "Default: {'maxiter': 10, 'ftol': 1e-7, 'gtol': 1e-8, 'disp': True," \
           "'maxcor': 50, 'maxls': 50}"
    _default = {'maxiter': 10, 'ftol': 1e-7, 'gtol': 1e-8, 'disp': True,
                'maxcor': 50, 'maxls': 50}
    add_setting()

    _key = "solver"
    _doc = "Solver for scipy.optimize.minimize. Default: 'L-BFGS-B'"
    _default = 'L-BFGS-B'
    add_setting()

    _key = "max_time_minimize"
    _doc = "Defines a maximum time for the minimization after it is terminated " \
           "in s. Default: None"
    _default = None
    add_setting()

    _key = "torch_type"
    _doc = "The used datatype for the torch tensors. " \
           "Options: 'float', 'double'. Default: 'double'"
    _default = 'double'
    add_setting()

    _key = "device"
    _doc = "The device of the computation, if 'cuda' it is checked if a GPU is " \
           "available, otherwise 'cpu' is selected. " \
           "Options: 'cpu', 'cuda'. Default: 'cuda'."
    _default = 'cpu'
    add_setting()

    if get_doc:
        return inversion_settings, inversion_settings_doc

    return inversion_settings


@entity_task(log, writes=['inversion_input'])
def prepare_for_agile_inversion(gdir, inversion_settings=None,
                                filesuffix='_agile'):
    """TODO
    """
    if inversion_settings is None:
        inversion_settings = get_default_inversion_settings(get_doc=False)

    # just writes the inversion settings to a file
    gdir.write_pickle(inversion_settings, filename='inversion_input',
                      filesuffix=filesuffix)


def get_control_var_bounds(data_logger):
    bounds = np.zeros(data_logger.len_unknown_parameter, dtype='object')

    for var in data_logger.parameter_indices.keys():
        var_indices = data_logger.parameter_indices[var]
        if var == 'bed_h':
            fl = data_logger.flowline_init
            ice_mask = data_logger.ice_mask
            bed_h_bounds = data_logger.bed_h_bounds

            h_max_w0 = (fl.widths_m[fl.thick > 0] - data_logger.min_w0_m) / \
                       fl._lambdas[fl.thick > 0]

            bounds[var_indices] = [(sfc_h - min(thick * bed_h_bounds[1],
                                                thick_max),
                                    sfc_h - thick * bed_h_bounds[0]
                                    )
                                   for sfc_h, thick, thick_max in
                                   zip(fl.surface_h[ice_mask],
                                       fl.thick[ice_mask],
                                       h_max_w0)]

            # Outdated calculation using GlabTop
            # upper_limits = get_adaptive_upper_ice_thickness_limit(
            #     fl,
            #     additional_ice_thickness=data_logger.additional_ice_thickness,
            #     max_thickness=data_logger.max_ice_thickness,
            #     w0_min=data_logger.min_w0_m)
            # bounds[var_indices] = [(sfc_h - upper_limit,
            #                         sfc_h - data_logger.min_ice_thickness)
            #                        for sfc_h, upper_limit in
            #                        zip(fl.surface_h[ice_mask],
            #                            upper_limits)]

        elif var == 'area_bed_h':
            fl = data_logger.flowline_init
            ice_mask = data_logger.ice_mask
            bed_h_bounds = data_logger.bed_h_bounds

            h_max_w0 = (fl.widths_m[fl.thick > 0] - data_logger.min_w0_m) / \
                       fl._lambdas[fl.thick > 0]

            bounds[var_indices] = [((sfc_h - min(thick * bed_h_bounds[1],
                                                 thick_max)) * width_m,
                                    (sfc_h - thick * bed_h_bounds[0]) * width_m
                                    )
                                   for sfc_h, thick, width_m, thick_max in
                                   zip(fl.surface_h[ice_mask],
                                       fl.thick[ice_mask],
                                       fl.widths_m[ice_mask],
                                       h_max_w0)]

            # Outdated calculation using GlabTop
            # upper_limits = get_adaptive_upper_ice_thickness_limit(
            #     fl,
            #     additional_ice_thickness=data_logger.additional_ice_thickness,
            #     max_thickness=data_logger.max_ice_thickness,
            #     w0_min=data_logger.min_w0_m)
            # bounds[var_indices] = [((sfc_h - upper_limit) * width_m,
            #                         (sfc_h - data_logger.min_ice_thickness) *
            #                         width_m)
            #                        for sfc_h, width_m, upper_limit in
            #                        zip(fl.surface_h[ice_mask],
            #                            fl.widths_m[ice_mask],
            #                            upper_limits)
            #                        ]

        elif var == 'surface_h':
            fl = data_logger.flowline_init
            bounds[var_indices] = [(sfc_h - data_logger.max_deviation_surface_h,
                                    sfc_h + data_logger.max_deviation_surface_h)
                                   for sfc_h in fl.surface_h]
        elif var == 'section':
            fl = data_logger.flowline_init
            terminus_grid_points = \
                data_logger.spinup_options['section']['extra_grid_points']
            terminus_index = sum(fl.thick > 0)
            limits = data_logger.spinup_options['section']['limits']
            # first define limits without terminus
            bounds_tmp = [(limits[0] * section, limits[1] * section)
                          for section in
                          fl.section[:terminus_index - terminus_grid_points]]

            # finally add terminus limits, with always 0 lower bound
            upper_terminus_bound = np.max(
                fl.section[terminus_index - terminus_grid_points:
                           terminus_index + terminus_grid_points]) * limits[1]
            for i in range(2*terminus_grid_points):
                bounds_tmp.append((0, upper_terminus_bound))
            bounds[var_indices] = bounds_tmp

        elif var == 'lambdas':
            bounds[var_indices] = [data_logger.limits_lambda]
        elif var == 'w0_m':
            fl = data_logger.flowline_init
            is_trapezoid = data_logger.is_trapezoid
            ice_mask = data_logger.ice_mask
            index_use = (is_trapezoid & ice_mask)
            bounds[var_indices] = [(data_logger.min_w0_m, max_w0_m)
                                   for max_w0_m in fl.widths_m[index_use]]
        elif var == 'height_shift_spinup':
            bounds[var_indices] = [data_logger.limits_height_shift_spinup]
        else:
            raise NotImplementedError(f'{var}')

    # need the bounds for min/max scaling of control variables
    data_logger.bounds = bounds

    return bounds


def get_adaptive_upper_ice_thickness_limit(fl, additional_ice_thickness=100,
                                           min_thickness=1, max_thickness=1000,
                                           f=0.8, w0_min=10):
    """
    Function to get an estimate for the maximum ice thickness using the
    approach from GlabTop (Linsbauer 2012) and adding an additional_thickness
    to this estimate.

    Parameters
    ----------
    fl
    additional_ice_thickness
    min_thickness
    max_thickness
    f
    w0_min

    Returns
    -------

    """
    ice_surface_h = fl.surface_h[fl.thick > 0]
    delta_height = ice_surface_h[0] - ice_surface_h[-1]

    if delta_height > 1600:
        tau = 150000  # Pa
    else:
        # equation calculates tau in bar -> convert to Pa with 1e5
        tau = (0.005 + 1.598 * delta_height / 1000 - 0.435 *
               (delta_height / 1000)**2) * 1e5  # Pa

    slope_stag = np.zeros(len(ice_surface_h) + 1)
    slope_stag[0] = 1
    slope_stag[1:-1] = (ice_surface_h[0:-1] - ice_surface_h[1:]) / fl.dx_meter
    slope_stag[-1] = slope_stag[-2]

    rho_ice = cfg.PARAMS['ice_density']

    h_shear_stress = tau / (f * rho_ice * G * np.abs(slope_stag))
    upper_h_limit = h_shear_stress[1:] + additional_ice_thickness
    upper_h_limit = np.clip(upper_h_limit, min_thickness, max_thickness)

    # also take here into account maximum thickness so that w0 > 0 always
    h_max_w0 = (fl.widths_m[fl.thick > 0] - w0_min) / fl._lambdas[fl.thick > 0]
    upper_h_limit = np.minimum(h_max_w0, upper_h_limit)

    return upper_h_limit


def save_past_evolution_to_disk(gdir, data_logger):
    # recreate initial flowline
    fl_final = copy.deepcopy(data_logger.flowlines[-1])
    # set to initial ice thickness distribution, this already includes a
    # potential spinup
    fl_final.surface_h = copy.deepcopy(data_logger.sfc_h_start[-1])
    # convert flowline to torch flowline
    fl_final = MixedBedFlowline(
        line=fl_final.line, dx=fl_final.dx, map_dx=fl_final.map_dx,
        surface_h=fl_final.surface_h, bed_h=fl_final.bed_h,
        section=fl_final.section, bed_shape=fl_final.bed_shape,
        is_trapezoid=fl_final.is_trapezoid, lambdas=fl_final._lambdas,
        w0_m=fl_final._w0_m, rgi_id=fl_final.rgi_id,
        water_level=fl_final.water_level, torch_type=data_logger.torch_type,
        device=data_logger.device)

    run_model_and_get_model_values(flowline=fl_final,
                                   dynamic_model=data_logger.dynamic_model,
                                   mb_models=initialise_mb_models(
                                       unknown_parameters=None,
                                       data_logger=data_logger),
                                   needed_model_data={}, save_run=True,
                                   gdir=gdir,
                                   output_filesuffix=data_logger.filename,
                                   force=False)


@entity_task(log, writes=['model_flowlines'])
def agile_inversion(gdir, inversion_input_filesuffix='_agile',
                    init_model_filesuffix=None, init_model_fls='_trapezoidal',
                    climate_filename='climate_historical',
                    climate_filesuffix='', output_filesuffix='_agile_output',
                    output_filepath=None, save_dataset=True,
                    give_data_logger_back=False, save_past_evolution=True):
    """TODO
    """
    log.debug('initialise Datalogger')
    data_logger = initialise_DataLogger(gdir,
                                        inversion_input_filesuffix=inversion_input_filesuffix,
                                        init_model_filesuffix=init_model_filesuffix,
                                        init_model_fls=init_model_fls,
                                        climate_filename=climate_filename,
                                        climate_filesuffix=climate_filesuffix,
                                        output_filesuffix=output_filesuffix,
                                        output_filepath=output_filepath)
    log.debug('initialise cost function')
    cost_fct = create_cost_fct(data_logger)

    log.debug('initialise control var bounds')
    get_control_var_bounds(data_logger)

    log.debug('initialise first guess')
    first_guess = get_first_guess(data_logger)

    # continue here
    try:
        res = minimize(fun=cost_fct,
                       x0=first_guess,
                       method=data_logger.solver,
                       jac=True,
                       bounds=Bounds(0,
                                     data_logger.control_vars_characteristic_scale),
                       options=data_logger.minimize_options,
                       callback=data_logger.callback_fct)
        data_logger.minimize_message = res.message
        data_logger.minimize_status = res.status
    except MaxCalculationTimeReached:
        data_logger.minimize_message = 'Maximum calculation time reached!'
        data_logger.minimize_status = 'max calc time reached'

    log.debug('Minimisation finished')
    # filter out data used by minimize function for exploratory
    data_logger.filter_data_from_optimization()

    data_logger.end_time = time.time()

    if save_dataset:
        log.debug('Save Dataset')
        # save results to file
        data_logger.create_and_save_dataset()

    if save_past_evolution:
        log.debug('Save past evolution')
        # only save if no MemoryError occured
        if not data_logger.memory_error:
            save_past_evolution_to_disk(gdir=gdir, data_logger=data_logger)

            # move the file to the output_folder if it differs from gdir.dir
            if output_filepath is not None:
                shutil.copy(gdir.get_filepath('model_diagnostics',
                                              filesuffix=data_logger.filename),
                            data_logger.output_filepath)
        else:
            log.debug('Could not save past evolution due to a MemoryError!')

    gdir.write_pickle(data_logger.flowlines[-1], 'model_flowlines',
                      filesuffix=output_filesuffix)
    log.debug('agile Inversion finished')

    if give_data_logger_back:
        return data_logger
    else:
        # delete the data_logger (hopefully saved before with save_dataset=True)
        # preventing memory overflow
        del data_logger
        return None
