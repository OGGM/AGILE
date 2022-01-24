import copy
import numpy as np
import torch
import time

from combine1d.core.dynamics import run_model_and_get_temporal_model_data
from combine1d.core.massbalance import ConstantMassBalanceTorch
from combine1d.core.flowline import MixedBedFlowline, FluxBasedModel


def create_cost_fct(data_logger):
    '''
    Creating a cost function for the given InversionGlacierDirectory.

    Returns
    -------
    Function which takes one input variable (unknown_parameter) and return two
    values (cost, grad)
    '''

    known_parameters = get_known_parameters(data_logger)
    data_logger.known_parameters = known_parameters

    # here create now the indices for all control variables of the unknown_parameters variable
    parameter_indices = get_indices_for_unknown_parameters(data_logger)
    data_logger.parameter_indices = parameter_indices

    def c_fun(unknown_parameters):
        return cost_fct(unknown_parameters,
                        data_logger)

    return c_fun


def get_known_parameters(data_logger):
    """TODO"""
    control_vars = data_logger.control_vars
    fl = data_logger.flowline_init
    ice_mask = data_logger.ice_mask
    is_rectangular = data_logger.is_rectangular
    is_parabolic = data_logger.is_parabolic
    is_trapezoid = data_logger.is_trapezoid

    known_parameters = dict()

    # save guard if new control vars are added
    potential_control_vars = ['bed_h', 'surface_h', 'lambdas', 'w0_m',
                              'height_shift_spinup']

    # extract the ice free parts of variables of the control_vars which are
    # assumed to be known
    for con_var in control_vars:
        if con_var not in potential_control_vars:
            raise NotImplementedError(f'Control var {con_var} not implemented!')

        if con_var in ['lambdas', 'w0_m']:
            prefix_var = '_'
            known_index = (is_rectangular | is_parabolic |
                           (is_trapezoid & ~ice_mask))
        elif con_var in ['bed_h']:
            prefix_var = ''
            known_index = ~ice_mask
        elif con_var in ['surface_h']:
            prefix_var = ''
            known_index = np.full(ice_mask.shape, False)
        elif con_var in ['height_shift_spinup']:
            # no known variable to save
            continue
        else:
            raise NotImplementedError(f'{con_var}')

        known_parameters[con_var] = getattr(fl, prefix_var + con_var)[known_index]

    return known_parameters


def get_indices_for_unknown_parameters(data_logger):
    """TODO"""
    parameter_indices = dict()
    current_start_ind = 0
    ice_grid_points = sum(data_logger.ice_mask)
    trapezoid_grid_points = sum(data_logger.is_trapezoid & data_logger.ice_mask)
    grid_points = len(data_logger.ice_mask)

    for control_var in data_logger.control_vars:
        # for these variables the length is assumed to be the whole ice area
        if control_var in ['bed_h']:
            parameter_length = ice_grid_points
        elif control_var in ['surface_h']:
            parameter_length = grid_points
        elif control_var in ['w0_m', 'lambdas']:
            parameter_length = trapezoid_grid_points
        elif control_var in ['height_shift_spinup']:
            parameter_length = 1
        else:
            raise NotImplementedError(f'Control var {control_var} is not '
                                      f'implemented!')

        parameter_indices[control_var] = np.arange(current_start_ind,
                                                   current_start_ind + parameter_length)
        current_start_ind += parameter_length

    # save the number (length) of the resulting unknown_parameter array
    data_logger.len_unknown_parameter = current_start_ind

    return parameter_indices


def cost_fct(unknown_parameters, data_logger):
    '''
    Calculates cost and gradient for the given parameters. At the moment only
    Trapezoidal optimisation.

    Parameters
    ----------
    unknown_parameters : :py:class:`numpy.ndarray`
        Unknown parameter values for the optimisation variable (ice part of
        the glacier). For simultaneously optimisation of two variables contains
        both unknown variables consecutively.
    data_logger : :py:class:`combine.core.data_logging.DataLogger`
        Keep track of all variables during the calculation.

    Returns
    -------
    :py:class:`numpy.ndarray`
        The actual value of the cost function.
    :py:class:`numpy.ndarray`
        The gradient for each unknown_parameter with respect to the cost value.

    '''

    flowline, fl_control_vars = initialise_flowline(unknown_parameters,
                                                    data_logger)

    mb_models, mb_control_vars = initialise_mb_models(unknown_parameters,
                                                      data_logger)

    if data_logger.spinup_type == 'height_shift_spinup':
        try:
            # Here a spinup run is conducted using the control variable
            # height_shift_spinup (vertically shift the whole mb profile)
            flowline, spinup_control_vars = \
                do_height_shift_spinup(flowline,
                                       unknown_parameters,
                                       data_logger)
        except MemoryError:
            msg = 'MemoryError during spinup run (due to a too small ' \
                  'timestep) -> set Costfunction to Inf'
            print(msg)
            data_logger.memory_error = msg
            cost = np.Inf
            grad = np.empty(len(unknown_parameters)) * np.nan
            return cost, grad
    elif data_logger.spinup_type not in [None, 'surface_h']:
        raise NotImplementedError(f'The spinup type {data_logger.spinup_type} '
                                  'possibility is not integrated!')
    else:
        spinup_control_vars = {}

    observations = data_logger.observations

    # forward run of model, try is needed to avoid a memory overflow
    try:
        observations_mdl, final_fl = run_model_and_get_temporal_model_data(
            flowline=flowline,
            mb_models=mb_models,
            observations=observations)

    except MemoryError:
        msg = 'MemoryError in forward model run (due to a too small ' \
              'timestep) -> set Costfunction to Inf'
        print(msg)
        data_logger.memory_error = msg
        cost = np.Inf
        grad = np.empty(len(unknown_parameters)) * np.nan
        return cost, grad

    # calculate terms of cost function
    c_terms = get_cost_terms(observations_mdl,
                             final_fl,  # for regularisation term 'smoothed_bed'
                             data_logger  # for reg_parameters
                             )

    # sum up cost function terms
    c = c_terms.sum()

    # calculate the gradient for the control variables
    c.backward()

    # convert cost to numpy array
    cost = c.detach().to('cpu').numpy().astype(np.float64)

    # get gradient/s as numpy array
    grad = get_gradients(fl_control_vars,
                         mb_control_vars,
                         spinup_control_vars,
                         data_logger,
                         length=len(unknown_parameters))

    # save data in data_logger
    data_logger.save_data_in_datalogger('flowlines', final_fl)
    data_logger.save_data_in_datalogger('costs', cost)
    data_logger.save_data_in_datalogger('grads', grad)
    data_logger.save_data_in_datalogger('c_terms', c_terms)
    data_logger.save_data_in_datalogger('unknown_parameters', unknown_parameters)
    data_logger.save_data_in_datalogger('time_needed',
                                        time.time() - data_logger.start_time)

    data_logger.fct_calls = np.append(data_logger.fct_calls,
                                      data_logger.fct_calls[-1] + 1)

    return cost, grad


def initialise_flowline(unknown_parameters, data_logger):
    """
    Initialise a MixedBedFlowline for Minimisation. At the moment only Trapezoidal unknown is
    supported (maybe integrate something like 'fl_control_ind & is_trapezoid')!!
    Parameters
    ----------
    unknown_parameters : :py:class:`numpy.ndarray`
        Unknown parameter values for the optimisation variable (ice part of
        the glacier). For simulaneously optimisation of two variables contains
        both unknown variables consecutively.
    data_logger : TODO

    Returns
    -------

    """

    torch_type = data_logger.torch_type
    device = data_logger.device
    nx = data_logger.flowline_init.nx
    parameter_indices = data_logger.parameter_indices
    known_parameters = data_logger.known_parameters
    fl_init = data_logger.flowline_init

    ice_mask = torch.tensor(data_logger.ice_mask,
                            dtype=torch.bool,
                            device=device,
                            requires_grad=False)

    trap_index = torch.tensor(data_logger.is_trapezoid,
                              dtype=torch.bool,
                              device=device,
                              requires_grad=False)

    # initialise all potential control variables for flowline as empty tensor and fill them
    all_potential_control_vars = ['bed_h', 'surface_h', 'lambdas', 'w0_m']
    fl_vars_total = {}  # they are used for actual initialisation
    fl_control_vars = {}  # are used for reading out the gradients later
    for var in all_potential_control_vars:
        fl_vars_total[var] = torch.empty(nx,
                                         dtype=torch_type,
                                         device=device,
                                         requires_grad=False)

        if var in parameter_indices.keys():
            fl_control_vars[var] = torch.tensor(unknown_parameters[parameter_indices[var]],
                                                dtype=torch_type,
                                                device=device,
                                                requires_grad=True)

            if var in ['bed_h']:
                var_index = ice_mask
            elif var in ['surface_h']:
                var_index = np.full(ice_mask.shape, True)
            elif var in ['lambdas', 'w0_m']:
                var_index = (trap_index & ice_mask)
            else:
                raise NotImplementedError(f'{var}')

            fl_vars_total[var][var_index] = fl_control_vars[var]
            fl_vars_total[var][~var_index] = torch.tensor(known_parameters[var],
                                                          dtype=torch_type,
                                                          device=device,
                                                          requires_grad=False)

        else:
            # if w0_m is no control variable it is calculated to fit widths_m of flowline_init
            if var == 'w0_m':
                var_index = (trap_index & ice_mask)
                init_widths = torch.tensor(fl_init.widths_m,
                                           dtype=torch_type,
                                           device=device,
                                           requires_grad=False)
                init_sfc_h = torch.tensor(fl_init.surface_h,
                                          dtype=torch_type,
                                          device=device,
                                          requires_grad=False)
                lambdas = torch.tensor(fl_init._lambdas,
                                       dtype=torch_type,
                                       device=device,
                                       requires_grad=False)

                fl_vars_total[var][var_index] = \
                    torch.clamp(init_widths[var_index] - lambdas[var_index] *
                                (init_sfc_h[var_index] -
                                 fl_vars_total['bed_h'][var_index]),
                                min=data_logger.min_w0_m)
                fl_vars_total[var][~var_index] = torch.tensor(
                    fl_init._w0_m[~(data_logger.ice_mask & data_logger.is_trapezoid)],
                    dtype=torch_type,
                    device=device,
                    requires_grad=False)
            else:
                if var == 'lambdas':
                    prefix = '_'
                else:
                    prefix = ''
                fl_vars_total[var] = torch.tensor(getattr(fl_init, prefix + var),
                                                  dtype=torch_type,
                                                  device=device,
                                                  requires_grad=False)

    fl = MixedBedFlowline(line=fl_init.line,
                          dx=fl_init.dx,
                          map_dx=fl_init.map_dx,
                          surface_h=fl_vars_total['surface_h'],
                          bed_h=fl_vars_total['bed_h'],
                          section=None,  # is not known during minimisation
                          bed_shape=fl_init.bed_shape,
                          is_trapezoid=fl_init.is_trapezoid,
                          lambdas=fl_vars_total['lambdas'],
                          w0_m=fl_vars_total['w0_m'],
                          rgi_id=fl_init.rgi_id,
                          water_level=fl_init.water_level,
                          torch_type=torch_type,
                          device=device)

    return fl, fl_control_vars


def initialise_mb_models(unknown_parameters,
                         data_logger):
    """TODO"""
    mb_models_settings = data_logger.mb_models_settings
    torch_type = data_logger.torch_type
    device = data_logger.device
    gdir = data_logger.gdir

    mb_models = {}
    for mb_mdl_set in mb_models_settings:
        if mb_models_settings[mb_mdl_set]['type'] == 'constant':
            y_start = mb_models_settings[mb_mdl_set]['years'][0]
            y_end = mb_models_settings[mb_mdl_set]['years'][1]
            # -1 because period defined as [y0 - halfsize, y0 + halfsize + 1]
            y0 = (y_start + y_end - 1) / 2
            halfsize = (y_end - y_start - 1) / 2
            mb_models[mb_mdl_set] = {'mb_model': ConstantMassBalanceTorch(gdir,
                                                                          y0=y0,
                                                                          halfsize=halfsize,
                                                                          torch_type=torch_type,
                                                                          device=device),
                                     'years': mb_models_settings[mb_mdl_set]['years']}
        else:
            raise NotImplementedError("The MassBalance type "
                                      f"{mb_models_settings[mb_mdl_set]['type']} is not "
                                      "implemented!")

    mb_control_var = {}  # only needed if in the future control variables are included in MB-Models
    return mb_models, mb_control_var


def do_height_shift_spinup(flowline, unknown_parameters, data_logger):
    """TODO"""
    mb_models_settings = data_logger.spinup_options['height_shift']['mb_model']
    torch_type = data_logger.torch_type
    device = data_logger.device
    gdir = data_logger.gdir
    parameter_indices = data_logger.parameter_indices

    y_start = mb_models_settings['years'][0]
    y_end = mb_models_settings['years'][1]
    # -1 because period defined as [y0 - halfsize, y0 + halfsize + 1]
    y0 = (y_start + y_end - 1) / 2
    halfsize = (y_end - y_start - 1) / 2

    height_shift = torch.tensor(
        unknown_parameters[parameter_indices['height_shift_spinup']],
        dtype=torch_type,
        device=device,
        requires_grad=True)

    spinup_control_vars = {'height_shift_spinup': height_shift}

    if mb_models_settings['type'] == 'constant':
        mb_spinup = ConstantMassBalanceTorch(gdir,
                                             y0=y0,
                                             halfsize=halfsize,
                                             height_shift=height_shift,
                                             torch_type=torch_type,
                                             device=device)
    else:
        raise NotImplementedError

    model = FluxBasedModel(flowline,
                           mb_spinup,
                           y0=y_start)
    model.run_until(y_end)

    return model.fls[0], spinup_control_vars


def get_cost_terms(observations_mdl,
                   flowline,
                   data_logger):
    """
    Returns the individual terms of the cost function. TODO

    Parameters
    ----------
    observations_mdl
    observations
    data_logger

    Returns
    -------
    costs : :py:class:`torch.Tensor`
        TODO

    """
    observations = data_logger.observations
    # calculate difference between observations and model
    dObs, nObs = calculate_difference_between_observation_and_model(observations,
                                                                    observations_mdl)

    # see if reg parameters need to be specified
    if 'scale' in data_logger.obs_reg_parameters.keys():
        # as a first step scale to Observation values
        define_reg_parameters(data_logger)

    assert 'scale' not in data_logger.obs_reg_parameters.keys()
    reg_parameters = data_logger.obs_reg_parameters

    # define number of cost terms
    if data_logger.regularisation_terms is not None:
        n_costs = nObs + len(data_logger.regularisation_terms.keys())
    else:
        n_costs = nObs

    cost_terms = torch.zeros(n_costs,
                             dtype=data_logger.torch_type,
                             device=data_logger.device,
                             requires_grad=False)

    cost_terms_description = copy.deepcopy(data_logger.observations)

    # to keep track of current calculated cost term
    i_costs = 0

    # calculate cost terms
    for obs_var in dObs.keys():
        for year in dObs[obs_var].keys():
            cost_terms[i_costs] = reg_parameters[obs_var][year] * \
                                  dObs[obs_var][year]

            cost_terms_description[obs_var][year] = \
                cost_terms[i_costs].detach().to('cpu').numpy()

            i_costs += 1

    # calculate regularisation terms
    for reg_term in data_logger.regularisation_terms.keys():
        if reg_term == 'smoothed_bed':
            b = flowline.bed_h
            dx = flowline.dx * flowline.map_dx
            db_dx = (b[1:] - b[:-1]) / dx
            reg_par = torch.tensor(data_logger.regularisation_terms[reg_term],
                                   dtype=data_logger.torch_type,
                                   device=data_logger.device,
                                   requires_grad=False)
            cost_terms[i_costs] = reg_par * db_dx.pow(2).sum()

            cost_terms_description[reg_term] = \
                cost_terms[i_costs].detach().to('cpu').numpy()

        else:
            raise NotImplementedError(f'{reg_term}')
        i_costs += 1

    assert i_costs == n_costs

    data_logger.save_data_in_datalogger('c_terms_description',
                                        cost_terms_description)

    return cost_terms


def calculate_difference_between_observation_and_model(observations, observations_mdl):
    dobs = copy.deepcopy(observations)
    nobs = 0
    for ob in observations.keys():
        for year in observations[ob]:
            nobs += 1
            dobs[ob][year] = (torch.tensor(observations[ob][year],
                                           dtype=observations_mdl[ob][year].dtype,
                                           device=observations_mdl[ob][year].device,
                                           requires_grad=False) -
                              observations_mdl[ob][year]).pow(2).sum()

    return dobs, nobs


def define_reg_parameters(data_logger):
    observations = data_logger.observations
    scales = data_logger.obs_reg_parameters['scale']
    torch_type = data_logger.torch_type
    device = data_logger.device

    reg_parameters = copy.deepcopy(observations)

    # reference Magnitude to scale to, arbitrary
    ref_magnitude = torch.tensor([100.],
                                 dtype=torch_type,
                                 device=device,
                                 requires_grad=False)

    # loop through Obs and choose reg_parameters to scale to the absolute value
    # of Observation, the calculated reg_parameters are squared because the
    # differences (obs-mdl) are also squared in cost calculation
    for obs_val in observations.keys():
        for year in observations[obs_val].keys():
            # sum is needed for observations along the flowline
            tmp_obs = torch.tensor(observations[obs_val][year],
                                   dtype=torch_type,
                                   device=device,
                                   requires_grad=False).sum()
            reg_parameters[obs_val][year] = (ref_magnitude / tmp_obs *
                                             torch.tensor(scales[obs_val],
                                                          dtype=torch_type,
                                                          device=device,
                                                          requires_grad=False)
                                             ).pow(2)

    data_logger.obs_reg_parameters = reg_parameters


def get_gradients(fl_control_vars, mb_control_vars, spinup_control_vars,
                  data_logger, length):
    parameter_indices = data_logger.parameter_indices
    grad = np.zeros(length, dtype='float64')

    for var in parameter_indices.keys():
        if var in fl_control_vars.keys():
            grad[parameter_indices[var]] = fl_control_vars[var].grad.detach(
            ).to('cpu').numpy().astype(np.float64)
        elif var in mb_control_vars.keys():
            grad[parameter_indices[var]] = mb_control_vars[var].grad.detach(
            ).to('cpu').numpy().astype(np.float64)
        elif var in spinup_control_vars.keys():
            grad[parameter_indices[var]] = spinup_control_vars[var].grad.detach(
            ).to('cpu').numpy().astype(np.float64)
        else:
            raise NotImplementedError('No gradient available for ' + var + '!')

    return grad
