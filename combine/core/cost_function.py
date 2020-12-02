import numpy as np
import torch

from combine.core.dynamics import run_flowline_forward_core
from combine.core.arithmetics import to_torch_tensor


def creat_spinup_cost_fct(measurements,
                          mb_model,
                          first_guess,
                          used_geometry,
                          geometry,
                          torch_type):
    '''
    Creates a cost function for optimizing t bias for the spinup
    '''

    def c_fun(t_bias):
        return spinup_cost_fct(t_bias,
                               measurements,
                               mb_model,
                               first_guess,
                               used_geometry,
                               geometry,
                               torch_type)

    return c_fun


def spinup_cost_fct(t_bias,
                    measurements,
                    mb_model,
                    first_guess,
                    used_geometry,
                    geometry,
                    torch_type='double'):
    # check which data type should be used for calculation
    if torch_type == 'double':
        torch_type = torch.double
    else:
        torch_type = torch.float

    temp_bias = torch.tensor(t_bias,
                             dtype=torch_type,
                             requires_grad=True)
    # set the temperature bias for the mass balance model
    mb_model.temp_bias = temp_bias

    # run the model with the first guess geometry to equilibrium and get volume
    model_flowline = run_flowline_forward_core(
            spinup_surf=first_guess['bed_h'],
            bed_h=first_guess['bed_h'],
            shape=first_guess['shape'],
            dx=geometry['map_dx'],
            torch_type=torch_type,
            mb_model=mb_model,
            yrs_to_run='equilibrium',
            used_geometry=used_geometry,
            ref_surf=measurements['sfc_h'],
            ref_width=measurements['widths'])
    model_volume = model_flowline.volume_m3

    ref_volume = to_torch_tensor(measurements['spinup_volume'], torch_type)

    c = torch.abs(model_volume - ref_volume)
    c.backward()

    cost = c.detach().numpy().astype(np.float64)
    grad = temp_bias.grad.detach().numpy().astype(np.float64)

    return cost, grad


def create_cost_fct(known_parameter,
                    geometry_var,
                    bed_geometry,
                    measurements,
                    reg_parameters,
                    dx,
                    mb_model,
                    opti_var,
                    datalogger,
                    only_get_c_terms=False,
                    torch_type='double'):
    '''
    Creating a cost function for the given parameters. For a detailed parameter
    description look at Docstring of function
    combine.core.cost_function.cost_fct

    Parameters
    ----------
    known_parameter : TYPE
        DESCRIPTION.
    geometry_var : TYPE
        DESCRIPTION.
    bed_geometry : TYPE
        DESCRIPTION.
    measurements : TYPE
        DESCRIPTION.
    reg_parameters : TYPE
        DESCRIPTION.
    dx : TYPE
        DESCRIPTION.
    mb_model : TYPE
        DESCRIPTION.
    opti_var : TYPE
        DESCRIPTION.
    datalogger : TYPE
        DESCRIPTION.
    get_c_terms : TYPE, optional
        DESCRIPTION. The default is False.
    torch_type : TYPE, optional
        DESCRIPTION. The default is 'double'.

    Returns
    -------
    Function which takes one input variable (unknown_parameter) and return two
    values (cost, grad)
    '''

    def c_fun(unknown_parameter):
        return cost_fct(unknown_parameter,
                        known_parameter,
                        geometry_var,
                        bed_geometry,
                        measurements,
                        reg_parameters,
                        dx,
                        mb_model,
                        opti_var,
                        datalogger,
                        only_get_c_terms,
                        torch_type)

    return c_fun


def cost_fct(unknown_parameter,
             known_parameter,
             geometry_var,
             bed_geometry,
             measurements,
             reg_parameters,
             map_dx,
             mb_model,
             opti_var,
             datalogger,
             only_get_c_terms=False,
             torch_type='double'):
    '''
    Calculates cost and gradient for the given parameters.

    Parameters
    ----------
    unknown_parameter : :py:class:`numpy.ndarray`
        Unknown parameter values for the optimisation variable (ice part of
        the glacier). For simulaneously optimisation of two variables contains
        both unknown variables consecutively.
    known_parameter : :py:class:`numpy.ndarray`
        Known parameter values for the optimisation variable (ice free part of
        the glacier). For simulaneously optimisation of two variables contains
        both known variables consecutively.
    geometry_var : :py:class:`numpy.ndarray`
        If only one optimisation variable contains the second variable which is
        needed to define the flowline shape.
    bed_geometry : str
        Defines the bed shape.
        Options: 'rectangular', 'parabolic' or 'trapezoidal'
    measurements : dict
        Dictionary containing the measurements from:
            'spinup_sfc_h' : the spinup surface height (start ice height)
            'sfc_h': the desired ice surface height at the end
            'widths': the desired widths at the end
            'yrs_to_run': the number of years the model should run
            'ice_mask': indicates where ice is located at the end (TRUE = ice)
    reg_parameters : :py:class:`numpy.ndarray`
        Regularisation parameters for the individual terms of the cost
        function.
    map_dx : float
        Model grid spacing in meters.
    mb_model : :py:class:`oggm.core.massbalance.MassBalanceModel`
        The mass balance model to use.
    opti_var : str
        Defines the optimisation parameter. Depending on the bed geometry this
        could be one ore two.
        Options for 'rectangular': 'bed_h'.
        Options for 'parabolic': 'bed_h', 'bed_shape' or 'bed_h and bed_shape'
        Options for 'trapezoidal': 'bed_h', 'w0' or 'bed_h and w0'
    datalogger : :py:class:`combine.core.data_logging.DataLogger`
        Datalogger to keep track of the calculations.
    get_c_terms : bool, optional
        If true the function only returns the cost function terms. This is a
        shortcut for determining the regularisation parameters only by a
        scaling of magnitude. The default is False.
    torch_type : str, optional
        Defines type for torch.Tensor. If 'double' use torch.double, otherwise
        use torch.float. The default is 'double'.

    Returns
    -------
    :py:class:`numpy.ndarray`
        The actual value of the cost function.
    :py:class:`numpy.ndarray`
        The gradient for each unknown_parameter with respect to the cost value.

    '''
    # check which data type should be used for calculation
    if torch_type == 'double':
        torch_type = torch.double
    else:
        torch_type = torch.float

    # try without checks and with bounds
    # check that all parameters are positive
    if np.any(np.sign(unknown_parameter) == -1):
        cost = np.Inf
        grad = np.empty(len(unknown_parameter)) * np.nan
        return cost, grad

    # ice mask is needed for cost calculation and to put parameters together
    ice_mask = torch.tensor(measurements['ice_mask'],
                            requires_grad=False,
                            dtype=torch.bool)

    # check which parameter should be optimized
    if opti_var == 'bed_h':
        bed_h_unknown = torch.tensor(unknown_parameter,
                                     dtype=torch_type,
                                     requires_grad=True)
        bed_h_known = torch.tensor(known_parameter,
                                   dtype=torch_type,
                                   requires_grad=False)
        bed_h = torch.empty(sum(list(bed_h_unknown.size() +
                                     bed_h_known.size())),
                            dtype=torch_type,
                            requires_grad=False)
        bed_h[ice_mask] = bed_h_unknown
        bed_h[~ice_mask] = bed_h_known

        shape_var = torch.tensor(geometry_var,
                                 dtype=torch_type,
                                 requires_grad=False)

    elif opti_var in ['bed_shape', 'w0']:
        shape_var_unknown = torch.tensor(unknown_parameter,
                                         dtype=torch_type,
                                         requires_grad=True)
        shape_var_known = torch.tensor(known_parameter,
                                       dtype=torch_type,
                                       requires_grad=False)
        shape_var = torch.empty(sum(list(shape_var_unknown.size() +
                                         shape_var_known.size())),
                                dtype=torch_type,
                                requires_grad=False)
        shape_var[ice_mask] = shape_var_unknown
        shape_var[~ice_mask] = shape_var_known

        bed_h = torch.tensor(geometry_var,
                             dtype=torch_type,
                             requires_grad=False)

    elif opti_var in ['bed_h and bed_shape', 'bed_h and w0']:
        split_point_unknown = int(len(unknown_parameter) / 2)
        split_point_known = int(len(known_parameter) / 2)

        bed_h_unknown = unknown_parameter[:split_point_unknown]
        bed_h_unknown = torch.tensor(bed_h_unknown,
                                     dtype=torch_type,
                                     requires_grad=True)

        bed_h_known = known_parameter[:split_point_known]
        bed_h_known = torch.tensor(bed_h_known,
                                   dtype=torch_type,
                                   requires_grad=False)

        bed_h = torch.empty(sum(list(bed_h_unknown.size() +
                                     bed_h_known.size())),
                            dtype=torch_type,
                            requires_grad=False)
        bed_h[ice_mask] = bed_h_unknown
        bed_h[~ice_mask] = bed_h_known

        shape_var_unknown = unknown_parameter[split_point_unknown:]
        shape_var_unknown = torch.tensor(shape_var_unknown,
                                         dtype=torch_type,
                                         requires_grad=True)

        shape_var_known = known_parameter[split_point_known:]
        shape_var_known = torch.tensor(shape_var_known,
                                       dtype=torch_type,
                                       requires_grad=False)

        shape_var = torch.empty(sum(list(shape_var_unknown.size() +
                                         shape_var_known.size())),
                                dtype=torch_type,
                                requires_grad=False)
        shape_var[ice_mask] = shape_var_unknown
        shape_var[~ice_mask] = shape_var_known

    else:
        raise ValueError('Unknown optimisation variable!')

    # check if bed_h and shape_var are the same length
    assert len(bed_h) == len(shape_var), 'Parameters not the same length!!!'

    # forward run of model, try is needed to avoid a memory overflow
    try:
        model_flowline = run_flowline_forward_core(
            bed_h=bed_h,
            shape_var=shape_var,
            bed_geometry=bed_geometry,
            mb_model=mb_model,
            spinup_sfc_h=measurements['spinup_sfc'],
            yrs_to_run=measurements['yrs_to_run'],
            map_dx=map_dx,
            torch_type=torch_type)

        model_sfc_h = model_flowline.surface_h
        model_widths = model_flowline.widths_m
        model_thicks = model_flowline.thick
    except MemoryError:
        print('MemoryError in forward model run (due to a too small timestep) \
              -> set Costfunction to Inf')
        cost = np.Inf
        grad = np.empty(len(unknown_parameter)) * np.nan
        return cost, grad

    # calculate terms of cost function
    c_terms = get_cost_terms_new(
        model_sfc_h=model_sfc_h,
        model_widths=model_widths,
        model_thicks=model_thicks,
        model_bed_h=bed_h,
        true_sfc_h=measurements['sfc_h'],
        true_widths=measurements['widths'],
        true_ice_mask=measurements['ice_mask'],
        reg_parameters=reg_parameters,
        dx=map_dx,
        torch_type=torch_type)

    # shortcut when regularisation parameters are searched
    if only_get_c_terms:
        return c_terms.detach().numpy().astype(np.float64)

    # sum up cost function terms
    c = c_terms.sum()

    # calculate the gradient for the optimisation parameter
    c.backward()

    # convert cost to numpy array
    cost = c.detach().numpy().astype(np.float64)

    # get gradient/s and convert to numpy array
    if opti_var == 'bed_h':
        grad = bed_h_unknown.grad.detach().numpy().astype(np.float64)
    elif opti_var in ['bed_shape', 'w0']:
        grad = shape_var_unknown.grad.detach().numpy().astype(np.float64)
    elif opti_var in ['bed_h and bed_shape', 'bed_h and w0']:
        grad_bed_h = bed_h_unknown.grad.detach().numpy().astype(np.float64)
        grad_shape_var = \
            shape_var_unknown.grad.detach().numpy().astype(np.float64)
        grad = np.append(grad_bed_h, grad_shape_var)
    else:
        raise ValueError('Unknown optimisation variable!')

    # help function to save data in datalogger as numpy array
    # def save_data_in_datalogger(datalogger_var, data):
    #    if type(data) == torch.Tensor:
    #        data = data.detach().numpy().astype(np.float64)

    #    datalogger_var = np.append(datalogger_var, data)

    # save data in datalogger
    datalogger.save_data_in_datalogger('costs', cost)
    datalogger.save_data_in_datalogger('c_terms', c_terms)
    datalogger.save_data_in_datalogger('sfc_h', model_sfc_h)
    datalogger.save_data_in_datalogger('widths', model_widths)
    datalogger.save_data_in_datalogger('opti_var_iteration', opti_var)
    datalogger.save_data_in_datalogger('current_main_iterations',
                                       datalogger.main_iterations[-1])
    if opti_var == 'bed_h':
        datalogger.save_data_in_datalogger('guessed_opti_var_1', bed_h_unknown)
        datalogger.save_data_in_datalogger('grads_opti_var_1', grad)
        # save second variable if currently in separated optimisation
        if datalogger.opti_var_2 is not None:
            datalogger.save_data_in_datalogger('guessed_opti_var_2',
                                               shape_var[ice_mask])
            datalogger.save_data_in_datalogger('grads_opti_var_2',
                                               np.empty(len(grad)) * np.nan)
    elif opti_var in ['bed_shape', 'w0']:
        # check if currently in separated optimisation
        if datalogger.opti_var_2 is None:
            datalogger.save_data_in_datalogger('guessed_opti_var_1',
                                               shape_var_unknown)
            datalogger.save_data_in_datalogger('grads_opti_var_1', grad)
        else:
            datalogger.save_data_in_datalogger('guessed_opti_var_1',
                                               bed_h[ice_mask])
            datalogger.save_data_in_datalogger('grads_opti_var_1',
                                               np.empty(len(grad)) * np.nan)

            datalogger.save_data_in_datalogger('guessed_opti_var_2',
                                               shape_var_unknown)
            datalogger.save_data_in_datalogger('grads_opti_var_2', grad)

    elif opti_var in ['bed_h and bed_shape', 'bed_h and w0']:
        datalogger.save_data_in_datalogger('guessed_opti_var_1', bed_h_unknown)
        datalogger.save_data_in_datalogger('grads_opti_var_1', grad_bed_h)
        datalogger.save_data_in_datalogger('guessed_opti_var_2',
                                           shape_var_unknown)
        datalogger.save_data_in_datalogger('grads_opti_var_2', grad_shape_var)
    else:
        raise ValueError('Unknown optimisation variable!')

    datalogger.fct_calls = np.append(datalogger.fct_calls,
                                     datalogger.fct_calls[-1] + 1)

    return cost, grad


def get_cost_terms(model_sfc_h,
                   model_widths,
                   model_thicks,
                   model_bed_h,
                   true_sfc_h,
                   true_widths,
                   true_ice_mask,
                   reg_parameters,
                   dx,
                   torch_type):
    '''
    Returns the individual terms of the cost function.

    Parameters
    ----------
    model_sfc_h : :py:class:`torch.Tensor`
        Surface heights of the modeled glacier.
    model_widths : :py:class:`torch.Tensor`
        Widths of the modeled glacier.
    model_thicks : :py:class:`torch.Tensor`
        Thickness of the modeled glacier.
    model_bed_h : :py:class:`torch.Tensor`
        Bed heights of the modeled glacier.
    true_sfc_h : :py:class:`numpy.ndarray`
        Surface heights from measurements.
    true_widths : :py:class:`numpy.ndarray`
        Widths from measurements.
    true_ice_mask : :py:class:`numpy.ndarray`
        Ice maks from measurements (1 = ice, 0 = no ice).
    reg_parameters : :py:class:`numpy.ndarray`
        Regularisation parameters for the individual terms.
    dx : float
        Model grid spacing in meters.
    torch_type : :py:class:`torch.dtype`
        Defines type for torch.Tensor.

    Returns
    -------
    costs : :py:class:`torch.Tensor`
        Contains the four terms of the final cost function.

    '''
    # calculate cost terms
    costs = torch.zeros(4,
                        dtype=torch_type)

    # misfit between modeled and measured surface height
    true_sfc_h = to_torch_tensor(true_sfc_h, torch_type)
    costs[0] = reg_parameters[0] * (true_sfc_h - model_sfc_h).pow(2).sum()

    # misfit between modeled and measured width
    true_widths = to_torch_tensor(true_widths, torch_type)
    costs[1] = reg_parameters[1] * ((true_widths - model_widths)).pow(2).sum()

    # ice thickness close to zero where no glacier should be and vice versa
    model_ice_mask = torch.where(model_thicks > 1e-2,
                                 torch.tensor(1),
                                 torch.tensor(0))
    true_ice_mask = to_torch_tensor(true_ice_mask, torch_type)
    costs[2] = reg_parameters[2] * torch.where(true_ice_mask != model_ice_mask,
                                               torch.tensor(1.),
                                               torch.tensor(0.)).sum()

    # smoothnes of glacier bed
    db_dx = (model_bed_h[1:] - model_bed_h[:-1]) / dx
    costs[3] = reg_parameters[3] * db_dx.pow(2).sum()

    return costs


def get_cost_terms_new(model_sfc_h,
                       model_widths,
                       model_thicks,
                       model_bed_h,
                       true_sfc_h,
                       true_widths,
                       true_ice_mask,
                       reg_parameters,
                       dx,
                       torch_type):
    '''
    Returns the individual terms of the cost function.

    Parameters
    ----------
    model_sfc_h : :py:class:`torch.Tensor`
        Surface heights of the modeled glacier.
    model_widths : :py:class:`torch.Tensor`
        Widths of the modeled glacier.
    model_thicks : :py:class:`torch.Tensor`
        Thickness of the modeled glacier.
    model_bed_h : :py:class:`torch.Tensor`
        Bed heights of the modeled glacier.
    true_sfc_h : :py:class:`numpy.ndarray`
        Surface heights from measurements.
    true_widths : :py:class:`numpy.ndarray`
        Widths from measurements.
    true_ice_mask : :py:class:`numpy.ndarray`
        Ice maks from measurements (1 = ice, 0 = no ice).
    reg_parameters : :py:class:`numpy.ndarray`
        Regularisation parameters for the individual terms.
    dx : float
        Model grid spacing in meters.
    torch_type : :py:class:`torch.dtype`
        Defines type for torch.Tensor.

    Returns
    -------
    costs : :py:class:`torch.Tensor`
        Contains the four terms of the final cost function.

    '''
    # calculate cost terms
    costs = torch.zeros(4,
                        dtype=torch_type)

    ice_mask = to_torch_tensor(true_ice_mask, torch_type)

    # misfit between modeled and measured surface height for points with ice
    true_sfc_h = to_torch_tensor(true_sfc_h, torch_type)
    costs[0] = reg_parameters[0] * (true_sfc_h[ice_mask] -
                                    model_sfc_h[ice_mask]).pow(2).sum()

    # misfit between modeled and measured width for points with ice
    true_widths = to_torch_tensor(true_widths, torch_type)
    costs[1] = reg_parameters[1] * ((true_widths[ice_mask] -
                                     model_widths[ice_mask])).pow(2).sum()

    # pnelalize ice outside of glacier outline
    costs[2] = reg_parameters[2] * (true_sfc_h[~ice_mask] -
                                    model_sfc_h[~ice_mask]).pow(2).sum()

    # smoothnes of glacier bed
    db_dx = (model_bed_h[1:] - model_bed_h[:-1]) / dx
    costs[3] = reg_parameters[3] * db_dx.pow(2).sum()

    return costs
