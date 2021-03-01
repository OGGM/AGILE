import numpy as np
import torch
import time

from combine.core.dynamics import run_flowline_forward_core
from combine.core.arithmetics import to_torch_tensor, to_numpy_array
from combine.core.massbalance_adapted import LinearMassBalance


def create_cost_fct(known_parameter,
                    geometry_var,
                    bed_geometry,
                    measurements,
                    reg_parameters,
                    dx,
                    mb_model,
                    opti_var,
                    two_parameter_option,
                    datalogger,
                    grad_scaling={'bed_h': 1,
                                  'shape_var': 1},
                    min_w0=10.,
                    spinup_sfc_known=True,
                    spinup_yrs=None,
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
                        two_parameter_option,
                        datalogger,
                        grad_scaling=grad_scaling,
                        min_w0=min_w0,
                        spinup_sfc_known=spinup_sfc_known,
                        spinup_yrs=spinup_yrs,
                        only_get_c_terms=only_get_c_terms,
                        torch_type=torch_type)

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
             two_parameter_option,
             datalogger,
             grad_scaling={'bed_h': 1,
                           'shape_var': 1},
             min_w0=10.,
             spinup_sfc_known=True,
             spinup_yrs=None,
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

    # save original length of unkown parameter,
    # needed for potential MemoryOverflowError
    unknown_paramter_length = len(unknown_parameter)

    # try without checks and with bounds
    # check that all parameters are positive
    # if np.any(np.sign(unknown_parameter) == -1):
    #    cost = np.Inf
    #    grad = np.empty(len(unknown_parameter)) * np.nan
    #    return cost, grad

    # ice mask is needed for cost calculation and to put parameters together
    ice_mask = torch.tensor(measurements['ice_mask'],
                            requires_grad=False,
                            dtype=torch.bool)

    if not spinup_sfc_known:
        if only_get_c_terms:
            spinup_ELA = torch.tensor(unknown_parameter[0],
                                      dtype=torch_type,
                                      requires_grad=True)
            unknown_parameter = unknown_parameter[1:]
            spinup_mb_model = LinearMassBalance(spinup_ELA,
                                                grad=mb_model['grad_spinup'])
        elif (two_parameter_option == 'separated') & \
             (datalogger.opti_var_2 is not None):
            if opti_var == 'bed_h':
                spinup_ELA = torch.tensor(unknown_parameter[0],
                                          dtype=torch_type,
                                          requires_grad=True)
                unknown_parameter = unknown_parameter[1:]
            else:
                spinup_ELA = torch.tensor(datalogger.spinup_ELA_guessed[-1],
                                          dtype=torch_type,
                                          requires_grad=False)
            spinup_mb_model = LinearMassBalance(spinup_ELA,
                                                grad=mb_model['grad_spinup'])

        else:
            spinup_ELA = torch.tensor(unknown_parameter[0],
                                      dtype=torch_type,
                                      requires_grad=True)
            unknown_parameter = unknown_parameter[1:]
            spinup_mb_model = LinearMassBalance(spinup_ELA,
                                                grad=mb_model['grad_spinup'])

        mb_model = {'spinup_mb_model': spinup_mb_model,
                    'known_mb_model': mb_model['model_known']}
    else:
        spinup_ELA = None

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

    elif ((two_parameter_option == 'at_once') &
          (opti_var in ['bed_h and bed_shape', 'bed_h and w0'])):
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

    elif ((two_parameter_option == 'calculated') &
          (opti_var == 'bed_h and bed_shape')):
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

        shape_var_known = torch.tensor(geometry_var,
                                       dtype=torch_type,
                                       requires_grad=False)
        true_widths = to_torch_tensor(measurements['widths'], torch_type)
        true_sfc_h = to_torch_tensor(measurements['sfc_h'], torch_type)

        # choose wich version to use for bed_shape calculation, 0 default
        bed_shape_calculation_version = 2

        if bed_shape_calculation_version == 0:
            # calculating with no restrictions
            shape_var_unknown = to_torch_tensor(4., torch_type) * \
                                (true_sfc_h[ice_mask] - bed_h_unknown) / \
                                (true_widths[ice_mask])**2
        elif bed_shape_calculation_version == 1:
            # use a minimum width of 10 for calculation, equ. OGGM first guess
            shape_var_unknown = torch.where(
                true_widths[ice_mask] >= to_torch_tensor(10., torch_type),
                to_torch_tensor(4., torch_type) *
                (true_sfc_h[ice_mask] - bed_h_unknown) /
                (true_widths[ice_mask])**2,
                to_torch_tensor(4., torch_type) *
                (true_sfc_h[ice_mask] - bed_h_unknown) /
                to_torch_tensor(10., torch_type)**2)
        elif bed_shape_calculation_version == 2:
            # define a maximum value of 3
            shape_var_unknown = torch.clamp(
                to_torch_tensor(4., torch_type) *
                (true_sfc_h[ice_mask] - bed_h_unknown) /
                (true_widths[ice_mask])**2,
                max=3)

        shape_var = torch.empty(sum(list(shape_var_unknown.size() +
                                         shape_var_known.size())),
                                dtype=torch_type,
                                requires_grad=False)
        shape_var[ice_mask] = shape_var_unknown
        shape_var[~ice_mask] = shape_var_known

    elif ((two_parameter_option == 'calculated') &
          (opti_var == 'bed_h and w0')):
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

        shape_var_known = torch.tensor(geometry_var,
                                       dtype=torch_type,
                                       requires_grad=False)
        true_widths = to_torch_tensor(measurements['widths'], torch_type)
        true_sfc_h = to_torch_tensor(measurements['sfc_h'], torch_type)
        # 1. for constant lambda of trapozidal bed geometry
        shape_var_unknown = torch.clamp(true_widths[ice_mask] -
                                        to_torch_tensor(1., torch_type) *
                                        (true_sfc_h[ice_mask] - bed_h_unknown),
                                        min=min_w0)
        shape_var = torch.empty(sum(list(shape_var_unknown.size() +
                                         shape_var_known.size())),
                                dtype=torch_type,
                                requires_grad=False)
        shape_var[ice_mask] = shape_var_unknown
        shape_var[~ice_mask] = shape_var_known

    else:
        raise ValueError('Unknown combination of opti var and '
                         'two parameter option!')

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
            spinup_sfc_known=spinup_sfc_known,
            yrs_to_run=measurements['yrs_to_run'],
            spinup_yrs=spinup_yrs,
            map_dx=map_dx,
            torch_type=torch_type)

        model_sfc_h = model_flowline.surface_h
        model_widths = model_flowline.widths_m
        model_thicks = model_flowline.thick
    except MemoryError:
        print('MemoryError in forward model run (due to a too small '
              'timestep) -> set Costfunction to Inf')
        cost = np.Inf
        grad = np.empty(unknown_paramter_length) * np.nan
        return cost, grad

    # calculate terms of cost function
    c_terms = get_cost_terms(
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
        if two_parameter_option == 'at_once':
            grad_bed_h = \
                bed_h_unknown.grad.detach().numpy().astype(np.float64) * \
                grad_scaling['bed_h']
            grad_shape_var = \
                shape_var_unknown.grad.detach().numpy().astype(np.float64) * \
                grad_scaling['shape_var']
            grad = np.append(grad_bed_h, grad_shape_var)
        elif two_parameter_option == 'calculated':
            grad = bed_h_unknown.grad.detach().numpy().astype(np.float64)
        else:
            raise ValueError('Unknown two opti parameter option!')
    else:
        raise ValueError('Unknown optimisation variable!')

    if not spinup_sfc_known:
        if (datalogger.two_parameter_option == 'separated') & \
           (datalogger.opti_var_2 is not None):
            if opti_var == 'bed_h':
                spinup_ELA_grad = \
                    spinup_ELA.grad.detach().numpy().astype(np.float64)
                grad_return = np.append(spinup_ELA_grad,
                                        grad)
            else:
                spinup_ELA_grad = np.nan
                grad_return = grad
        else:
            spinup_ELA_grad = \
                spinup_ELA.grad.detach().numpy().astype(np.float64)
            grad_return = np.append(spinup_ELA_grad,
                                    grad)
    else:
        grad_return = grad

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
    datalogger.save_data_in_datalogger('time_needed',
                                       time.time() - datalogger.start_time)

    if not spinup_sfc_known:
        datalogger.save_data_in_datalogger('spinup_ELA_guessed', spinup_ELA)
        datalogger.save_data_in_datalogger('spinup_ELA_grad', spinup_ELA_grad)
        datalogger.save_data_in_datalogger('spinup_sfc_h_guessed',
                                           model_flowline.spinup_sfc_h)
        datalogger.save_data_in_datalogger('spinup_widths_guessed',
                                           model_flowline.spinup_widths)

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
        if two_parameter_option == 'at_once':
            datalogger.save_data_in_datalogger('guessed_opti_var_1',
                                               bed_h_unknown)
            datalogger.save_data_in_datalogger('grads_opti_var_1', grad_bed_h)
            datalogger.save_data_in_datalogger('guessed_opti_var_2',
                                               shape_var_unknown)
            datalogger.save_data_in_datalogger('grads_opti_var_2',
                                               grad_shape_var)
        elif two_parameter_option == 'calculated':
            datalogger.save_data_in_datalogger('guessed_opti_var_1',
                                               bed_h_unknown)
            datalogger.save_data_in_datalogger('grads_opti_var_1', grad)
            datalogger.save_data_in_datalogger('guessed_opti_var_2',
                                               shape_var_unknown)
            datalogger.save_data_in_datalogger('grads_opti_var_2',
                                               np.empty(len(grad)) * np.nan)
    else:
        raise ValueError('Unknown optimisation variable!')

    datalogger.fct_calls = np.append(datalogger.fct_calls,
                                     datalogger.fct_calls[-1] + 1)

    return cost, grad_return


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

    ice_mask = to_torch_tensor(true_ice_mask, torch.bool)

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

    # smoothnes of glacier bed, use mean of forward and backward
    db_dx = (model_bed_h[1:] - model_bed_h[:-1]) / dx
    costs[3] = reg_parameters[3] * db_dx.pow(2).sum()

    # smooth transition from at boundering pixel
    # last_glacier_pixel = np.argwhere(true_ice_mask)[-1][-1]
    # costs[4] = reg_parameters[4] * ((model_bed_h[last_glacier_pixel + 1] -
    #                                 2 * model_bed_h[last_glacier_pixel] +
    #                                 model_bed_h[last_glacier_pixel - 1]) /
    #                                dx**2).pow(2)

    return costs
