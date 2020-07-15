import numpy as np

from combine.core.cost_function import creat_cost_fct
from scipy.optimize import minimize
from combine.core.idealized_experiments import define_geometry,\
    define_mb_model, create_measurements, get_first_guess, get_reg_parameters,\
    plot_result
from combine.core.data_logging import DataLogger_bed_h_and_shape


def optimize_bed_h_and_shape(bed_h_guess,
                             shape_guess,
                             bed_h_known,
                             shape_known,
                             mb_model,
                             spinup_surf,
                             ref_surf,
                             ref_widths,
                             yrs_to_run,
                             map_dx,
                             minimize_options_bed_h,
                             minimize_options_shape,
                             reg_parameter_bed_h=np.array([1.]),
                             reg_parameter_shape=np.array([1.]),
                             torch_type='double',
                             used_geometry='parabolic',
                             data_logger=None,
                             iterations=5,
                             check_cost_terms=False,
                             ):
    for loop in range(iterations):
        bed_h_success = False
        shape_success = False

        if data_logger is not None:
            data_logger.main_iterations.append(loop + 1)
            data_logger.main_iteration_callback()

        # create cost function for bed height
        shape_all = np.append(shape_guess, shape_known)

        cost_fct_bed_h = creat_cost_fct(
            bed_h=bed_h_known,
            shape=shape_all,
            spinup_surf=spinup_surf,
            reg_parameter=reg_parameter_bed_h,
            ref_surf=ref_surf,
            ref_width=ref_widths,
            yrs_to_run=yrs_to_run,
            dx=map_dx,
            mb_model=mb_model,
            opti_var='bed_h',
            torch_type=torch_type,
            used_geometry=used_geometry,
            data_logger=data_logger)

        if check_cost_terms:
            cost_fct_bed_h(bed_h_guess)
            break

        minimize_options_bed_h['maxiter'] = loop + 1
        res = minimize(fun=cost_fct_bed_h,
                       x0=bed_h_guess,
                       method='L-BFGS-B',
                       jac=True,
                       # bounds=bounds,
                       options=minimize_options_bed_h,
                       callback=data_logger.callback_fct)

        bed_h_guess = res.x
        bed_h_success = res.success

        if data_logger is not None:
            data_logger.current_step_indices_bed_h = []

        # create cost function for bed shape
        bed_h_all = np.append(bed_h_guess, bed_h_known)

        cost_fct_shape = creat_cost_fct(
            bed_h=bed_h_all,
            shape=shape_known,
            spinup_surf=spinup_surf,
            reg_parameter=reg_parameter_shape,
            ref_surf=ref_surf,
            ref_width=ref_widths,
            yrs_to_run=yrs_to_run,
            dx=map_dx,
            mb_model=mb_model,
            opti_var='shape',
            torch_type=torch_type,
            used_geometry=used_geometry,
            data_logger=data_logger)

        minimize_options_shape['maxiter'] = loop + 1
        res = minimize(fun=cost_fct_shape,
                       x0=shape_guess,
                       method='L-BFGS-B',
                       jac=True,
                       # bounds=bounds,
                       options=minimize_options_shape,
                       callback=data_logger.callback_fct)

        shape_guess = res.x
        shape_success = res.success

        if data_logger is not None:
            data_logger.current_step_indices_shape = []

        if bed_h_success and shape_success:
            print('Optimisation was successful!')
            break

    return bed_h_guess, shape_guess


def idealized_inversion_experiment(used_bed_h_geometry='linear',
                                   used_along_glacier_geometry='linear',
                                   bed_shape='parabolic',
                                   mb_type='linear',
                                   mb_opts={'ELA': np.array([3000.]),
                                            'grad': np.array([4.])},
                                   glacier_state='equilibrium',
                                   add_measurement_noise=False,
                                   first_guess_method='oggm',
                                   first_guess_const=None,
                                   opti_parameter='bed_h and shape at once',
                                   use_datalogger=True,
                                   reg_parameters=None,
                                   wanted_c_terms=None,
                                   grad_scaling={'bed_h': 1,
                                                 'shape': 1},
                                   grad_smoothing={'bed_h': '2nd is 1st',
                                                   'shape': 'last 3 same'},
                                   torch_type='double',
                                   minimize_options={'maxiter': 10,
                                                     'ftol': 1e-7,
                                                     'gtol': 1e-8,
                                                     'disp': True,
                                                     'maxcor': 50,
                                                     'maxls': 50},
                                   show_plot=True):
    # define glacier bed geometry
    print('- Define geometry: ')
    geometry = define_geometry(used_bed_h_geometry,
                               used_along_glacier_geometry,
                               bed_shape)
    print('---DONE---')

    # define mass balance profile
    print('\n- Define mass balance model: ')
    mb_model = define_mb_model(mb_type, mb_opts)
    print('---DONE---')

    # create some measurements
    print('\n- Create Measurements: ')
    measurements = create_measurements(geometry,
                                       mb_model,
                                       bed_shape=bed_shape,
                                       glacier_state=glacier_state,
                                       add_noise=add_measurement_noise)

    if glacier_state == 'retreating':
        # use only second mass balance model for optimization
        mb_model = mb_model[1]
    print('---DONE---')

    # get first guess
    if opti_parameter == 'bed_h':
        opti_parameter_first_guess = 'bed_h'
    elif opti_parameter == 'shape':
        opti_parameter_first_guess = 'shape'
    elif ((opti_parameter == 'bed_h and shape at once') or
          (opti_parameter == 'bed_h and shape separeted')):
        opti_parameter_first_guess = 'bed_h and shape'
    else:
        raise ValueError('Unkown opti parameter!')

    print('\n- Get first guess: ')
    first_guess = get_first_guess(measurements,
                                  method=first_guess_method,
                                  bed_shape=bed_shape,
                                  const=first_guess_const,
                                  opti_parameter=opti_parameter_first_guess)
    print('---DONE---')

    # create Datalogger
    if use_datalogger:
        dl = DataLogger_bed_h_and_shape(
            true_bed=measurements['bed_unknown'],
            exact_surf=measurements['sfc_h'],
            ref_surf=measurements['sfc_h'],
            first_guessed_bed=first_guess['bed_h'],
            true_shape=measurements['shape_unknown'],
            first_guessed_shape=first_guess['shape'])
        dl.geometry = geometry
        dl.measurements = measurements

    # create cost function and define regularisation parameters
    if reg_parameters is None:
        print('\n- Calculate regularization parameters: ')
        reg_parameters = get_reg_parameters(opti_parameter_first_guess,
                                            measurements,
                                            geometry,
                                            mb_model,
                                            torch_type,
                                            bed_shape,
                                            first_guess,
                                            glacier_state,
                                            wanted_c_terms)
        print('---DONE---')

    # create cost function
    if opti_parameter == 'bed_h':
        bed_h = measurements['bed_known']
        shape = measurements['shape_all']
        first_guess_cost_fct = first_guess['bed_h']
    elif opti_parameter == 'shape':
        bed_h = measurements['bed_all']
        shape = measurements['shape_known']
        first_guess_cost_fct = first_guess['shape']
    elif ((opti_parameter == 'bed_h and shape separeted') or
          (opti_parameter == 'bed_h and shape at once')):
        bed_h = measurements['bed_known']
        shape = measurements['shape_known']
        first_guess_cost_fct = np.append(first_guess['bed_h'],
                                         first_guess['shape'])

    cost_fct = creat_cost_fct(
        bed_h=bed_h,
        shape=shape,
        spinup_surf=measurements['spinup_sfc'],
        reg_parameter=reg_parameters,
        ref_surf=measurements['sfc_h'],
        ref_width=measurements['widths'],
        ice_mask=measurements['ice_mask'],
        yrs_to_run=measurements['yrs_to_run'],
        dx=geometry['map_dx'],
        mb_model=mb_model,
        opti_var=opti_parameter_first_guess,
        torch_type=torch_type,
        used_geometry=bed_shape,
        data_logger=dl,
        grad_scaling=grad_scaling,
        grad_smoothing=grad_smoothing)

    print('\n- Start minimising:')
    res = minimize(fun=cost_fct,
                   x0=first_guess_cost_fct,
                   method='L-BFGS-B',
                   jac=True,
                   # bounds=bounds,
                   options=minimize_options,
                   callback=dl.callback_fct)

    if show_plot:
        result_plot = plot_result(res,
                                  measurements,
                                  geometry,
                                  first_guess,
                                  opti_parameter,
                                  bed_shape)

        return dl, res, result_plot

    return dl, res
