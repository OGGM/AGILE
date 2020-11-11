import numpy as np

from combine.core.cost_function import create_cost_fct
from scipy.optimize import minimize
from combine.core.idealized_experiments import define_geometry,\
    define_mb_model, create_measurements, get_first_guess, get_reg_parameters,\
    plot_result, get_spinup_sfc
from combine.core.data_logging import DataLogger


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
                             ice_mask=None
                             ):
    for loop in range(iterations):
        bed_h_success = False
        shape_success = False

        if data_logger is not None:
            data_logger.main_iterations.append(loop + 1)
            data_logger.main_iteration_callback()

        # create cost function for bed height
        shape_all = np.append(shape_guess, shape_known)

        cost_fct_bed_h = create_cost_fct(
            bed_h=bed_h_known,
            shape=shape_all,
            spinup_surf=spinup_surf,
            reg_parameter=reg_parameter_bed_h,
            ref_surf=ref_surf,
            ref_width=ref_widths,
            ice_mask=ice_mask,
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

        res_bed_h = minimize(fun=cost_fct_bed_h,
                             x0=bed_h_guess,
                             method='L-BFGS-B',
                             jac=True,
                             # bounds=bounds,
                             options=minimize_options_bed_h,
                             callback=data_logger.callback_fct)

        bed_h_guess = res_bed_h.x
        bed_h_success = res_bed_h.success

        if data_logger is not None:
            data_logger.current_step_indices_bed_h = []

        # create cost function for bed shape
        bed_h_all = np.append(bed_h_guess, bed_h_known)

        cost_fct_shape = create_cost_fct(
            bed_h=bed_h_all,
            shape=shape_known,
            spinup_surf=spinup_surf,
            reg_parameter=reg_parameter_shape,
            ref_surf=ref_surf,
            ref_width=ref_widths,
            ice_mask=ice_mask,
            yrs_to_run=yrs_to_run,
            dx=map_dx,
            mb_model=mb_model,
            opti_var='shape',
            torch_type=torch_type,
            used_geometry=used_geometry,
            data_logger=data_logger)

        res_shape = minimize(fun=cost_fct_shape,
                             x0=shape_guess,
                             method='L-BFGS-B',
                             jac=True,
                             # bounds=bounds,
                             options=minimize_options_shape,
                             callback=data_logger.callback_fct)

        shape_guess = res_shape.x
        shape_success = res_shape.success

        if data_logger is not None:
            data_logger.current_step_indices_shape = []

        if bed_h_success and shape_success:
            print('Optimisation was successful!')
            break

    res = [res_bed_h, res_shape]
    return data_logger, res


def idealized_inversion_experiment(used_bed_h_geometry='linear',
                                   used_along_glacier_geometry='constant',
                                   bed_geometry='rectangular',
                                   mb_opts={'ELA': np.array([3000.]),
                                            'grad': np.array([4.])},
                                   glacier_state='equilibrium',
                                   opti_parameter='bed_h',
                                   two_parameter_option=None, # separated, at once
                                   main_iterations_separeted=1,
                                   reg_parameters=None,
                                   wanted_c_terms=None,
                                   torch_type='double',
                                   minimize_options={'maxiter': 10,
                                                     'ftol': 1e-7,
                                                     'gtol': 1e-8,
                                                     'disp': True,
                                                     'maxcor': 50,
                                                     'maxls': 50},
                                   solver='L-BFGS-B',
                                   show_plot=True,
                                   minimize_options_spinup={'maxiter': 10,
                                                            'ftol': 1e-7,
                                                            'gtol': 1e-8,
                                                            'disp': True,
                                                            'maxcor': 50,
                                                            'maxls': 50}
                                   ):
    # define glacier bed geometry
    print('- Define geometry: ')
    geometry = define_geometry(used_bed_h_geometry,
                               used_along_glacier_geometry,
                               bed_geometry)
    print('---DONE---')

    # define mass balance profile
    print('\n- Define mass balance model: ')
    mb_model = define_mb_model(mb_opts)
    print('---DONE---')

    # create some measurements
    print('\n- Create Measurements: ')
    measurements = create_measurements(geometry,
                                       mb_model,
                                       bed_geometry=bed_geometry,
                                       glacier_state=glacier_state)

    if (glacier_state == 'retreating' or
       glacier_state == 'retreating with unknow spinup'):
        # use only second mass balance model for optimization
        mb_model = mb_model[1]
    print('---DONE---')

    print('\n- Get first guess: ')
    first_guess = get_first_guess(measurements,
                                  bed_geometry=bed_geometry,
                                  opti_parameter=opti_parameter)
    print('---DONE---')


    # TODO: This option is not tested or working
    if glacier_state == 'retreating with unknow spinup':
        print('\n- Calculate spinup surface:')
        measurements['spinup_sfc'] = get_spinup_sfc(measurements,
                                                    mb_model,
                                                    first_guess,
                                                    minimize_options_spinup,
                                                    bed_geometry,
                                                    geometry,
                                                    torch_type=torch_type)
        print('---DONE---')

    # define regularisation parameters
    if reg_parameters is None:
        print('\n- Calculate regularization parameters: ')
        reg_parameters = get_reg_parameters(opti_parameter,
                                            measurements,
                                            geometry,
                                            mb_model,
                                            bed_geometry,
                                            first_guess,
                                            torch_type,
                                            wanted_c_terms)
        print('---DONE---')

    # create Datalogger according to the used bed geometry and save some data,
    # Datalogger also checks if the inputs work together
    dl = DataLogger(
        bed_geometry=bed_geometry,
        opti_parameter=opti_parameter,
        two_parameter_option=two_parameter_option,
        main_iterations_separeted=main_iterations_separeted,
        geometry=geometry,
        measurements=measurements,
        first_guess=first_guess,
        reg_parameters=reg_parameters,
        used_bed_h_geometry=used_bed_h_geometry,
        used_along_glacier_geometry=used_along_glacier_geometry,
        minimize_options=minimize_options,
        solver=solver,
        glacier_state=glacier_state,
        mb_opts=mb_opts)

    # create an array with separareted optimisation variables if needed
    if dl.two_parameter_option == 'separated':
        if dl.opti_parameter == 'bed_h and bed_shape':
            opti_var_to_loop_through = ['bed_h', 'bed_shape']
        elif dl.opti_parameter == 'bed_h and w0':
            opti_var_to_loop_through = ['bed_h', 'w0']
        else:
            raise ValueError('Unknown opti_parameter for separated '
                             'optimisatoin')
    elif dl.opti_parameter in ['bed_h', 'bed_shape', 'bed_h and bed_shape',
                               'w0', 'bed_h and w0']:
        opti_var_to_loop_through = dl.opti_parameter
        # set the main iterations to 1, because only needed for separated
        # optimisation, in the other case the number of iterations is defined
        # in the minimize_options
        main_iterations_separeted = 1
    else:
        raise ValueError('Unknown opti_parameter!')

    # the next two for loops only has an effect for separated optimisation
    for loop in np.arange(main_iterations_separeted):
        dl.main_iterations = np.append(dl.main_iterations, loop + 1)
        dl.main_iteration_callback()

        for loop_opti_var in opti_var_to_loop_through:
            if dl.two_parameter_option == 'separated':
                # switch variables, consider first guess in first round
                # during first main loop use the first guess values
                if loop == 0:
                    if loop_opti_var == 'bed_h':
                        guess_parameter = dl.first_guessed_opti_var_1
                        known_parameter = dl.known_opti_var_1
                        geometry_var = np.append(dl.first_guessed_opti_var_2,
                                                 dl.known_opti_var_2)
                    else:
                        guess_parameter = dl.first_guessed_opti_var_2
                        known_parameter = dl.known_opti_var_2
                        geometry_var = np.append(dl.guessed_opti_var_1[-1],
                                                 dl.known_opti_var_1)
                # if not in first main loop use already guessed variables
                else:
                    if loop_opti_var == 'bed_h':
                        guess_parameter = dl.guessed_opti_var_1[-1]
                        known_parameter = dl.known_opti_var_1
                        geometry_var = np.append(dl.guessed_opti_var_2[-1],
                                                 dl.known_opti_var_2)
                    else:
                        guess_parameter = dl.guessed_opti_var_2[-1]
                        known_parameter = dl.known_opti_var_2
                        geometry_var = np.append(dl.guessed_opti_var_1[-1],
                                                 dl.known_opti_var_1)

            # here you have only one optimisation variable or two at once
            else:
                if loop_opti_var in ['bed_h', 'bed_shape', 'w0']:
                    guess_parameter = dl.first_guessed_opti_var_1
                    known_parameter = dl.known_opti_var_1
                    # here the second variable is selected (defined when
                    # creating the datalogger)
                    geometry_var = dl.geometry[dl.geometry_var]
                elif loop_opti_var in ['bed_h and bed_shape', 'bed_h and w0']:
                    guess_parameter = np.append(dl.first_guessed_opti_var_1,
                                                dl.first_guessed_opti_var_2)
                    known_parameter = np.append(dl.known_opti_var_1,
                                                dl.known_opti_var_2)
                    geometry_var = None
                else:
                    raise ValueError('Unknown optimisation variable!')

            cost_fct = create_cost_fct(
                known_parameter=known_parameter,
                geometry_var=geometry_var,
                bed_geometry=bed_geometry,
                measurements=measurements,
                reg_parameters=reg_parameters,
                dx=dl.geometry['map_dx'],
                mb_model=mb_model,
                opti_var=loop_opti_var,
                datalogger=dl)

            res = minimize(fun=cost_fct,
                           x0=guess_parameter,
                           method=solver,
                           jac=True,
                           # bounds=bounds,
                           options=minimize_options,
                           callback=dl.callback_fct)

    # filter out data used by minimize function for exploratory
    dl.filter_data_from_optimization()

    # save results to netcdf file
    dl.create_and_save_dataset()

    # show a plot if wanted
    if show_plot:
        plot_result(dl)

    '''
    # define initial values
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
    else:
        raise ValueError('Unknown optimisation parameter!')

    # start minimize
    if ((opti_parameter == 'bed_h') or (opti_parameter == 'shape') or
       (opti_parameter == 'bed_h and shape at once')):
        cost_fct = create_cost_fct(
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
            opti_var=opti_parameter,
            torch_type=torch_type,
            used_geometry=bed_geometry,
            data_logger=dl,
            grad_scaling=grad_scaling,
            grad_smoothing=grad_smoothing)

        print('\n- Start minimising:')
        res = minimize(fun=cost_fct,
                       x0=first_guess_cost_fct,
                       method=solver,
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
                                      bed_geometry)

            return dl, res, result_plot

        return dl, res

    elif opti_parameter == 'bed_h and shape separeted':
        dl, res = optimize_bed_h_and_shape(
            bed_h_guess=first_guess['bed_h'],
            shape_guess=first_guess['shape'],
            bed_h_known=bed_h,
            shape_known=shape,
            mb_model=mb_model,
            spinup_surf=measurements['spinup_sfc'],
            ref_surf=measurements['sfc_h'],
            ref_widths=measurements['widths'],
            yrs_to_run=measurements['yrs_to_run'],
            map_dx=geometry['map_dx'],
            minimize_options_bed_h=minimize_options,
            minimize_options_shape=minimize_options,
            reg_parameter_bed_h=reg_parameters,
            reg_parameter_shape=reg_parameters,
            torch_type=torch_type,
            used_geometry=bed_geometry,
            data_logger=dl,
            iterations=main_iterations_separeted,
            check_cost_terms=False,
            ice_mask=measurements['ice_mask']
            )

        print('---DONE---')

        if show_plot:
            result_plot = plot_result(res,
                                      measurements,
                                      geometry,
                                      first_guess,
                                      opti_parameter,
                                      bed_geometry)

            return dl, res, result_plot

        return dl, res'''
