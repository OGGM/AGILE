import numpy as np

from combine.core.cost_function import create_cost_fct
from scipy.optimize import minimize
from combine.core.idealized_experiments import define_geometry,\
    define_mb_model, create_measurements, get_first_guess, get_reg_parameters,\
    plot_result, get_spinup_sfc, first_guess_run, get_bounds
from combine.core.data_logging import DataLogger
import time


def idealized_inversion_experiment(used_bed_h_geometry='linear',
                                   used_along_glacier_geometry='constant',
                                   bed_geometry='rectangular',
                                   mb_opts={'ELA': np.array([3100., 3300.]),
                                            'grad': np.array([4., 4.])},
                                   glacier_state='equilibrium',
                                   opti_parameter='bed_h',
                                   # separated, at_once, calculated
                                   two_parameter_option='None',
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
                                   use_bounds=True,
                                   min_ice_h=0.01,  # in m
                                   max_ice_h=1000,  # in m
                                   min_w0=1.,  # in m
                                   solver='L-BFGS-B',
                                   save_plot=True,
                                   filename_suffix='',
                                   minimize_options_spinup={'maxiter': 10,
                                                            'ftol': 1e-7,
                                                            'gtol': 1e-8,
                                                            'disp': True,
                                                            'maxcor': 50,
                                                            'maxls': 50},
                                   years_to_run=None,  # needed for equ exp. (how long should equilibrium run should last)
                                   job_id=0,  # only needed for cluster
                                   task_id=0  # only needed for cluster
                                   ):
    # define glacier bed geometry
    print('- Define geometry: ')
    geometry = define_geometry(used_bed_h_geometry,
                               used_along_glacier_geometry,
                               bed_geometry)
    print('\n    ---DONE---')

    # define mass balance profile
    print('\n- Define mass balance model: ')
    mb_model = define_mb_model(mb_opts)
    print('\n    ---DONE---')

    # create some measurements
    print('\n- Create Measurements: ')
    measurements = create_measurements(geometry,
                                       mb_model,
                                       bed_geometry=bed_geometry,
                                       glacier_state=glacier_state)

    if glacier_state in ['retreating', 'retreating with unknow spinup',
                         'equilibrium']:
        # use only second mass balance model for optimization
        mb_model = mb_model[1]
    elif glacier_state in ['advancing']:
        mb_model = mb_model[0]

    # only needed for different years for equilibrium experiment
    if years_to_run is not None:
        measurements['yrs_to_run'] = years_to_run

    print('\n    ---DONE---')

    print('\n- Get first guess: ')
    first_guess = get_first_guess(measurements,
                                  bed_geometry=bed_geometry,
                                  opti_parameter=opti_parameter,
                                  job_id=job_id,
                                  task_id=task_id)
    print('\n    ---DONE---')

    print('\n- First guess run for fg_sfc_h and fg_widths')
    fg_sfc_h, fg_widths = first_guess_run(first_guess,
                                          bed_geometry,
                                          measurements,
                                          mb_model,
                                          geometry,
                                          opti_parameter=opti_parameter)
    print('\n    ---DONE---')

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
        print('\n    ---DONE---')

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
        print('\n    ---DONE---')

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
        fg_sfc_h=fg_sfc_h,
        fg_widths=fg_widths,
        reg_parameters=reg_parameters,
        used_bed_h_geometry=used_bed_h_geometry,
        used_along_glacier_geometry=used_along_glacier_geometry,
        minimize_options=minimize_options,
        solver=solver,
        glacier_state=glacier_state,
        mb_opts=mb_opts,
        filename_suffix=filename_suffix,
        task_id=task_id)

    print('\n- Start minimising (start timer):')

    # save start time
    dl.start_time = time.time()

    # create an array with separareted optimisation variables if needed
    if (dl.two_parameter_option == 'separated') & (dl.opti_var_2 is not None):
        if dl.opti_parameter == 'bed_h and bed_shape':
            opti_var_to_loop_through = ['bed_h', 'bed_shape']
        elif dl.opti_parameter == 'bed_h and w0':
            opti_var_to_loop_through = ['bed_h', 'w0']
        else:
            raise ValueError('Unknown opti_parameter for separated '
                             'optimisatoin')
    elif dl.opti_parameter in ['bed_h', 'bed_shape', 'bed_h and bed_shape',
                               'w0', 'bed_h and w0']:
        opti_var_to_loop_through = [dl.opti_parameter]
        # set the main iterations to 1, because only needed for separated
        # optimisation, in the other case the number of iterations is defined
        # in the minimize_options
        main_iterations_separeted = 1
    else:
        raise ValueError('Unknown opti_parameter!')

    # calculate control variable boundaries if wanted
    if use_bounds is True:
        sfc_h_ice = measurements['sfc_h'][measurements['ice_mask']]
        widths_ice = measurements['widths'][measurements['ice_mask']]
        bounds = {}
        if dl.two_parameter_option == 'separated':
            bounds['bed_h'] = get_bounds('bed_h',
                                         measurements=sfc_h_ice,
                                         min_value=min_ice_h,
                                         max_value=max_ice_h)
            if dl.opti_parameter == 'bed_h and bed_shape':
                bounds['shape_var'] = get_bounds('bed_shape',
                                                 measurements=sfc_h_ice)
            elif dl.opti_parameter == 'bed_h and w0':
                bounds['shape_var'] = get_bounds('w0',
                                                 measurements=widths_ice,
                                                 min_value=min_w0)
        else:
            if dl.opti_parameter in ['bed_h', 'bed_h and bed_shape',
                                     'bed_h and w0']:
                bounds['bed_h'] = get_bounds('bed_h',
                                             measurements=sfc_h_ice,
                                             min_value=min_ice_h,
                                             max_value=max_ice_h)
            if dl.opti_parameter in ['bed_shape', 'bed_h and bed_shape']:
                bounds['bed_shape'] = get_bounds('bed_shape',
                                                 measurements=sfc_h_ice)
                bounds['shape_var'] = get_bounds('bed_shape',
                                                 measurements=sfc_h_ice)
            if dl.opti_parameter in ['w0', 'bed_h and w0']:
                bounds['w0'] = get_bounds('w0',
                                          measurements=widths_ice,
                                          min_value=min_w0)
                bounds['shape_var'] = get_bounds('w0',
                                                 measurements=widths_ice,
                                                 min_value=min_w0)
            if dl.opti_parameter in ['bed_h and bed_shape', 'bed_h and w0']:
                bounds['two_opti_var'] = bounds['bed_h'] + bounds['shape_var']
    else:
        minimize_bounds = None

    # the next two for loops only has an effect for separated optimisation
    for loop in np.arange(main_iterations_separeted):
        dl.main_iterations = np.append(dl.main_iterations, loop + 1)
        dl.main_iteration_callback()

        # variables to check if minimise was succesfull for separated
        # optimisation
        success_opti_var_1 = False
        success_opti_var_2 = False

        for loop_opti_var in opti_var_to_loop_through:
            if (dl.two_parameter_option == 'separated') & \
               (dl.opti_var_2 is not None):
                # switch variables, consider first guess in first round
                # during first main loop use the first guess values
                if loop == 0:
                    if loop_opti_var == 'bed_h':
                        guess_parameter = dl.first_guessed_opti_var_1
                        known_parameter = dl.known_opti_var_1
                        geometry_var = np.append(dl.first_guessed_opti_var_2,
                                                 dl.known_opti_var_2)
                        if use_bounds is True:
                            minimize_bounds = bounds['bed_h']
                    else:
                        guess_parameter = dl.first_guessed_opti_var_2
                        known_parameter = dl.known_opti_var_2
                        geometry_var = np.append(dl.guessed_opti_var_1[-1],
                                                 dl.known_opti_var_1)
                        if use_bounds is True:
                            minimize_bounds = bounds['shape_var']
                # if not in first main loop use already guessed variables
                else:
                    if loop_opti_var == 'bed_h':
                        guess_parameter = dl.guessed_opti_var_1[-1]
                        known_parameter = dl.known_opti_var_1
                        geometry_var = np.append(dl.guessed_opti_var_2[-1],
                                                 dl.known_opti_var_2)
                        if use_bounds is True:
                            minimize_bounds = bounds['bed_h']
                    else:
                        guess_parameter = dl.guessed_opti_var_2[-1]
                        known_parameter = dl.known_opti_var_2
                        geometry_var = np.append(dl.guessed_opti_var_1[-1],
                                                 dl.known_opti_var_1)
                        if use_bounds is True:
                            minimize_bounds = bounds['shape_var']

            # here you have only one optimisation variable or
            # two at_once/calculated
            else:
                if loop_opti_var in ['bed_h', 'bed_shape', 'w0']:
                    guess_parameter = dl.first_guessed_opti_var_1
                    known_parameter = dl.known_opti_var_1
                    # here the second variable is selected (defined when
                    # creating the datalogger)
                    geometry_var = dl.geometry[dl.geometry_var]
                    if use_bounds is True:
                        minimize_bounds = bounds[loop_opti_var]
                elif (
                     (loop_opti_var in ['bed_h and bed_shape', 'bed_h and w0'])
                      & (dl.two_parameter_option == 'at_once')
                     ):
                    guess_parameter = np.append(dl.first_guessed_opti_var_1,
                                                dl.first_guessed_opti_var_2)
                    known_parameter = np.append(dl.known_opti_var_1,
                                                dl.known_opti_var_2)
                    geometry_var = None
                    if use_bounds is True:
                        minimize_bounds = bounds['two_opti_var']
                elif (
                     (loop_opti_var in ['bed_h and bed_shape', 'bed_h and w0'])
                      & (dl.two_parameter_option == 'calculated')
                     ):
                    guess_parameter = dl.first_guessed_opti_var_1
                    known_parameter = dl.known_opti_var_1
                    geometry_var = dl.known_opti_var_2
                    if use_bounds is True:
                        minimize_bounds = bounds['bed_h']
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
                two_parameter_option=dl.two_parameter_option,
                datalogger=dl)

            res = minimize(fun=cost_fct,
                           x0=guess_parameter,
                           method=solver,
                           jac=True,
                           bounds=minimize_bounds,
                           options=minimize_options,
                           callback=dl.callback_fct)

            # checking success for separated optimisation
            if (dl.two_parameter_option == 'separated') & \
               (dl.opti_var_2 is not None):
                # status = 0 is success, status = 2 no further minimisation
                # possible (e.g. wrong line search direction)
                if res.status in [0, 2]:
                    if loop_opti_var == 'bed_h':
                        success_opti_var_1 = True
                    else:
                        success_opti_var_2 = True
                if loop_opti_var == 'bed_h':
                    message_opti_var_1 = res.message
                else:
                    message_opti_var_2 = res.message
            elif loop_opti_var in ['bed_h', 'bed_shape', 'w0']:
                message_opti_var_1 = res.message
            elif loop_opti_var in ['bed_h and bed_shape', 'bed_h and w0']:
                message_opti_var_1 = res.message
                message_opti_var_2 = res.message

        # now in mainiteration for loop check if separated optimisation was
        # successfull for both variables and exit loop if so
        if success_opti_var_1 & success_opti_var_2:
            break

    end_time = time.time()
    print('\n    ---Done (stop timer) ---')

    print('\n- Create Dataset and save as NetCDF data')

    # add minimize message to datalogger
    dl.message_opti_var_1 = str(message_opti_var_1)
    if dl.opti_var_2 is not None:
        dl.message_opti_var_2 = str(message_opti_var_2)
    dl.total_computing_time = end_time - dl.start_time
    # filter out data used by minimize function for exploratory
    dl.filter_data_from_optimization()

    # save results to netcdf file
    dl.create_and_save_dataset()

    print('\n    ---Done---')

    # show a plot if wanted
    if save_plot:
        print('\n- Create and save figure')
        dl.save_result_plot()
        print('\n    ---Done---')

    print('\n-----COMBINE finished-----')

    return dl
