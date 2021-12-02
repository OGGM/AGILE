import numpy as np

from combine1d.core.cost_function import create_cost_fct
from scipy.optimize import minimize
from combine1d.sandbox.idealized_experiments_help_functions import \
    define_geometry, define_mb_model, create_measurements, get_reg_parameters,\
    get_bounds
from combine1d.core.first_guess import get_first_guess, first_guess_run
from combine1d.core.data_logging import DataLogger
from combine1d.core.exception import MaxCalculationTimeReached
import time


def idealized_inversion_experiment(used_bed_h_geometry='linear',
                                   used_along_glacier_geometry='constant',
                                   bed_geometry='rectangular',
                                   mb_opts={'ELA': np.array([3100., 3300.]),
                                            'grad': np.array([4., 4.])},
                                   glacier_state='equilibrium',
                                   spinup_yrs=200,
                                   opti_parameter='bed_h',
                                   two_parameter_option='None',
                                   main_iterations_iterative=1,
                                   reg_parameters=None,
                                   wanted_c_terms=None,
                                   torch_type='double',
                                   minimize_options={'maxiter': 10,
                                                     'ftol': 1e-7,
                                                     'gtol': 1e-8,
                                                     'disp': True,
                                                     'maxcor': 50,
                                                     'maxls': 50},
                                   max_time_minimize=None,  # in s
                                   use_bounds=True,
                                   min_ice_h=0.01,  # in m
                                   max_ice_h=1000,  # in m
                                   min_w0=10.,  # in m
                                   min_bed_shape=0.001,
                                   grad_scaling={'bed_h': 1,
                                                 'shape_var': 1},
                                   solver='L-BFGS-B',
                                   save_plot=False,
                                   filename_suffix='',
                                   # minimize_options_spinup={'maxiter': 10,
                                   #                         'ftol': 1e-7,
                                   #                         'gtol': 1e-8,
                                   #                         'disp': True,
                                   #                         'maxcor': 50,
                                   #                         'maxls': 50},
                                   years_to_run=None,
                                   job_id=0,
                                   task_id=0):
    '''
    Function to conduct idealized inversion experiments with COMBINE1D.

    Parameters
    ----------
    used_bed_h_geometry: string, optional
        Defines the geometry of the glacier bed height.
        Options: 'linear', 'cliff' or 'random'
        The default is 'linear'.
    used_along_glacier_geometry: string, optional
        Defines the geometry along the glacier.
        Options: 'constant' or 'random'
        The default is 'constant'.
    bed_geometry: string, optional
        Defines the shape of the glacier.
        Options: 'rectangular', 'parabolic' or 'trapezoidal'.
        The default is 'rectangular'.
    mb_opts: dict of ELAs and grads, optional
        ELA defines the equilibrium line altitude and grad defines the
        gradient of the mass-balance-profile. The length of 'ELA' and 'grad'
        must be the same.
        The default is {'ELA': np.array([3100., 3300.]),
                        'grad': np.array([4., 4.])}.
    glacier_state: string, optional
        Defines the glacier state at the end, the options are: 'equilibrium',
        'advancing' or 'retreating'. For 'equilibrium' the glacier is runs
        until it reaches equilibrium, for 'advancing' the glacier runs half the
        time before it reaches equilibrium and in the case of 'retreating'
        the glacier runs until equilibrium with the first MassBalanceModel and
        then runs half the time until it reaches the equilibrium with the
        second MassBalanceModel, defined with 'mb_opts'. Or
        'retreating with unknow spinup' starts an experiment with an adaptive
        glacier spinup. Initial glacier spinup time for measurements creation
        is defined with 'spinup_yrs'.
        The default is 'equilibrium'.
    spinup_yrs: int, optional
        If glaicer state is 'retreating with unknow spinup' this defines the
        length of the glacier spinup for the measurements creation
    opti_parameter: str, optional
        Defines the optimisation parameter. Depending on the bed geometry this
        could be one ore two.
        Options for 'rectangular': 'bed_h'.
        Options for 'parabolic': 'bed_h', 'bed_shape' or 'bed_h and bed_shape'
        Options for 'trapezoidal': 'bed_h', 'w0' or 'bed_h and w0'
    two_parameter_option: str, optional
        If bed_geometry is parabolic or trapezoidal choose between 'explicit',
        'implicit' or 'iterative'.
        The default is 'None'.
    main_iterations_iterative: int, optional
        If two_parameter_option is 'iterative' this defines the number of
        minimisation iterations before the control varialbe is switched.
        The default is 1.
    reg_parameters: :py:class:`numpy.ndarray`, optional
        Define the regularisation parameters for the individual cost function
        terms. Must define four parameters for to the four cost function terms.
        In the thesis the first two are always 1 and have no own variable, the
        remaining two are called lambda_0 and lambda_1. If this is None the
        regularisation parameters will be calculated in a way that all cost
        function terms have the same order of magnitude (see also
        wanted_c_terms).
        The default is None.
    wanted_c_terms: :py:class:`numpy.ndarray`, optional
        Can scale the individual cost terms,
        e.g. wanted_c_terms=[1., 10., 1., 1.] equals that the magnitude of the
        second cost term is 1/10 of the other terms. The default is None.
    torch_type: str, optional
        Defines type for torch.Tensor. If 'double' use torch.double, otherwise
        use torch.float.
        The default is 'double'.
    minimize_options: dict, optional
        Options for the scipy minimisation algorithm.
        The default is {'maxiter': 10, 'ftol': 1e-7, 'gtol': 1e-8,
                        'disp': True, 'maxcor': 50, 'maxls': 50}
    max_time_minimize: int, optional
        Can define a maximum calculation time in s after which the inversion is
        stopped and the last iteration is saved.
        The default is None.
    use_bounds: bool, optional
        Defines if bounds for the control varialbes should be used during the
        inversion.
        The default is True.
    min_ice_h: float, optional
        Defines the minimum ice thickness in m. Only used if use_bounds is
        True.
        The default is 0.01
    max_ice_h:float, optional
        Defines the maximum ice thickness in m. Only used if use_bounds is
        True.
        The default is 1000
    min_w0: float, optional
        Defines the minimum bottom width in m for a trapezoidal bed_geometry.
        Only used if use_bounds is True.
        The default is 10.
    min_bed_shape:float, optional
        Defines the minimum shape factor for a parabolic bed_geometry. Only
        used if use_bounds is True.
        The default is 0.001.
    grad_scaling: dict, optional
        Can define a scaling of the gradients if the explicit
        two_parameter_option is used.
        The default is {'bed_h': 1, 'shape_var': 1}.
    solver: str, optional
        Defines the used minimisation algorithm of scipy.optimize.minimize.
        The default is 'L-BFGS-B', not tested with other solvers.
    save_plot: bool, optional
        If a first result plot should be saved after the experiment.
        The default is False.
    filename_suffix: str, optional
        Can add a suffix to the resulting experiment file.
        The default is ''.
    years_to_run: int, optional
        If the used glacier state is equilibrium here you can define the run
        time of the forward model run. If None the same run time is used as the
        forward model needed for the creation of the measurements.
        The default is None.
    job_id: int, optional
        Id of the slurm job, only needed when computing experiment on cluster.
        The default is 0.
    task_id: int, optional
        Id of the slurm task, only needed when compuntin experiment on cluster.
        The default is 0.
    '''
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

    if glacier_state in ['retreating', 'equilibrium']:
        # use only second mass balance model for optimization
        mb_model = mb_model[1]
    elif glacier_state in ['advancing']:
        mb_model = mb_model[0]
    elif glacier_state == 'retreating with unknow spinup':
        mb_model = {'model_known': mb_model[1],
                    'grad_spinup': mb_opts['grad'][0]}

    # only needed for different years for equilibrium experiment
    if years_to_run is not None:
        measurements['yrs_to_run'] = years_to_run

    print('\n    ---DONE---')

    print('\n- Get first guess: ')
    first_guess = get_first_guess(measurements,
                                  bed_geometry=bed_geometry,
                                  opti_parameter=opti_parameter,
                                  glacier_state=glacier_state,
                                  job_id=job_id,
                                  task_id=task_id)
    print('\n    ---DONE---')

    print('\n- First guess run for fg_sfc_h and fg_widths')
    fg_run = first_guess_run(first_guess,
                             bed_geometry,
                             measurements,
                             mb_model,
                             geometry,
                             glacier_state=glacier_state,
                             spinup_yrs=spinup_yrs,
                             opti_parameter=opti_parameter)
    print('\n    ---DONE---')

    # define regularisation parameters
    if reg_parameters is None:
        print('\n- Calculate regularization parameters: ')
        reg_parameters = get_reg_parameters(opti_parameter,
                                            measurements,
                                            geometry,
                                            mb_model,
                                            glacier_state,
                                            bed_geometry,
                                            first_guess,
                                            spinup_yrs,
                                            torch_type,
                                            wanted_c_terms)
        print('\n    ---DONE---')

    # create Datalogger according to the used bed geometry and save some data,
    # Datalogger also checks if the inputs work together
    dl = DataLogger(
        bed_geometry=bed_geometry,
        opti_parameter=opti_parameter,
        two_parameter_option=two_parameter_option,
        main_iterations_iterative=main_iterations_iterative,
        geometry=geometry,
        measurements=measurements,
        first_guess=first_guess,
        fg_run=fg_run,
        reg_parameters=reg_parameters,
        used_bed_h_geometry=used_bed_h_geometry,
        used_along_glacier_geometry=used_along_glacier_geometry,
        minimize_options=minimize_options,
        max_time_minimize=max_time_minimize,
        solver=solver,
        glacier_state=glacier_state,
        mb_opts=mb_opts,
        grad_scaling=grad_scaling,
        filename_suffix=filename_suffix,
        task_id=task_id)

    if glacier_state == 'retreating with unknow spinup':
        dl.spinup_yrs = spinup_yrs

    print('\n- Start minimising (start timer):')

    # save start time
    dl.start_time = time.time()

    # create an array with separareted optimisation variables if needed
    if (dl.two_parameter_option == 'iterative') & (dl.opti_var_2 is not None):
        if dl.opti_parameter == 'bed_h and bed_shape':
            opti_var_to_loop_through = ['bed_h', 'bed_shape']
        elif dl.opti_parameter == 'bed_h and w0':
            opti_var_to_loop_through = ['bed_h', 'w0']
        else:
            raise ValueError('Unknown opti_parameter for iterative '
                             'optimisatoin')
    elif dl.opti_parameter in ['bed_h', 'bed_shape', 'bed_h and bed_shape',
                               'w0', 'bed_h and w0']:
        opti_var_to_loop_through = [dl.opti_parameter]
        # set the main iterations to 1, because only needed for iterative
        # optimisation, in the other case the number of iterations is defined
        # in the minimize_options
        main_iterations_iterative = 1
    else:
        raise ValueError('Unknown opti_parameter!')

    # calculate control variable boundaries if wanted
    if use_bounds is True:
        sfc_h_ice = measurements['sfc_h'][measurements['ice_mask']]
        widths_ice = measurements['widths'][measurements['ice_mask']]
        bounds = {}
        if dl.two_parameter_option == 'iterative':
            bounds['bed_h'] = get_bounds('bed_h',
                                         measurements=sfc_h_ice,
                                         min_value=min_ice_h,
                                         max_value=max_ice_h)
            if dl.opti_parameter == 'bed_h and bed_shape':
                bounds['shape_var'] = get_bounds('bed_shape',
                                                 measurements=sfc_h_ice,
                                                 min_value=min_bed_shape)
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
                                                 measurements=sfc_h_ice,
                                                 min_value=min_bed_shape)
                bounds['shape_var'] = get_bounds('bed_shape',
                                                 measurements=sfc_h_ice,
                                                 min_value=min_bed_shape)
            if dl.opti_parameter in ['w0', 'bed_h and w0']:
                bounds['w0'] = get_bounds('w0',
                                          measurements=widths_ice,
                                          min_value=min_w0)
                bounds['shape_var'] = get_bounds('w0',
                                                 measurements=widths_ice,
                                                 min_value=min_w0)
            if dl.opti_parameter in ['bed_h and bed_shape', 'bed_h and w0']:
                bounds['two_opti_var'] = bounds['bed_h'] + bounds['shape_var']

        if glacier_state == 'retreating with unknow spinup':
            bounds['spinup_ELA'] = [(None, first_guess['max_spinup_ELA'])]
    else:
        minimize_bounds = None

    # the next two for loops only has an effect for iterative optimisation
    for loop in np.arange(main_iterations_iterative):
        dl.main_iterations = np.append(dl.main_iterations, loop + 1)
        dl.main_iteration_callback()

        # variables to check if minimise was succesfull for iterative
        # optimisation
        success_opti_var_1 = False
        success_opti_var_2 = False

        for loop_opti_var in opti_var_to_loop_through:
            if (dl.two_parameter_option == 'iterative') & \
               (dl.opti_var_2 is not None):
                # switch variables, consider first guess in first round
                # during first main loop use the first guess values
                if loop == 0:
                    if loop_opti_var == 'bed_h':
                        guess_parameter = dl.first_guessed_opti_var_1
                        known_parameter = dl.known_opti_var_1

                        geometry_var = np.zeros(len(measurements['ice_mask']))
                        geometry_var[measurements['ice_mask']] = \
                            dl.first_guessed_opti_var_2
                        geometry_var[~measurements['ice_mask']] = \
                            dl.known_opti_var_2

                        if use_bounds is True:
                            minimize_bounds = bounds['bed_h']
                    else:
                        guess_parameter = dl.first_guessed_opti_var_2
                        known_parameter = dl.known_opti_var_2

                        geometry_var = np.zeros(len(measurements['ice_mask']))
                        geometry_var[measurements['ice_mask']] = \
                            dl.guessed_opti_var_1[-1]
                        geometry_var[~measurements['ice_mask']] = \
                            dl.known_opti_var_1

                        if use_bounds is True:
                            minimize_bounds = bounds['shape_var']
                # if not in first main loop use already guessed variables
                else:
                    if loop_opti_var == 'bed_h':
                        guess_parameter = dl.guessed_opti_var_1[-1]
                        known_parameter = dl.known_opti_var_1

                        geometry_var = np.zeros(len(measurements['ice_mask']))
                        geometry_var[measurements['ice_mask']] = \
                            dl.guessed_opti_var_2[-1]
                        geometry_var[~measurements['ice_mask']] = \
                            dl.known_opti_var_2

                        if use_bounds is True:
                            minimize_bounds = bounds['bed_h']
                    else:
                        guess_parameter = dl.guessed_opti_var_2[-1]
                        known_parameter = dl.known_opti_var_2

                        geometry_var = np.zeros(len(measurements['ice_mask']))
                        geometry_var[measurements['ice_mask']] = \
                            dl.guessed_opti_var_1[-1]
                        geometry_var[~measurements['ice_mask']] = \
                            dl.known_opti_var_1

                        if use_bounds is True:
                            minimize_bounds = bounds['shape_var']

            # here you have only one optimisation variable or
            # two explicit/implicit
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
                      & (dl.two_parameter_option == 'explicit')
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
                      & (dl.two_parameter_option == 'implicit')
                     ):
                    guess_parameter = dl.first_guessed_opti_var_1
                    known_parameter = dl.known_opti_var_1
                    geometry_var = dl.known_opti_var_2
                    if use_bounds is True:
                        minimize_bounds = bounds['bed_h']
                else:
                    raise ValueError('Unknown optimisation variable!')

            spinup_sfc_known = True
            if glacier_state == 'retreating with unknow spinup':
                spinup_sfc_known = False
                # include ELA optimisation with bed_h optimisation
                if (dl.two_parameter_option == 'iterative') & \
                   (dl.opti_var_2 is not None):
                    if loop_opti_var == 'bed_h':
                        # in first main iteration use first guess
                        if loop == 0:
                            guess_parameter = \
                                np.append(dl.first_guess['spinup_ELA'],
                                          guess_parameter)
                        else:
                            guess_parameter = \
                                np.append(dl.spinup_ELA_guessed[-1],
                                          guess_parameter)
                        if use_bounds is True:
                            minimize_bounds = bounds['spinup_ELA'] + \
                                minimize_bounds

                else:
                    guess_parameter = np.append(dl.first_guess['spinup_ELA'],
                                                guess_parameter)
                    if use_bounds is True:
                        minimize_bounds = bounds['spinup_ELA'] + \
                            minimize_bounds

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
                datalogger=dl,
                grad_scaling=grad_scaling,
                min_w0=min_w0,
                spinup_sfc_known=spinup_sfc_known,
                spinup_yrs=spinup_yrs)

            try:
                res = minimize(fun=cost_fct,
                               x0=guess_parameter,
                               method=solver,
                               jac=True,
                               bounds=minimize_bounds,
                               options=minimize_options,
                               callback=dl.callback_fct)
                minimize_message = res.message
                minimize_status = res.status
            except MaxCalculationTimeReached:
                minimize_message = 'Maximum calculation time reached!'
                minimize_status = 'max calc time reached'

            # checking success for iterative optimisation
            if (dl.two_parameter_option == 'iterative') & \
               (dl.opti_var_2 is not None):
                # is needed if max calc time is reached in first main iteration
                message_opti_var_2 = ''

                if loop_opti_var == 'bed_h':
                    message_opti_var_1 = minimize_message
                else:
                    message_opti_var_2 = minimize_message
                # status = 0 is success, status = 2 no further minimisation
                # possible (e.g. wrong line search direction)
                if minimize_status in [0, 2]:
                    if loop_opti_var == 'bed_h':
                        success_opti_var_1 = True
                    else:
                        success_opti_var_2 = True
                elif minimize_status == 'max calc time reached':
                    success_opti_var_1 = True
                    success_opti_var_2 = True
                    break

            elif loop_opti_var in ['bed_h', 'bed_shape', 'w0']:
                message_opti_var_1 = minimize_message

            elif loop_opti_var in ['bed_h and bed_shape', 'bed_h and w0']:
                message_opti_var_1 = minimize_message
                message_opti_var_2 = minimize_message

        # now in mainiteration for loop check if iterative optimisation was
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
