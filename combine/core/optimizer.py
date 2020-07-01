import numpy as np

from combine.core.cost_function import creat_cost_fct
from scipy.optimize import minimize


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
            data_logger.main_iterations.append(loop)
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

        res = minimize(fun=cost_fct_bed_h,
                       x0=bed_h_guess,
                       method='L-BFGS-B',
                       jac=True,
                       # bounds=bounds,
                       options=minimize_options_bed_h,
                       callback=callback)

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

        res = minimize(fun=cost_fct_shape,
                       x0=shape_guess,
                       method='L-BFGS-B',
                       jac=True,
                       # bounds=bounds,
                       options=minimize_options_shape,
                       callback=callback)

        shape_guess = res.x
        shape_success = res.success

        if data_logger is not None:
            data_logger.current_step_indices_shape = []

        if bed_h_success and shape_success:
            print('Optimisation was successful!')
            break
