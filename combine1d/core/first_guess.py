import copy

import numpy as np
from oggm.core.flowline import FluxBasedModel
from oggm.core.massbalance import MultipleFlowlineMassBalance, ConstantMassBalance


def get_first_guess(data_logger):
    fl = data_logger.flowline_init
    parameter_indices = data_logger.parameter_indices
    ice_mask = data_logger.ice_mask
    is_trapezoid = data_logger.is_trapezoid

    first_guess = np.empty(data_logger.len_unknown_parameter)
    first_guess[:] = np.nan

    for ind in parameter_indices:
        if ind in ['bed_h']:
            ind_first_guess = getattr(fl, ind)[ice_mask]
        elif ind in ['area_bed_h']:
            ind_first_guess = fl.bed_h[ice_mask] * fl.widths_m[ice_mask]
        elif ind in ['w0_m', 'lambdas']:
            ind_first_guess = getattr(fl, '_' + ind)[(is_trapezoid & ice_mask)]
        elif ind in ['surface_h']:
            ind_first_guess = get_first_guess_surface_h(data_logger)
        elif ind in ['height_shift_spinup']:
            ind_first_guess = \
                data_logger.spinup_options['height_shift']['mb_model']['fg_height_shift']
        elif ind in ['section']:
            extra_grid_points =\
                data_logger.spinup_options['section']['extra_grid_points']
            nx = sum(ice_mask)
            ind_first_guess = fl.section[:nx + extra_grid_points]
        else:
            raise NotImplementedError(f'{ind} is not implemented!')

        first_guess[parameter_indices[ind]] = ind_first_guess

    # scale first guess according to control var min/max scaling
    min_bound, max_bound = list(zip(*data_logger.bounds))
    min_bound = np.array(min_bound)
    max_bound = np.array(max_bound)
    scale = data_logger.control_vars_characteristic_scale
    first_guess = (first_guess - min_bound) / (max_bound - min_bound) * scale

    return first_guess


def get_first_guess_surface_h(data_logger):
    """TODO: Function to conduct spinup for first guess surface_h"""
    fl = data_logger.flowline_init

    if 'surface_h' in list(data_logger.spinup_options.keys()):
        mb_options = data_logger.spinup_options['surface_h']['mb_model']
        if mb_options['type'] == 'constant':
            yr_start_run = mb_options['years'][0]
            yr_end_run = mb_options['years'][1]
            halfsize = (yr_end_run - yr_start_run) / 2
            mb_spinup = MultipleFlowlineMassBalance(data_logger.gdir,
                                                    fls=[fl],
                                                    mb_model_class=ConstantMassBalance,
                                                    filename='climate_historical',
                                                    input_filesuffix='',
                                                    y0=yr_start_run + halfsize,
                                                    halfsize=halfsize)
            mb_spinup.temp_bias = mb_options['t_bias']

            model = FluxBasedModel(copy.deepcopy([fl]),
                                   mb_spinup,
                                   y0=yr_start_run)
            model.run_until(yr_end_run)

            return model.fls[0].surface_h
        else:
            raise NotImplementedError(f'mb type {mb_options["type"]} not implemented!')

    else:
        raise NotImplementedError('The provided spinup option is not implemented!')
