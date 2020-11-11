import numpy as np
import salem
import torch

from combine.core.dynamics import run_forward_core
from combine.core.dynamics import run_flowline_forward_core
from combine.core.arithmetics import to_torch_tensor, magnitude


def create_cost_func(gdir, data_logger=None, surface_noise=None,
                     bed_measurements=None):
    """
    Creates a cost function based on the glacier directory.

    Parameters
    ----------
    gdir: NonRGIGlacierDirectory
        GlacierDirectory containing precomputed spinup surface and
        "observed" surface for final state. Further on contains inversion
        settings
    data_logger: DataLogger
        optionally logs data

    Returns
    -------
    tuple of (cost, grad) with cost as float and grad being a ndarray
    with same shape as b
    """

    # precompute known data to avoid recomputation during each call of
    # cost_fucntion
    conv_filter = torch.ones((1, 1, 3, 3), requires_grad=False)
    # TODO: think about whether cross is better suited (in forward model no diagonal transport
    # conv_filter = torch.tensor([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]],
    #                           dtype=torch.float, requires_grad=True)
    spinup_surf = salem.GeoTiff(gdir.get_filepath('spinup_dem')).get_vardata()
    ref_surf = salem.GeoTiff(gdir.get_filepath('ref_dem')).get_vardata()
    if surface_noise is not None:
        spinup_surf += surface_noise
        ref_surf += surface_noise
        # TODO: allow for independent surface perturbations

    gpr = None
    if bed_measurements is not None:
        # PyTorch is a bit messy with masks.
        # Instead we use full tensors and multiply by a mask.
        gpr_data = torch.tensor(np.ma.filled(bed_measurements, -9999),
                                dtype=torch.float,
                                requires_grad=False)
        gpr_mask = torch.tensor(1 - bed_measurements.mask,
                                dtype=torch.float, requires_grad=False)
        gpr = (gpr_data, gpr_mask)

    spinup_surf = torch.tensor(spinup_surf, dtype=torch.float,
                               requires_grad=False)
    ref_surf = torch.tensor(ref_surf, dtype=torch.float,
                            requires_grad=False)
    ref_ice_mask = np.load(gdir.get_filepath('ref_ice_mask'))
    ref_ice_mask = torch.tensor(ref_ice_mask.astype(np.int),
                                dtype=torch.float, requires_grad=False)
    ref_inner_mask = torch.zeros(ref_ice_mask.shape)
    ref_inner_mask[1:-1, 1:-1] = torch.conv2d(
        ref_ice_mask.unsqueeze(0).unsqueeze(0), conv_filter) == 9

    inv_settings = gdir.inversion_settings
    reg_parameters = inv_settings['reg_parameters']
    yrs_to_run = inv_settings['yrs_forward_run']
    case = inv_settings['case']
    mb = inv_settings['mb_forward_run']
    if mb is None:
        mb = case.get_mb_model()

    def c_fun(b):
        """
        Wrapper for cost_function. First step for easy exchangeability
        afterwards and to get a cost_function signature exhibiting all true
        input arguments.

        Parameters
        ----------
        b: ndarray
            bed heights for which the costs should be calculated. (unit: [m])

        Returns
        -------
        tuple of (cost, grad) with cost as float and grad being a ndarray
        with same shape as b
        """
        return cost_function(b, reg_parameters, ref_surf, ref_ice_mask,
                             ref_inner_mask, spinup_surf, conv_filter,
                             yrs_to_run, case.dx, mb, gpr, data_logger)

    return c_fun


def cost_function(b, reg_parameters, ref_surf, ref_ice_mask, ref_inner_mask,
                  spinup_surf, conv_filter, yrs_to_run, dx, mb,
                  gpr=None, data_logger=None):
    """
    Calculates cost for a given bed and other given parameters.

    Parameters
    ----------
    b: ndarray
        Array of bed heights (unit: [m])
    reg_parameters: list or ndarray or torch.Tensor
        list or ndarray or torch.Tensor giving reg_parameters. Expected size: 11
    ref_surf: FloatTensor
        "Observed" surface height after forward run. This is supposed to be
        achieved. (unit: [m])
    ref_ice_mask: FloatTensor
        Tensor containing only 1's and 0's masking everything outside the
        glacier (border is included)
    ref_inner_mask: FloatTensor
        Tensor containing only 1's and 0's masking everything except the
        interior of the glacier (border is excluded)
    spinup_surf: FloatTensor
        Surface height after spinup (unit: [m])
    conv_filter: FloatTensor
        Precomputed FloatTensor for convolution to create inner_masks
    yrs_to_run: float
        years to run for forward modeling (unit: [a])
    dx: float
        grid spacing (unit: [m])
    mb: MassBalanceModel
        Model for the mass-balance needed in the forward run
    data_logger: DataLogger
        optionally logs data

    Returns
    -------
    tuple of (cost, grad) with cost as float and grad being a ndarray
    with same shape as b
    """
    guessed_bed = torch.tensor(b.reshape(ref_surf.shape), dtype=torch.float,
                               requires_grad=True)

    # run model forward
    init_ice_thick = spinup_surf - guessed_bed
    model_surf = run_forward_core(yrs_to_run, guessed_bed, dx, mb,
                                  init_ice_thick)
    model_ice_mask = ((model_surf - guessed_bed) > 0.).type(
        dtype=torch.float)
    model_inner_mask = torch.zeros(model_ice_mask.shape)
    model_inner_mask[1:-1, 1:-1] = torch.conv2d(
        model_ice_mask.unsqueeze(0).unsqueeze(0), conv_filter) == \
                                   conv_filter.sum()

    # quantify costs (all terms)
    c_terms = get_costs(reg_parameters, ref_surf, ref_ice_mask, ref_inner_mask,
                        guessed_bed, model_surf, model_ice_mask,
                        model_inner_mask, dx, gpr)

    # Calculate costs and gradient w.r.t guessed_bed
    c = c_terms.sum()
    c.backward()  # This is where the magic happens
    g = guessed_bed.grad  # And this is where we can now find the gradient

    # Format for scipy.optimize.minimize
    grad = g.detach().numpy().reshape(b.shape).astype(np.float64)
    cost = c.detach().numpy().astype(np.float64)

    # Do keep data for logging if desired
    if data_logger is not None:
        data_logger.c_terms.append(c_terms.detach().numpy())
        data_logger.costs.append(cost)
        data_logger.grads.append(grad)
        data_logger.beds.append(guessed_bed.detach().numpy())
        data_logger.surfs.append(model_surf.detach().numpy())

    return cost, grad


def creat_cost_function_flowline(bed_known, shape_known, spinup_surf,
                                 reg_parameters, ref_surf, ref_width,
                                 yrs_to_run, dx, mb_model, torch_type='float',
                                 used_geometry='parabolic', data_logger=None):

    def c_fun(parameters_unknown):
        return cost_fucntion_flowline(parameters_unknown,
                                      bed_known,
                                      shape_known,
                                      spinup_surf,
                                      reg_parameters,
                                      ref_surf,
                                      ref_width,
                                      yrs_to_run,
                                      dx,
                                      mb_model,
                                      torch_type,
                                      used_geometry,
                                      data_logger)

    return c_fun


def get_costs(reg_parameters, ref_surf, ref_ice_mask, ref_inner_mask,
              guessed_bed, model_surf, model_ice_mask, model_inner_mask, dx,
              gpr=None):
    """
    TODO: Documentation

    Parameters
    ----------
    reg_parameters
    ref_surf
    ref_ice_mask
    ref_inner_mask
    guessed_bed
    model_surf
    model_ice_mask
    model_inner_mask
    dx
    bed_measurements

    Returns
    -------

    """
    n_inner_mask = model_inner_mask.sum()
    n_ice_mask = ref_ice_mask.sum()
    n_grid = ref_surf.numel()
    margin = ref_ice_mask - ref_inner_mask
    cost = torch.zeros(len(reg_parameters) + 1)

    # TODO recheck all indices for reg_parameters and cost
    cost[-1] = ((ref_surf - model_surf) * ref_inner_mask).pow(2).sum()
    cost[0] = reg_parameters[0] *\
              ((ref_surf - model_surf) * margin).pow(2).sum()

    if reg_parameters[1] != 0:
        # penalizes ice thickness, where ice thickness should be 0
        cost[1] = reg_parameters[1] * (((model_surf - guessed_bed)
                                        * (1. - ref_ice_mask)).pow(2).sum())

    if reg_parameters[2] != 0:
        # penalize large derivatives of bed under glacier
        # -> avoids numerical instabilites
        db_dx1 = (guessed_bed[:, :-2] - guessed_bed[:, 1:-1]) / dx
        db_dx2 = (guessed_bed[:, 1:-1] - guessed_bed[:, 2:]) / dx
        db_dy1 = (guessed_bed[:-2, :] - guessed_bed[1:-1, :]) / dx
        db_dy2 = (guessed_bed[1:-1, :] - guessed_bed[2:, :]) / dx
        db_dx_sq = 0.5 * (db_dx1.pow(2) + db_dx2.pow(2)) * ref_ice_mask[:, 1:-1]
        db_dy_sq = 0.5 * (db_dy1.pow(2) + db_dy2.pow(2)) * ref_ice_mask[1:-1, :]
        cost[2] = reg_parameters[2] * 0.5 * ((db_dx_sq.sum() + db_dy_sq.sum()))
                # TODO: think about first squaring forward and backward and then adding vs adding and then squaring
                # then an additional .abs() is required for db_dx1, ...

    if reg_parameters[3] != 0:
        # penalize high curvature of bed exactly at boundary pixels of
        # glacier for a smooth transition from glacier-free to glacier
        ddb_dx = (guessed_bed[:, :-2] + guessed_bed[:, 2:]
                  - 2 * guessed_bed[:, 1:-1]) / dx ** 2
        ddb_dy = (guessed_bed[:-2, :] + guessed_bed[2:, :]
                  - 2 * guessed_bed[1:-1, :]) / dx ** 2
        ddb_dx = ddb_dx * margin[:, 1:-1]
        ddb_dy = ddb_dy * margin[1:-1, :]
        #ddb_dx = ddb_dx * ref_ice_mask[:, 1:-1]
        #ddb_dy = ddb_dy * ref_ice_mask[1:-1, :]
        cost[3] = reg_parameters[3] \
                  * ((ddb_dx.pow(2).sum() + ddb_dy.pow(2).sum()))

    if len(reg_parameters) > 4 and reg_parameters[4] != 0:
        # penalize high curvature of surface in glacier bounds
        dds_dx = (model_surf[:, :-2] + model_surf[:, 2:]
                  - 2 * model_surf[:, 1:-1]) / dx ** 2
        dds_dy = (model_surf[:-2, :] + model_surf[2:, :]
                  - 2 * model_surf[1:-1, :]) / dx ** 2
        dds_dx = dds_dx * model_inner_mask[:, 1:-1]
        dds_dy = dds_dy * model_inner_mask[1:-1, :]
        cost[4] = reg_parameters[4] \
                  * ((dds_dx.pow(2).sum() + dds_dy.pow(2).sum()))
    
    if gpr is not None and reg_parameters[5] != 0:
        # penalize large deviations from bed measurements
        # bed measurements should be given as two tensors, one for the data
        # and one for the mask
        gpr_data, gpr_mask = gpr
        cost[5] = reg_parameters[5] * ((guessed_bed - gpr_data)
                                       * gpr_mask).pow(2).sum()

    """
    if reg_parameters[3] != 0:
        # penalize large derivatives of ice thickness
        it = model_surf - guessed_bed
        dit_dx = (it[:, :-2] - it[:, 2:]) / (2. * dx)
        dit_dy = (it[:-2, :] - it[2:, :]) / (2. * dx)
        dit_dx = dit_dx * model_inner_mask[:, 1:-1]
        dit_dy = dit_dy * model_inner_mask[1:-1, :]
        cost[3] = reg_parameters[3] * (
                (dit_dx.pow(2).sum() + dit_dy.pow(2).sum()) / n_inner_mask)

    if reg_parameters[4] != 0:
        # penalize large derivatives of bed inside glacier bounds
        # TODO: is central differences really what we want here?
        db_dx = (guessed_bed[:, :-2] - guessed_bed[:, 2:]) / dx
        db_dy = (guessed_bed[:-2, :] - guessed_bed[2:, :]) / dx
        db_dx = db_dx * model_inner_mask[:, 1:-1]
        db_dy = db_dy * model_inner_mask[1:-1, :]
        cost[4] = reg_parameters[4] * (
                (db_dx.pow(2).sum() + db_dy.pow(2).sum()) / n_inner_mask)

    if reg_parameters[5] != 0:
        # penalizes bed != reference surf where we know about the bed
        # height because of ice thickness == 0
        cost[5] = reg_parameters[5] * \
                  (((ref_surf - guessed_bed)
                    * (1. - ref_ice_mask)).pow(2).sum()
                   / (n_grid - n_ice_mask))

    if reg_parameters[6] != 0:
        # penalize high curvature of ice thickness (in glacier bounds)
        it = model_surf - guessed_bed
        ddit_dx = (it[:, :-2] + it[:, 2:] - 2 * it[:, 1:-1]) / dx ** 2
        ddit_dy = (it[:-2, :] + it[2:, :] - 2 * it[1:-1, :]) / dx ** 2
        ddit_dx = ddit_dx * model_inner_mask[:, 1:-1]
        ddit_dy = ddit_dy * model_inner_mask[1:-1, :]
        cost[6] = reg_parameters[6] * ((ddit_dx.pow(2).sum()
                                        + ddit_dy.pow(2).sum())
                                       / (2 * n_inner_mask))

    if reg_parameters[7] != 0:
        # penalize high curvature of bed (in glacier bounds)
        ddb_dx = (guessed_bed[:, :-2] + guessed_bed[:, 2:]
                  - 2 * guessed_bed[:, 1:-1]) / dx ** 2
        ddb_dy = (guessed_bed[:-2, :] + guessed_bed[2:, :]
                  - 2 * guessed_bed[1:-1, :]) / dx ** 2
        ddb_dx = ddb_dx * model_inner_mask[:, 1:-1]
        ddb_dy = ddb_dy * model_inner_mask[1:-1, :]
        cost[7] = reg_parameters[7] * ((ddb_dx.pow(2).sum()
                                        + ddb_dy.pow(2).sum())
                                       / (2. * n_inner_mask))

    if reg_parameters[8] != 0:
        # penalize high curvature of surface inside glacier
        dds_dx = (model_surf[:, :-2] + model_surf[:, 2:]
                  - 2 * model_surf[:, 1:-1]) / dx ** 2
        dds_dy = (model_surf[:-2, :] + model_surf[2:, :]
                  - 2 * model_surf[1:-1, :]) / dx ** 2
        dds_dx = dds_dx * model_inner_mask[:, 1:-1]
        dds_dy = dds_dy * model_inner_mask[1:-1, :]
        cost[8] = reg_parameters[8] * ((dds_dx.pow(2).sum()
                                        + dds_dy.pow(2).sum()) / n_inner_mask)

    if reg_parameters[9] != 0:
        lmsd = LocalMeanSquaredDifference.apply
        cost[9] = reg_parameters[9] * lmsd(model_surf, ref_surf, ref_ice_mask,
                                           ref_ice_mask, guessed_bed)
    """

    return cost


class LocalMeanSquaredDifference(torch.autograd.Function):
    """
    More or less test class for own functions on tensors with custom
    backward functions

    Deprecated, only use as example
    """
    @staticmethod
    def forward(ctx, modelled_surf, surface_to_match, ice_region, ice_mask, bed):
        ctx.save_for_backward(modelled_surf, surface_to_match, ice_region, ice_mask, bed)
        msd = (modelled_surf - surface_to_match).pow(2).sum() / ice_region.sum().type(dtype=torch.float)
        return msd

    @staticmethod
    def backward(ctx, grad_output):
        modelled_surf, observed_surf, ice_region, ice_mask, bed = ctx.saved_tensors
        grad_modelled_surf = (modelled_surf - observed_surf) * ice_mask
        return None, None, None, None, grad_modelled_surf



def get_costs_arr(reg_parameters, ref_surf, ref_ice_mask, ref_inner_mask,
                  guessed_bed, model_surf, model_ice_mask, model_inner_mask,
                  dx):

    margin = np.logical_xor(ref_ice_mask, ref_inner_mask)
    cost = np.zeros(len(reg_parameters) + 1).tolist()

    # TODO recheck all indices for reg_parameters and cost
    cost[-1] = ((ref_surf - model_surf) * ref_inner_mask)**2  # Big
    #  TODO
    cost[0] = reg_parameters[0] * \
              ((ref_surf - model_surf) * margin)**2

    if reg_parameters[1] != 0:
        # penalizes ice thickness, where ice thickness should be 0
        cost[1] = reg_parameters[1] * (((model_surf - guessed_bed)
                                        * np.logical_not(ref_ice_mask))**2)

    if reg_parameters[2] != 0:
        # penalize large derivatives of bed under glacier
        # -> avoids numerical instabilites
        db_dx1 = (guessed_bed[:, :-2] - guessed_bed[:, 1:-1]) / dx
        db_dx2 = (guessed_bed[:, 1:-1] - guessed_bed[:, 2:]) / dx
        db_dy1 = (guessed_bed[:-2, :] - guessed_bed[1:-1, :]) / dx
        db_dy2 = (guessed_bed[1:-1, :] - guessed_bed[2:, :]) / dx
        db_dx_sq = 0.5 * (db_dx1**2 + db_dx2**2) * ref_ice_mask[:,                                                   1:-1]
        db_dx_sq_full = np.zeros(guessed_bed.shape)
        db_dx_sq_full[:, 1:-1] = db_dx_sq
        db_dy_sq = 0.5 * (db_dy1**2 + db_dy2**2) * ref_ice_mask[1:-1,                                 :]
        db_dy_sq_full = np.zeros(guessed_bed.shape)
        db_dy_sq_full[1:-1, :] = db_dy_sq
        cost[2] = reg_parameters[2] * 0.5 * (db_dx_sq_full + db_dy_sq_full)
        # TODO: think about first squaring forward and backward and then adding vs adding and then squaring
        # then an additional .abs() is required for db_dx1, ...

    if reg_parameters[3] != 0:
        # penalize high curvature of bed exactly at boundary pixels of
        # glacier for a smooth transition from glacier-free to glacier
        ddb_dx = (guessed_bed[:, :-2] + guessed_bed[:, 2:]
                  - 2 * guessed_bed[:, 1:-1]) / dx ** 2
        ddb_dy = (guessed_bed[:-2, :] + guessed_bed[2:, :]
                  - 2 * guessed_bed[1:-1, :]) / dx ** 2

        ddb_dx = ddb_dx * margin[:, 1:-1]
        ddb_dy = ddb_dy * margin[1:-1, :]
        #ddb_dy = ddb_dy * ref_ice_mask[1:-1, :]
        #ddb_dx = ddb_dx * ref_ice_mask[:, 1:-1]
        ddb_dx_full = np.zeros(guessed_bed.shape)
        ddb_dx_full[:, 1:-1] = ddb_dx
        ddb_dy_full = np.zeros(guessed_bed.shape)
        ddb_dy_full[1:-1, :] = ddb_dy
        cost[3] = reg_parameters[3] * (ddb_dx_full**2 + ddb_dy_full**2)

    if len(reg_parameters) > 4 and reg_parameters[4] != 0:
        # penalize high curvature of surface in glacier bounds
        dds_dx = (model_surf[:, :-2] + model_surf[:, 2:]
                  - 2 * model_surf[:, 1:-1]) / dx ** 2
        dds_dy = (model_surf[:-2, :] + model_surf[2:, :]
                  - 2 * model_surf[1:-1, :]) / dx ** 2
        dds_dx_full = np.zeros(guessed_bed.shape)
        dds_dx_full[:, 1:-1] = dds_dx * model_inner_mask[:, 1:-1]
        #dds_dy = dds_dy * model_inner_mask[1:-1, :]
        dds_dy_full = np.zeros(guessed_bed.shape)
        dds_dy_full[1:-1, :] = dds_dy * model_inner_mask[1:-1, :]
        cost[4] = reg_parameters[4] \
                  * (dds_dx_full**2 + dds_dy_full**2)

    if len(reg_parameters) > 5 and reg_parameters[5] != 0:
        # penalize large derivatives of surface
        # -> avoids numerical instabilites
        ds_dx1 = (model_surf[:, :-2] - model_surf[:, 1:-1]) / dx
        ds_dx2 = (model_surf[:, 1:-1] - model_surf[:, 2:]) / dx
        ds_dy1 = (model_surf[:-2, :] - model_surf[1:-1, :]) / dx
        ds_dy2 = (model_surf[1:-1, :] - model_surf[2:, :]) / dx
        ds_dx_sq = 0.5 * (ds_dx1**2
                          + ds_dx2**2) * model_inner_mask[:, 1:-1]
        ds_dy_sq = 0.5 * (ds_dy1**2
                          + ds_dy2**2) * model_inner_mask[1:-1, :]
        ds_dx_sq_full = np.zeros(guessed_bed.shape)
        ds_dx_sq_full[:, 1:-1] = ds_dx_sq
        ds_dy_sq_full = np.zeros(guessed_bed.shape)
        ds_dy_sq_full[1:-1, :] = ds_dy_sq
        cost[5] = reg_parameters[5] * 0.5 * ((ds_dx_sq_full + ds_dy_sq_full))
        # TODO: think about first squaring forward and backward and then adding vs adding and then squaring
        # then an additional .abs() is required for db_dx1, ...
    return cost


def cost_fucntion_flowline(parameters_unknown, bed_known, shape_known,
                           spinup_surf, reg_parameters, ref_surf, ref_width,
                           yrs_to_run, dx, mb_model, torch_type='float',
                           used_bed_geometry='rectangular', data_logger=None):
    if torch_type == 'double':
        torch_type = torch.double
    else:
        torch_type = torch.float

    # check if only bed is unknown or if shape and bed is unknown
    if len(bed_known) == len(shape_known):
        # check if bed and shape unknown have the same length
        assert np.mod(len(parameters_unknown), 2) == 0

        split_point = int(len(parameters_unknown) / 2)
        bed_unknown = parameters_unknown[:split_point]
        shape_unknown = parameters_unknown[split_point:]

        bed_mag = magnitude(np.median(bed_unknown))
        shape_mag = magnitude(np.median(shape_unknown))

        # create variables for optimization
        bed_unknown = torch.tensor(bed_unknown,
                                   dtype=torch_type,
                                   requires_grad=True)
        shape_unknown = torch.tensor(shape_unknown,
                                     dtype=torch_type,
                                     requires_grad=True)

        bed_known = torch.tensor(bed_known,
                                 dtype=torch_type,
                                 requires_grad=False)
        shape_known = torch.tensor(shape_known,
                                   dtype=torch_type,
                                   requires_grad=False)

        bed = torch.cat((bed_unknown, bed_known), 0)
        shape = torch.cat((shape_unknown, shape_known), 0)
    else:
        bed_unknown = torch.tensor(parameters_unknown,
                                   dtype=torch_type,
                                   requires_grad=True)
        bed_known = torch.tensor(bed_known,
                                 dtype=torch_type,
                                 requires_grad=False)
        bed = torch.cat((bed_unknown, bed_known), 0)

        shape_unknown = None
        shape = torch.tensor(shape_known,
                             dtype=torch_type,
                             requires_grad=False)

    model_flowline = run_flowline_forward_core(
        spinup_surf,
        bed,
        shape,
        dx,
        torch_type,
        mb_model,
        yrs_to_run,
        used_bed_geometry)

    model_surf = model_flowline.surface_h
    model_width = model_flowline.widths_m
    model_thick = model_flowline.thick

    c_terms = get_flowline_costs(reg_parameters, ref_surf, ref_width, dx,
                                 bed, model_surf, model_width,
                                 torch_type)

    c = c_terms.sum()
    c.backward()

    bed_grad = bed_unknown.grad.detach().numpy().astype(np.float64)

    bed_grad_mag = magnitude(np.max(np.abs(bed_grad)))

    # scaling factor to comensate different magnitudes of bed h and shape
    bed_grad = bed_grad * 100

    # smoothing for overestimation of gradient in first point
    bed_grad[0] = bed_grad[1]

    if shape_unknown is not None:
        shape_grad = shape_unknown.grad.detach().numpy().astype(np.float64)

        shape_grad_mag = magnitude(np.max(np.abs(shape_grad)))
        # scaling factor to compensate different magnitudes of bed h and shape
        shape_grad = shape_grad# / 100

        if np.abs(shape_grad[-2]) < np.abs(shape_grad[-3]):
            shape_grad[-2] = shape_grad[-3]  # * np.sign(grad[-2])
        # if np.abs(grad[-1]) < np.abs(grad[-3]):
        shape_grad[-1] = shape_grad[-2]
        # smoothing grad at last gird point
        thick = model_thick[:len(shape_grad)].detach().numpy().\
            astype(np.float64)
        # good_indices = np.argwhere(thick > 20)
        # shape_grad = np.where(thick > 20,
        #                      shape_grad,
        #                      shape_grad[good_indices[-1]])
        shape_grad = shape_grad / 100
    else:
        shape_grad = np.array([])

    grad = np.append(bed_grad, shape_grad)

    cost = c.detach().numpy().astype(np.float64)

    #print('bed: ' + str(bed_mag) + '\nbed grad: ' + str(bed_grad_mag) +
    #      '\nshape: ' + str(shape_mag) + '\nshape grad: ' + str(shape_grad_mag))
    # Do keep data for logging if desired
    if data_logger is not None:
        data_logger.c_terms.append(c_terms.detach().numpy())
        data_logger.costs.append(cost)
        data_logger.grads.append(grad)
        data_logger.beds.append(bed_unknown.detach().numpy())
        data_logger.surfs.append(model_surf.detach().numpy())
        data_logger.widths.append(model_width.detach().numpy())
        if shape_unknown is not None:
            data_logger.shapes.append(shape_unknown.detach().numpy())

    return cost, grad


def get_flowline_costs(reg_parameters, ref_surf, ref_width, dx,
                       bed, model_surf, model_width, torch_type):
    cost = torch.zeros(3,
                       dtype=torch_type)

    if reg_parameters[0] != 0:
        ref_surf = to_torch_tensor(ref_surf, torch_type)
        cost[0] = (reg_parameters[0] * (ref_surf - model_surf).pow(2).sum())

    if reg_parameters[1] != 0:
        ref_width = to_torch_tensor(ref_width, torch_type)
        cost[1] = (reg_parameters[1] * (ref_width - model_width).pow(2).sum())

    if reg_parameters[2] != 0:
        db_dx = (bed[1:] - bed[:-1]) / dx
        cost[2] = reg_parameters[2] * db_dx.pow(2).sum()

    return cost


def create_cost_fct_old(bed_h,
                    shape,
                    spinup_surf,
                    reg_parameter,
                    ref_surf,
                    ref_width,
                    ice_mask,
                    yrs_to_run,
                    dx,
                    mb_model,
                    opti_var='bed_h',
                    torch_type='double',
                    used_geometry='parabolic',
                    data_logger=None,
                    get_c_terms=False,
                    grad_scaling={'bed_h': 1,
                                  'shape': 1},
                    grad_smoothing={'bed_h': 'no',
                                    'shape': 'no'}):
    '''
    Creates a cost function for optimizing bed height or for optimizing bed
    shape
    '''

    def c_fun(parameter_unknown):
        return cost_fct(parameter_unknown,
                        opti_var,
                        bed_h,
                        shape,
                        spinup_surf,
                        reg_parameter,
                        ref_surf,
                        ref_width,
                        ice_mask,
                        yrs_to_run,
                        dx,
                        mb_model,
                        torch_type,
                        used_geometry,
                        data_logger,
                        get_c_terms,
                        grad_scaling,
                        grad_smoothing)

    return c_fun


def cost_fct_old(parameter_unknown,
             opti_var,
             bed_h,
             shape,
             spinup_surf,
             reg_parameter,
             ref_surf,
             ref_width,
             ice_mask,
             yrs_to_run,
             dx,
             mb_model,
             torch_type='double',
             used_geometry='parabolic',
             data_logger=None,
             get_c_terms=False,
             grad_scaling={'bed_h': 1,
                           'shape': 1},
             grad_smoothing={'bed_h': 'no',
                             'shape': 'no'}):
    # check which data type should be used for calculation
    if torch_type == 'double':
        torch_type = torch.double
    else:
        torch_type = torch.float

    # check that all parameters are positive
    if np.any(np.sign(parameter_unknown) == -1):
        cost = np.Inf
        grad = np.empty(len(parameter_unknown)) * np.nan
        return cost, grad

    # ice mask is needed for cost calculation
    ice_mask = torch.tensor(ice_mask,
                            requires_grad=False,
                            dtype=torch.bool)

    # check which parameter should be optimized
    if opti_var == 'bed_h':
        bed_unknown = torch.tensor(parameter_unknown,
                                   dtype=torch_type,
                                   requires_grad=True)
        bed_known = torch.tensor(bed_h,
                                 dtype=torch_type,
                                 requires_grad=False)
        bed_h = torch.empty(sum(list(bed_unknown.size() +
                                     bed_known.size())),
                            dtype=torch_type,
                            requires_grad=False)
        bed_h[ice_mask] = bed_unknown
        bed_h[~ice_mask] = bed_known
        # torch.cat((bed_unknown, bed_known), 0)

        if used_geometry == 'parabolic':
            shape = torch.tensor(shape,
                                 dtype=torch_type,
                                 requires_grad=False)

    elif opti_var == 'shape':
        shape_unknown = torch.tensor(parameter_unknown,
                                     dtype=torch_type,
                                     requires_grad=True)
        shape_known = torch.tensor(shape,
                                   dtype=torch_type,
                                   requires_grad=False)
        shape = torch.empty(sum(list(shape_unknown.size() +
                                     shape_known.size())),
                            dtype=torch_type,
                            requires_grad=False)
        shape[ice_mask] = shape_unknown
        shape[~ice_mask] = shape_known
        # torch.cat((shape_unknown, shape_known), 0)

        bed_h = torch.tensor(bed_h,
                             dtype=torch_type,
                             requires_grad=False)

    elif opti_var == 'bed_h and shape':
        split_point = int(len(parameter_unknown) / 2)

        bed_unknown = parameter_unknown[:split_point]
        bed_unknown = torch.tensor(bed_unknown,
                                   dtype=torch_type,
                                   requires_grad=True)
        bed_known = torch.tensor(bed_h,
                                 dtype=torch_type,
                                 requires_grad=False)
        bed_h = torch.cat((bed_unknown, bed_known), 0)
        # torch.empty(sum(list(bed_unknown.size() +
        #                              bed_known.size())),
        #                     dtype=torch_type,
        #                     requires_grad=False)
        # bed_h[ice_mask] = bed_unknown
        # bed_h[~ice_mask] = bed_known
        # torch.cat((bed_unknown, bed_known), 0)

        shape_unknown = parameter_unknown[split_point:]
        shape_unknown = torch.tensor(shape_unknown,
                                     dtype=torch_type,
                                     requires_grad=True)
        shape_known = torch.tensor(shape,
                                   dtype=torch_type,
                                   requires_grad=False)
        shape = torch.cat((shape_unknown, shape_known), 0)
        # torch.empty(sum(list(shape_unknown.size() +
        #                              shape_known.size())),
        #                     dtype=torch_type,
        #                     requires_grad=False)
        # shape[ice_mask] = shape_unknown
        # shape[~ice_mask] = shape_known
        # torch.cat((shape_unknown, shape_known), 0)

    else:
        raise ValueError('Optimisation variable unknown!')

    if used_geometry == 'parabolic':
        # check if shape and bed_h same length
        assert len(bed_h) == len(shape), 'Parameters not the same length!!!'

    # forward run of model
    try:
        model_flowline = run_flowline_forward_core(
            spinup_surf,
            bed_h,
            shape,
            dx,
            torch_type,
            mb_model,
            yrs_to_run,
            used_geometry,
            ref_surf,
            ref_width)

        model_surf = model_flowline.surface_h
        model_width = model_flowline.widths_m
        model_thick = model_flowline.thick
    except MemoryError:
        print('MemoryError in forward model run (due to a too small timestep) \
              -> set Costfunction to Inf')
        cost = np.Inf
        grad = np.empty(len(parameter_unknown)) * np.nan
        return cost, grad

    # calculate terms of cost function
    c_terms, max_diff = get_cost_terms(reg_parameter, ref_surf, ref_width, dx,
                                       bed_h, shape, model_surf, model_width,
                                       torch_type, model_thick, ice_mask,
                                       used_geometry)

    # shortcut when regularisation parameters are searched
    if get_c_terms:
        return c_terms.detach().numpy().astype(np.float64)

    # sum up cost function terms
    c = c_terms.sum()

    # calculate the gradient for the optimisation parameter
    c.backward()

    # convert cost to numpy array
    cost = c.detach().numpy().astype(np.float64)

    # convert thick to numpy array
    thick = model_thick.detach().numpy().astype(np.float64)

    # convert gradient to numpy array and do some smoothing
    if (opti_var == 'bed_h') or (opti_var == 'bed_h and shape'):
        g_bed = bed_unknown.grad
        grad_bed = g_bed.detach().numpy().astype(np.float64)
        if grad_smoothing['bed_h'] == 'no':
            # do nothing
            pass
        elif grad_smoothing['bed_h'] == '2nd is 1st':
            grad_bed[0] = grad_bed[1]
        # stopping criterion to avoid over minimizing
        elif grad_smoothing['bed_h'] == 'stop if small enough':
            if max_diff[0] < 0.5:
                print('stopped becaus max_diff sfc is {}'.format(max_diff[0]))
                cost = np.zeros(1)
                grad = np.zeros(len(parameter_unknown))
                return cost, grad
        else:
            raise ValueError('Unknown gradient smoothing for bed_h!')

    if (opti_var == 'shape') or (opti_var == 'bed_h and shape'):
        g_shape = shape_unknown.grad
        grad_shape = g_shape.detach().numpy().astype(np.float64)
        # grad_unsmoothed = np.copy(grad_shape)
        if grad_smoothing['shape'] == 'no':
            # do nothing
            pass
        elif grad_smoothing['shape'] == 'last 3 same':
            if np.abs(grad_shape[-2]) < np.abs(grad_shape[-3]):
                grad_shape[-2] = grad_shape[-3]
            grad_shape[-1] = grad_shape[-2]
        # stopping criterion to avoid over minimizing
        elif grad_smoothing['shape'] == 'stop if small enough':
            if max_diff[1] < 0.5:
                print('stopped becaus max_diff width is {}'
                      .format(max_diff[1]))
                cost = np.zeros(1)
                grad = np.zeros(len(parameter_unknown))
                return cost, grad
        else:
            raise ValueError('Unknown gradient smoothing for shape!')

    if opti_var == 'bed_h':
        grad = grad_bed * grad_scaling['bed_h']
    elif opti_var == 'shape':
        grad = grad_shape * grad_scaling['shape']
    elif opti_var == 'bed_h and shape':
        # scaling for gradients
        grad_bed = grad_bed * grad_scaling['bed_h']
        grad_shape = grad_shape * grad_scaling['shape']
        grad = np.append(grad_bed, grad_shape)

    # Do keep data for logging if desired
    if data_logger is not None:
        data_logger.fct_calls.append(data_logger.fct_calls[-1] + 1)
        if opti_var == 'bed_h':
            data_logger.in_bed_h_opti = True
            data_logger.bed_h_c_terms.append(c_terms.detach().numpy())
            data_logger.bed_h_costs.append(cost)
            data_logger.bed_h_grads.append(grad)
            data_logger.beds.append(bed_unknown.detach().numpy())
            data_logger.bed_h_surfs.append(model_surf.detach().numpy())
            data_logger.bed_h_widths.append(model_width.detach().numpy())
            data_logger.bed_h_thicks.append(thick)

        elif opti_var == 'shape':
            data_logger.in_shape_opti = True
            data_logger.shape_c_terms.append(c_terms.detach().numpy())
            data_logger.shape_costs.append(cost)
            data_logger.shape_grads_smoothed.append(grad)
            # data_logger.shape_grads.append(grad_unsmoothed)
            data_logger.shapes.append(shape_unknown.detach().numpy())
            data_logger.shape_surfs.append(model_surf.detach().numpy())
            data_logger.shape_widths.append(model_width.detach().numpy())
            data_logger.shape_thicks.append(thick)
        elif opti_var == 'bed_h and shape':
            data_logger.in_bed_h_and_shape_opti = True
            data_logger.c_terms.append(c_terms.detach().numpy())
            data_logger.costs.append(cost)
            data_logger.grads.append(grad)
            data_logger.beds.append(bed_unknown.detach().numpy())
            data_logger.surfs.append(model_surf.detach().numpy())
            data_logger.widths.append(model_width.detach().numpy())
            data_logger.shapes.append(shape_unknown.detach().numpy())
            data_logger.thicks.append(thick)

    return cost, grad


def get_cost_terms_old(reg_parameter,
                   ref_surf,
                   ref_width,
                   dx,
                   bed_h,
                   shape,
                   model_surf,
                   model_width,
                   torch_type,
                   model_thick,
                   ice_mask,
                   used_geometry):
    # calculate cost terms
    costs = torch.zeros(6,
                        dtype=torch_type)

    # misfit between modeled and measured surface height
    ref_surf = to_torch_tensor(ref_surf, torch_type)
    costs[0] = reg_parameter[0] * (ref_surf - model_surf).pow(2).sum()

    # smoothnes of glacier bed
    db_dx = (bed_h[1:] - bed_h[:-1]) / dx
    costs[1] = reg_parameter[1] * db_dx.pow(2).sum()

    # misfit between modeled and measured width
    ref_width = to_torch_tensor(ref_width, torch_type)
    costs[2] = reg_parameter[2] * ((ref_width - model_width)).pow(2).sum()

    if used_geometry == 'parabolic':
        # smoothnes of shape parameter
        dshape_dx = (shape[1:] - shape[:-1]) / dx
        costs[3] = reg_parameter[3] * dshape_dx.pow(2).sum()
    else:
        costs[3] = 0.

    # guessed glacier bed higher than measured surface
    costs[4] = reg_parameter[4] * torch.where(bed_h > ref_surf,
                                              torch.tensor(1.),
                                              torch.tensor(0.)).sum()

    # ice thickness zero where no glacier should be and vice versa
    modeled_ice = torch.where(model_thick > 1e-2,
                              torch.tensor(1),
                              torch.tensor(0))
    costs[5] = reg_parameter[5] * torch.where(ice_mask != modeled_ice,
                                              torch.tensor(1.),
                                              torch.tensor(0.)).sum()

    # for having a break criterion if the largest difference is below a certain
    # threshold
    max_diff = np.zeros(2)
    max_diff[0] = np.max(np.abs(ref_surf.detach().numpy() -
                         model_surf.detach().numpy()))
    max_diff[1] = np.max(np.abs(ref_width.detach().numpy() -
                         model_width.detach().numpy()))

    return costs, max_diff


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
                    get_c_terms=False,
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
                        get_c_terms,
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
        split_point = int(len(unknown_parameter) / 2)

        bed_unknown = unknown_parameter[:split_point]
        bed_unknown = torch.tensor(bed_unknown,
                                   dtype=torch_type,
                                   requires_grad=True)

        bed_known = known_parameter[:split_point]
        bed_known = torch.tensor(bed_known,
                                 dtype=torch_type,
                                 requires_grad=False)

        bed_h = torch.empty(sum(list(bed_h_unknown.size() +
                                     bed_h_known.size())),
                            dtype=torch_type,
                            requires_grad=False)
        bed_h[ice_mask] = bed_h_unknown
        bed_h[~ice_mask] = bed_h_known

        shape_var_unknown = unknown_parameter[split_point:]
        shape_var_unknown = torch.tensor(shape_var_unknown,
                                         dtype=torch_type,
                                         requires_grad=True)

        shape_var_known = known_parameter[split_point:]
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
        grad = bed_unknown.grad.detach().numpy().astype(np.float64)
    elif opti_var in ['bed_shape', 'w0']:
        grad = shape_var_unknown.grad.detach().numpy().astype(np.float64)
    elif opti_var in ['bed_h and bed_shape', 'bed_h and w0']:
        grad_bed_h = bed_unknown.grad.detach().numpy().astype(np.float64)
        grad_shape_var = \
            shape_var_unknown.grad.detach().numpy().astype(np.float64)
        grad = np.append(grad_bed_h, grad_shape_var)
    else:
        raise ValueError('Unknown optimisation variable!')

    # help function to save data in datalogger as numpy array
    def save_data_in_datalogger(datalogger_var, data):
        if type(data) == torch.Tensor:
            data = data.detach().numpy().astype(np.float64)

        datalogger_var = np.append(datalogger_var, data)

    # save data in datalogger
    save_data_in_datalogger(datalogger.costs, cost)
    save_data_in_datalogger(datalogger.c_terms, c_terms)
    save_data_in_datalogger(datalogger.sfc_h, model_sfc_h)
    save_data_in_datalogger(datalogger.widths, model_widths)
    save_data_in_datalogger(datalogger.opti_var_iteration, opti_var)
    if opti_var == 'bed_h':
        save_data_in_datalogger(datalogger.guessed_opti_var_1, bed_unknown)
        save_data_in_datalogger(datalogger.grads_opti_var_1, grad)
    elif opti_var in ['bed_shape', 'w0']:
        save_data_in_datalogger(datalogger.guessed_opti_var_1,
                                shape_var_unknown)
        save_data_in_datalogger(datalogger.grads_opti_var_1, grad)
    elif opti_var in ['bed_h and bed_shape', 'bed_h and w0']:
        save_data_in_datalogger(datalogger.guessed_opti_var_1, bed_unknown)
        save_data_in_datalogger(datalogger.grads_opti_var_1, grad_bed_h)
        save_data_in_datalogger(datalogger.guessed_opti_var_2,
                                shape_var_unknown)
        save_data_in_datalogger(datalogger.grads_opti_var_2, grad_shape_var)
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

    # smoothnes of glacier bed
    db_dx = (model_bed_h[1:] - model_bed_h[:-1]) / dx
    costs[1] = reg_parameters[1] * db_dx.pow(2).sum()

    # misfit between modeled and measured width
    true_widths = to_torch_tensor(true_widths, torch_type)
    costs[2] = reg_parameters[2] * ((true_widths - model_widths)).pow(2).sum()

    # ice thickness close to zero where no glacier should be and vice versa
    model_ice_mask = torch.where(model_thicks > 1e-2,
                                 torch.tensor(1),
                                 torch.tensor(0))
    true_ice_mask = to_torch_tensor(true_ice_mask, torch_type)
    costs[3] = reg_parameters[3] * torch.where(true_ice_mask != model_ice_mask,
                                               torch.tensor(1.),
                                               torch.tensor(0.)).sum()

    return costs
