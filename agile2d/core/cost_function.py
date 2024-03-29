import numpy as np
import salem
import torch

from agile2d.core.dynamics import run_forward_core


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


def get_costs(reg_parameters, ref_surf, ref_ice_mask, ref_inner_mask, guessed_bed,
              model_surf, model_ice_mask, model_inner_mask, dx,
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

