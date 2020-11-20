import torch
import numpy as np
import rasterio
import salem
from oggm import entity_task, cfg
from combine.core.sia2d_adapted import Upstream2D
import logging
from combine.core.flowline_adapted import ParabolicBedFlowline
from combine.core.flowline_adapted import RectangularBedFlowline
from combine.core.flowline_adapted import TrapezoidalBedFlowline
from combine.core.flowline_adapted import FluxBasedModel
# -------------------------------
# Further initialization / extended import tasks
# Module logger
log = logging.getLogger(__name__)


def run_flowline_forward_core(bed_h, shape_var, bed_geometry, mb_model,
                              spinup_sfc_h, yrs_to_run, map_dx, torch_type):
    '''
    Performs the forward run of the Flowlinemodel

    Parameters
    ----------
    bed_h : :py:class:`torch.Tensor`
        Height of the glacier bed.
    shape_var : :py:class:`torch.Tensor`
        Second variable describing the glacier bed. The meaning depends on the
        'bed_geometry'. ('rectangular' = width, 'parabolic' = shape_factor,
                         'trapezoidal' = bottom width)
    bed_geometry : str
        Defines the bed shape.
        Options: 'rectangular', 'parabolic' or 'trapezoidal'
    mb_model : :py:class:`oggm.core.massbalance.MassBalanceModel`
        The mass balance model to use.
    spinup_sfc_h : :py:class:`numpy.ndarray`
        Surface height of the glacier at the start of the model run.
    yrs_to_run : float
        Years for how long the glacier should be modeled with the given mass
        balance model.
    map_dx : float
        Model grid spacing in meters.
    torch_type : :py:class:`torch.dtype`
        Defines type for torch.Tensor.

    Returns
    -------
    :py:class:`oggm.Flowline`
        The flowline at the end of the model run.

    '''
    # initialise flowline according to bed_geometry
    if bed_geometry == 'rectangular':
        flowline = RectangularBedFlowline(surface_h=spinup_sfc_h,
                                          bed_h=bed_h,
                                          widths=shape_var,
                                          map_dx=map_dx,
                                          torch_type=torch_type)
    elif bed_geometry == 'parabolic':
        flowline = ParabolicBedFlowline(surface_h=spinup_sfc_h,
                                        bed_h=bed_h,
                                        bed_shape=shape_var,
                                        map_dx=map_dx,
                                        torch_type=torch_type)
    elif bed_geometry == 'trapezoidal':
        # lambda is set constant 1
        # (see https://docs.oggm.org/en/latest/ice-dynamics.html#trapezoidal)
        lambdas = torch.tensor([1.],
                               dtype=torch_type,
                               requires_grad=False)

        flowline = TrapezoidalBedFlowline(surface_h=spinup_sfc_h,
                                          bed_h=bed_h,
                                          w0=shape_var,
                                          lambdas=lambdas,
                                          map_dx=map_dx,
                                          torch_type=torch_type)
    else:
        raise ValueError('Unknown bed geometry!')

    model = FluxBasedModel(flowline,
                           mb_model=mb_model,
                           y0=0.,
                           fs=0.)

    model.run_until(yrs_to_run)

    return model.fls[0]
