import torch
import numpy as np
import logging
from combine.core.flowline_adapted import ParabolicBedFlowline
from combine.core.flowline_adapted import RectangularBedFlowline
from combine.core.flowline_adapted import TrapezoidalBedFlowline
from combine.core.flowline_adapted import FluxBasedModel
# -------------------------------
# Further initialization / extended import tasks
# Module logger
log = logging.getLogger(__name__)


def run_flowline_forward_core(bed_h,
                              shape_var,
                              bed_geometry,
                              mb_model,
                              spinup_sfc_h,
                              spinup_sfc_known,
                              yrs_to_run,
                              spinup_yrs,
                              map_dx,
                              torch_type):
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
                                          widths_m=shape_var,
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

        # w0 = torch.clamp(shape_var, min=1.) should be done by bounds
        flowline = TrapezoidalBedFlowline(surface_h=spinup_sfc_h,
                                          bed_h=bed_h,
                                          w0_m=shape_var,
                                          lambdas=lambdas,
                                          map_dx=map_dx,
                                          torch_type=torch_type)
    else:
        raise ValueError('Unknown bed geometry!')

    if spinup_sfc_known:
        final_model = FluxBasedModel(flowline,
                                     mb_model=mb_model,
                                     y0=0.,
                                     fs=0.,
                                     mb_elev_feedback='annual')

        final_model.run_until(yrs_to_run)
    else:
        spinup_model = FluxBasedModel(flowline,
                                      mb_model=mb_model['spinup_mb_model'],
                                      y0=0.,
                                      fs=0.,
                                      mb_elev_feedback='annual')
        spinup_model.run_until(spinup_yrs)

        final_model = FluxBasedModel(spinup_model.fls[0],
                                     mb_model=mb_model['known_mb_model'],
                                     y0=0.)
        final_model.run_until(yrs_to_run)

    return final_model.fls[0]
