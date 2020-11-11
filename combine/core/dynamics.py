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


# @entity_task(log, writes=['dem_spinup', 'ice_thickness_spinup'])
@torch.no_grad()
def spinup(gdir, case, yr_spinup_end, mb=None):
    """
    Performs a forward run on the DEM in gdir for given number of years.
    Writes the found spinup_surface

    Parameters
    ----------
    gdir
    case
    yr_spinup_end
    mb

    Returns
    -------

    """

    with rasterio.open(gdir.get_filepath('dem')) as src:
        bed = src.read(1)
        profile = src.profile

    spinup_surf = run_forward(gdir, case, yr_spinup_end, bed, mb=mb)

    profile['dtype'] = 'float32'
    with rasterio.open(gdir.get_filepath('spinup_dem'),
                       'w', **profile) as dst:
        dst.write(spinup_surf, 1)
    spinup_surf = salem.GeoTiff(gdir.get_filepath('spinup_dem')).get_vardata()
    spinup_it = spinup_surf - bed
    np.save(gdir.get_filepath('spinup_ice_thickness'), spinup_it)

    return spinup_surf, spinup_it


@torch.no_grad()
def run_forward(gdir, case, yrs, bed, mb=None, init_ice_thick=None):
    """
    Wrapper for run_forward_core. Can derive mass-balance from case, if not
    given, accepts strings for bed_files as well as ndarrays and tensors


    Parameters
    ----------
    gdir: NonRGIGlacierDirectory
        The GlacierDirectory containing the data
    case: TestCase
        Case to be run, giving dx and mb (if not specified)
    yrs: float
        yrs to run forward
    bed: FloatTensor, ndarray, string or list/tuple of strings
        either array/tensor of bed height (unit: [m]) or filename
        (+filesuffix) of bed to be loaded from gdir (e.g. 'dem')
    mb:
        Mass-balance model
    init_ice_thick: ndarray
        optional, additional ice thickness for the model run

    Returns
    -------
    ndarray of surface height (unit: [m])
    """

    mb_model = mb
    if mb is None:
        mb_model = case.get_mb_model()

    if isinstance(bed, np.ndarray) or isinstance(bed, torch.Tensor):
        bed_arr = bed
    elif isinstance(bed, str):
        bed_arr = salem.GeoTiff(gdir.get_filepath(bed)).get_vardata()
    elif isinstance(bed, (list, tuple)) and len(bed) == 2 and all(
            isinstance(s, str) for s in bed):
        bed_arr = salem.GeoTiff(
            gdir.get_filepath(bed[0], filesuffix=bed[1])
        ).get_vardata()
    else:
        raise TypeError('Unexpected Type of argument "bed" in "run_forward"')

    bed_h = torch.tensor(bed_arr, dtype=torch.float,
                         requires_grad=False)
    ice_thick = init_ice_thick
    if init_ice_thick is not None:
        ice_thick = torch.tensor(ice_thick, dtype=torch.float,
                                 requires_grad=False)
    return run_forward_core(yrs, bed_h, case.dx, mb_model,
                            ice_thick).numpy()


def run_forward_core(yrs, bed, dx, mb_model, init_ice_thick):
    """

    Parameters
    ----------
    yrs: float
        yrs to run forward
    bed: FloatTensor
        tensor of bed height (unit: [m])
    dx: float
        model resolution (unit: [m])
    mb_model: MassBalanceModel
        Mass-balance-model used in the model
    init_ice_thick: ndarray
        initial ice thickness, if None it will be set to 0 for the whole
        domain (unit: [m])

    Returns
    -------
    FloatTensor of surface height (unit: [m])
    """
    model = Upstream2D(bed, dx=dx, mb_model=mb_model, y0=0,
                       glen_a=cfg.PARAMS['glen_a'], ice_thick_filter=None,
                       init_ice_thick=init_ice_thick)
    model.run_until(yrs)
    return model.surface_h


# @entity_task(log, writes=[''])
@torch.no_grad()
def create_glacier(gdir, run_spinup=True):
    """
    Creates a DEM-file for a glacier surface by running a forward model
    for spin-up to a first state and based on this state further on to a
    next state

    Parameters
    ----------
    gdir: NonRGIGlacierDirectory
        GlacierDirectory possibly containing spinup-state and used for
        saving the final reference state
    run_spinup: bool
        whether to run spin-up or rely on existing state

    Returns
    -------

    """
    inv_settings = gdir.inversion_settings
    if run_spinup:
        spinup(gdir, inv_settings['case'],
               inv_settings['yrs_spinup'],
               mb=inv_settings['mb_spinup'])

    spinup_it = np.load(gdir.get_filepath('spinup_ice_thickness'))
    spinup_surf = salem.GeoTiff(gdir.get_filepath('spinup_dem')).get_vardata()

    with rasterio.open(gdir.get_filepath('dem')) as src:
        bed = src.read(1)
        profile = src.profile

    ref_surf = run_forward(gdir, inv_settings['case'],
                           inv_settings['yrs_forward_run'], bed,
                           mb=inv_settings['mb_forward_run'],
                           init_ice_thick=spinup_it)

    profile['dtype'] = 'float32'
    with rasterio.open(gdir.get_filepath('ref_dem'),
                       'w', **profile) as dst:
        dst.write(ref_surf, 1)

    ref_surf = salem.GeoTiff(gdir.get_filepath('ref_dem')).get_vardata()
    ref_it = ref_surf - bed
    ref_ice_mask = ref_it > 0
    np.save(gdir.get_filepath('ref_ice_thickness'), ref_it)
    np.save(gdir.get_filepath('ref_ice_mask'), ref_ice_mask)


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
                           y0=0.)

    model.run_until(yrs_to_run)

    return model.fls[0]
