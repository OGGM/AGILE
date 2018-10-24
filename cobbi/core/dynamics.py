import torch
import numpy as np
import rasterio
import salem
from oggm import entity_task
import logging
# -------------------------------
# Further initialization / extended import tasks
# Module logger
log = logging.getLogger(__name__)


@entity_task(log, writes=['spinup_dem', 'spinup_ice_thickness'])
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

    spinup_surf = run_forward(gdir, yr_spinup_end, case, bed)

    with rasterio.open(gdir.get_filepath('dem', '_spinup'),
                       'w', **profile) as dst:
        dst.write(spinup_surf, 1)

    spinup_it = spinup_surf - bed
    np.save(gdir.get_filepath('ice_thickness', '_spinup'), spinup_it)

    return spinup_surf, spinup_it

@torch.no_grad()
def run_forward(gdir, yrs, case, bed, mb=None, init_ice_thick=None):
    """
    Wrapper for run_forward_core. Can derive mass-balance from case, if not
    given, accepts strings for bed_files as well as ndarrays and tensors


    Parameters
    ----------
    gdir: NonRGIGlacierDirectory
        The GlacierDirectory containing the data
    yrs: float
        yrs to run forward
    case: TestCase
        Case to be run, giving dx and mb (if not specified)
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
            gdir.get_filepath(bed[0],filesuffix=bed[1])
        ).get_vardata()
    else:
        raise TypeError('Unexpected Type of argument "bed" in "run_forward"')

    bed_h = torch.tensor(bed_arr, dtype=torch.float,
                         requires_grad=False)
    return run_forward_core(yrs, bed_h, case.dx, mb_model,
                            init_ice_thick).numpy()


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
    model = Upstream2D(bed, dx=dx, mb_model=mb_model,
                              y0=0, glen_a=cfg.PARAMS['glen_a'],
                              ice_thick_filter=None,
                              init_ice_thick=init_ice_thick)
    model.run_until(yrs)
    return model.surface_h


@entity_task(log, writes=[''])
@torch.no_grad()
def create_glacier(gdir, case, yrs_to_run, yrs_for_spinup=None):
    """
    Creates a DEM-file for a glacier surface by running a forward model on
    a spinup-state for a certain number of years. If there is no
    spinup-state but a number of years given, the spin-up state is first
    computed

    Parameters
    ----------
    gdir: NonRGIGlacierDirectory
        GlacierDirectory possibly containing spinup-state and used for
        saving the final reference state
    case: TestCase
        TestCase giving information about e.g. dx and mass-balance
    yrs_to_run: float
        yrs to run forward from spinup-state to create final glacier.
        (unit: [a])
    yrs_for_spinup: float
        optional argument. If this is specified, spinup is run for the given
        number of years. (unit: [a])

    Returns
    -------

    """
    if yrs_for_spinup is not None:
        spinup(gdir, case, yrs_for_spinup)

    spinup_it = gdir.get_filepath('ice_thickness', '_spinup')
    spinup_surf = gdir.get_filepath('dem', '_spinup')

    with rasterio.open(gdir.get_filepath('dem')) as src:
        bed = src.read(1)
        profile = src.profile

    ref_surf = run_forward(gdir, yrs_to_run, case, bed,
                           init_ice_thick=spinup_it)
    ref_it = ref_surf - bed
    ref_ice_mask = ref_it > 0

    with rasterio.open(gdir.get_filepath('dem', '_ref')) as src:
        bed = src.read(1)
        profile = src.profile

    np.save(gdir.get_filepath('ice_thickness', '_ref'), ref_it)
    np.save(gdir.get_filepath('ice_mask', '_ref'), ref_ice_mask)


