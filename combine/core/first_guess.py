import torch
import numpy as np
import salem
import rasterio
from oggm import cfg, entity_task
import logging
from scipy.signal import convolve2d
from combine.core.arithmetics import compute_inner_mask

# -------------------------------
# Further initialization / extended import tasks
# Module logger
log = logging.getLogger(__name__)


def first_guess(surf, ice_mask, dx, slope_cutoff_angle=5.0, factor=1):
    """
    First guess for ice thickness and therefore also the bed height. Based on
    GlabTop, see:

    Paul, F., and A. Linsbauer (2012), Modeling of glacier bed topography from
    glacier outlines, central branch lines and a DEM,
    Int. J. Geograph. Inf. Sci., doi:10.1080/13658816.2011.627859.

    Frey, H. et al.: Estimating the volume of glaciers in the
    Himalayan–Karakoram region using different methods, The Cryosphere,
    8, 2313–2333, doi:10.5194/tc-8-2313-2014, 2014.

    Parameters
    ----------
    surf: ndarray
        'observed' surface height of the domain as 2d-Array. (unit: [m])
    ice_mask: ndarray
        2d mask for the whole surface masking non-glaciated areas. Same
        shape as surf
    dx: float
        resolution of the surface grid. (unit: [m])
    slope_cutoff_angle: float
        determines minimal slope used for inversion. Smaller slopes are
        capped. Small slopes can otherwise cause unrealistically large
        thicknesses. Default value (from papers: 5.0) (unit: [°])
    factor: float
        shape factor for calculation of shear stress on centerline.
        See e.g. Cuffey and Patterson p. 342. Default value used for valley
        glaciers is 0.8 (Paul and Linsbauer) but should be more like 1 in
        case of ice caps (depending on how strong the ice cap is governed
        by single flow arms. (unit: [])

    Returns
    -------
    estimated bed height as ndarray with same shape as surf

    """

    # think about smoothing first guess?
    # -> anyway not exactly as in paper, only rough estimate
    # --> possible improvement, but not for now

    glacier_surf = np.ma.masked_array(surf, np.logical_not(ice_mask))
    h_difference = (glacier_surf.max() - glacier_surf.min()) / 1e3
    h_difference = min(1.6, h_difference)
    tau = 0.005 + 1.598 * h_difference - 0.435 * h_difference ** 2
    if h_difference >= 1.6:
        tau = 1.5
    tau = tau * 1e5  # convert from bar to Pa

    f = factor

    gradients = np.gradient(surf, dx)
    sqrd_gradients = gradients[0] ** 2 + gradients[1] ** 2
    sqrd_gradients[sqrd_gradients == 0] = 1e-7
    gradients = np.sqrt(sqrd_gradients)

    sin_angle = np.sin(np.arctan(gradients))

    min_sin_angle = np.sin(np.deg2rad(slope_cutoff_angle))

    sin_angle[sin_angle < min_sin_angle] = min_sin_angle

    thick = tau / (cfg.PARAMS['ice_density'] * cfg.G * f * sin_angle)
    bed = surf - thick * ice_mask
    return bed


# @entity_task(log, writes=['first_guessed_bed'])
def compile_first_guess(gdir):
    """
    Runs first guess on a glacier directory and saves result to the
    glacier_directory

    Name dedicated to @fmaussion ;-)

    dictionary inversion settings in gdir should contain:
    case for case.dx: float
        resolution of the surface grid. (unit: [m])
    slope_cutoff_angle: float
        determines minimal slope used for inversion. Smaller slopes are
        capped. Small slopes can otherwise cause unrealistically large
        thicknesses. Default value (from papers: 5.0) (unit: [°])
    factor: float
        shape factor for calculation of shear stress on centerline.
        See e.g. Cuffey and Patterson p. 342. Default value used for valley
        glaciers is 0.8 (Paul and Linsbauer) but should be more like 1 in
        case of ice caps (depending on how strong the ice cap is governed
        by single flow arms. (unit: [])
    fg_min_height: float
        if included, first guess is cut at this height, i.e. no lower values
        are allowed

    Parameters
    ----------
    gdir: NonRGIGlacierDirectory
        GlacierDirectory containing 'dem_ref' as a reference surface and
        'dem' as the

    Returns
    -------
    estimated bed height as ndarray with same shape as surface array of domain

    """
    inv_settings = gdir.inversion_settings
    case = inv_settings['case']
    slope_cutoff_angle = inv_settings['fg_slope_cutoff_angle']
    factor = inv_settings['fg_shape_factor']
    ice_mask = np.load(gdir.get_filepath('ref_ice_mask'))

    with rasterio.open(gdir.get_filepath('ref_dem')) as src:
        surf = src.read(1)
        profile = src.profile

    first_guessed_bed = first_guess(surf, ice_mask, case.dx,
                                    slope_cutoff_angle,
                                    factor)

    if 'fg_min_height' in inv_settings and \
            inv_settings['fg_min_height'] is not None:
        first_guessed_bed = np.clip(first_guessed_bed,
                                    inv_settings['fg_min_height'], None)
    if 'fg_interp_boundary' in inv_settings and \
        inv_settings['fg_interp_boundary']:
        first_guessed_bed = interpolate_all_boundary(first_guessed_bed,
                                                     ice_mask)

    with rasterio.open(gdir.get_filepath('first_guessed_bed'),
                       'w', **profile) as dst:
        dst.write(first_guessed_bed, 1)

    return first_guessed_bed

def interpolate_all_boundary(first_guessed_bed, ice_mask):
    # TODO: ugly, documentation
    #conv_array = np.ones((3,3))
    inner_mask = compute_inner_mask(ice_mask)
    boundary = np.logical_xor(ice_mask, inner_mask)
    indices = np.argwhere(boundary)
    for ind in indices:
        bed_vals = first_guessed_bed[
                   ind[0] - 1:ind[0] + 2,
                   ind[1] - 1:ind[1] + 2]

        bound_vals = boundary[
                   ind[0] - 1:ind[0] + 2,
                   ind[1] - 1:ind[1] + 2]

        #mask_vals = ice_mask[
        #           ind[0] - 1:ind[0] + 1,
        #           ind[0] - 1:ind[0] + 1]

        masked_bed_vals = np.ma.masked_array(bed_vals,
                                             mask=bound_vals)
        # TODO: replace by proper interpolation
        first_guessed_bed[ind[0], ind[1]] = np.mean(masked_bed_vals)
    return first_guessed_bed


def apply_bed_measurements_to_first_guess(gdir):
    bed_measurements = np.load(gdir.get_filepath('bed_measurements'))
    with rasterio.open(gdir.get_filepath('first_guessed_bed')) as src:
        fg = src.read(1)
        profile = src.profile

    fg[~bed_measurements.mask] = bed_measurements[~bed_measurements.mask]

    with rasterio.open(gdir.get_filepath('first_guessed_bed'),
                       'w', **profile) as dst:
        dst.write(fg, 1)


def compile_biased_first_guess(gdir, desired_mean_bias):
    ref_ice_thickness = np.load(gdir.get_filepath('ref_ice_thickness'))
    ref_ice_mask = np.load(gdir.get_filepath('ref_ice_mask'))
    true_bed = salem.GeoTiff(gdir.get_filepath('dem')).get_vardata()

    with rasterio.open(gdir.get_filepath('ref_dem')) as src:
        surf = src.read(1)
        profile = src.profile

    gamma = desired_mean_bias / (ref_ice_thickness.sum() / ref_ice_mask.sum())

    if gamma > 1:
        raise AttributeError('Given desired mean bias not applicable;'
                             'ice volume exceeded')

    first_guessed_bed = true_bed + gamma * ref_ice_thickness

    with rasterio.open(gdir.get_filepath('first_guessed_bed'),
                       'w', **profile) as dst:
        dst.write(first_guessed_bed, 1)

    return first_guessed_bed


def compile_rmsed_first_guess(gdir, desired_rmse):
    ref_ice_thickness = np.load(gdir.get_filepath('ref_ice_thickness'))
    ref_ice_mask = np.load(gdir.get_filepath('ref_ice_mask'))
    true_bed = salem.GeoTiff(gdir.get_filepath('dem')).get_vardata()

    with rasterio.open(gdir.get_filepath('ref_dem')) as src:
        surf = src.read(1)
        profile = src.profile

    std = desired_rmse
    perturbation = std * np.random.randn(*ref_ice_mask.shape) * ref_ice_mask

    if np.any(perturbation > ref_ice_thickness) > 1:
        raise AttributeError('Perturbation peek through surface; Aborting')

    first_guessed_bed = true_bed + perturbation
    first_guessed_bed.dtype = np.float32

    with rasterio.open(gdir.get_filepath('first_guessed_bed'),
                       'w', **profile) as dst:
        dst.write(first_guessed_bed, 1)

    return first_guessed_bed


def first_guess_flowline(fl, ice_mask, dx, slope_cutoff_angle=5.0, factor=1):
    """
    First guess for ice thickness and therefore also the bed height. Based on
    GlabTop, see:

    Paul, F., and A. Linsbauer (2012), Modeling of glacier bed topography from
    glacier outlines, central branch lines and a DEM,
    Int. J. Geograph. Inf. Sci., doi:10.1080/13658816.2011.627859.

    Frey, H. et al.: Estimating the volume of glaciers in the
    Himalayan–Karakoram region using different methods, The Cryosphere,
    8, 2313–2333, doi:10.5194/tc-8-2313-2014, 2014.

    Parameters
    ----------
    fl: oggm.core.flowline or oggm.core.centerlines.Centerline
        'observed' surface height along a flowline. (unit: [m])
    ice_mask: ndarray
        index which indicating ice with TRUE and no ice withh FALSE, same
        length as fls
    dx: float
        resolution of the surface grid. (unit: [m])
    slope_cutoff_angle: float
        determines minimal slope used for inversion. Smaller slopes are
        capped. Small slopes can otherwise cause unrealistically large
        thicknesses. Default value (from papers: 5.0) (unit: [°])
    factor: float
        shape factor for calculation of shear stress on centerline.
        See e.g. Cuffey and Patterson p. 342. Default value used for valley
        glaciers is 0.8 (Paul and Linsbauer) but should be more like 1 in
        case of ice caps (depending on how strong the ice cap is governed
        by single flow arms. (unit: [])

    Returns
    -------
    estimated bed height as ndarray with same length as fls

    """

    # think about smoothing first guess?
    # -> anyway not exactly as in paper, only rough estimate
    # --> possible improvement, but not for now

    glacier_surf = fl.surface_h[ice_mask]
    h_difference = (glacier_surf.max() - glacier_surf.min()) / 1e3
    h_difference = min(1.6, h_difference)
    tau = 0.005 + 1.598 * h_difference - 0.435 * h_difference ** 2
    if h_difference >= 1.6:
        tau = 1.5
    tau = tau * 1e5  # convert from bar to Pa

    f = factor

    gradients = np.gradient(glacier_surf, dx)

    sin_angle = np.sin(np.arctan(gradients))

    min_sin_angle = np.sin(np.deg2rad(slope_cutoff_angle))

    sin_angle[sin_angle < min_sin_angle] = min_sin_angle

    thick = tau / (cfg.PARAMS['ice_density'] * cfg.G * f * sin_angle)
    bed = glacier_surf - thick
    return bed


def compile_first_guess_flowline(gdir):
    """
    NOT IMPLEMENTED
    Runs first guess on a glacier directory and saves result to the
    glacier_directory

    Name dedicated to @fmaussion ;-)

    dictionary inversion settings in gdir should contain:
    case for case.dx: float
        resolution of the surface grid. (unit: [m])
    slope_cutoff_angle: float
        determines minimal slope used for inversion. Smaller slopes are
        capped. Small slopes can otherwise cause unrealistically large
        thicknesses. Default value (from papers: 5.0) (unit: [°])
    factor: float
        shape factor for calculation of shear stress on centerline.
        See e.g. Cuffey and Patterson p. 342. Default value used for valley
        glaciers is 0.8 (Paul and Linsbauer) but should be more like 1 in
        case of ice caps (depending on how strong the ice cap is governed
        by single flow arms. (unit: [])
    fg_min_height: float
        if included, first guess is cut at this height, i.e. no lower values
        are allowed

    Parameters
    ----------
    gdir: NonRGIGlacierDirectory
        GlacierDirectory containing 'dem_ref' as a reference surface and
        'dem' as the

    Returns
    -------
    estimated bed height as ndarray with same shape as surface array of domain

    """
    inv_settings = gdir.inversion_settings
    case = inv_settings['case']
    slope_cutoff_angle = inv_settings['fg_slope_cutoff_angle']
    factor = inv_settings['fg_shape_factor']
    ice_mask = np.load(gdir.get_filepath('ref_ice_mask'))

    with rasterio.open(gdir.get_filepath('ref_dem')) as src:
        surf = src.read(1)
        profile = src.profile

    first_guessed_bed = first_guess(surf, ice_mask, case.dx,
                                    slope_cutoff_angle,
                                    factor)

    if 'fg_min_height' in inv_settings and \
            inv_settings['fg_min_height'] is not None:
        first_guessed_bed = np.clip(first_guessed_bed,
                                    inv_settings['fg_min_height'], None)
    if 'fg_interp_boundary' in inv_settings and \
        inv_settings['fg_interp_boundary']:
        first_guessed_bed = interpolate_all_boundary(first_guessed_bed,
                                                     ice_mask)

    with rasterio.open(gdir.get_filepath('first_guessed_bed'),
                       'w', **profile) as dst:
        dst.write(first_guessed_bed, 1)

    return first_guessed_bed