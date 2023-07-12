import numpy as np
import rasterio
import shutil
import salem
from noise import pnoise2
from scipy.ndimage import interpolation
from agile2d.core.arithmetics import RMSE


def add_noise_to_first_guess(gdir, noise, cut_noise=True, min_ice_thick=5):
    """
    Adds noise to the first guess. Saves this to the file with the first guess
    and also saves the applied noise.

    Parameters
    ----------
    gdir: NonRGIGlacierDirectory
        GlacierDirectory containing the first guess
    noise: ndarray
        noise to apply to the first guess
    cut_noise: bool
        whether or not the noise should be cut to not penetrate the surface
        and require a minimum ice thickness if applied to first guess
    min_ice_thick: float
        minimum ice thickness, only applied if noise is cut
    """

    fg_filepath = gdir.get_filepath('first_guessed_bed')

    with rasterio.open(fg_filepath) as src:
        first_guessed_bed = src.read(1)
        profile = src.profile

    if cut_noise:
        ref_ice_mask = np.load(gdir.get_filepath('ref_ice_mask'))
        desired_rmse = RMSE(noise, 0, ref_ice_mask)
        ref_surf = salem.GeoTiff(gdir.get_filepath('ref_dem')).get_vardata()

        penetrating = (first_guessed_bed + noise - min_ice_thick > ref_surf)
        penetrating *= ref_ice_mask

        noise = np.where(penetrating,
                         ref_surf - first_guessed_bed - min_ice_thick,
                         noise)
        # TODO: will result in problems -> iteratively?
        rmse = RMSE(noise, 0, ref_ice_mask)
        print('desired rmse: {:g}\\rmse after cutting: {:g}'.format(
            desired_rmse, rmse))
        # noise *= desired_rmse / rmse  # rescale to desired RMSE
        # if np.any(first_guessed_bed + noise > ref_surf):
        #    raise ValueError('First guess is found to penetrate ice surface; '
        #                     'Aborting')

    first_guessed_bed = first_guessed_bed + noise

    profile['dtype'] = 'float64'
    with rasterio.open(fg_filepath, 'w', **profile) as dst:
        dst.write(first_guessed_bed, 1)

    np.save(gdir.get_filepath('first_guessed_bed_noise'), noise)


def take_true_bed_as_first_guess(gdir):
    """
    Copies the true bed (in 'dem.tiff') to the file used as a first guess.
    This method can be used to
    1) ensure optimization with true first guess converges immediately
    2) add noise to this ideal first guess later on
       (method: add_noise_to_first_guess) to assess convergence in dependence of
       surface noise

    Parameters
    ----------
    gdir: NonRGIGlacierDirectory
        GlacierDirectory containing the first guess and the true bed
    """
    shutil.copy(gdir.get_filepath('dem'),
                gdir.get_filepath('first_guessed_bed'))


def add_surface_noise(gdir, noise):
    """
    Adds noise which is applied during inversion to the spinup surface and
    reference surface.
    Saves the applied noise.

    Parameters
    ----------
    gdir: NonRGIGlacierDirectory
        GlacierDirectory containing the spinup surface and reference surface
    noise: ndarray
        noise to apply to the surfaces
    """
    np.save(gdir.get_filepath('dem_noise'), noise)


def create_noise(gdir, std=3, zoom=-1, glacier_only=True):
    """
    Creates noise that can be added to the glacier surface or bed for
    evaluation of behaviour with uncertainties.

    Parameters
    ----------
    gdir: NonRGIGlacierDirectory
        GlacierDirectory containing an ice mask giving the required shape of
        noise and optionally used for restricting noise to ice areas only
    std: float
        standard deviation used for creation of noise
    zoom: float
        if > 0, scipys zoom functionality is used on the noise array to get
        noise with spatial dependency
    glacier_only: bool
        whether or not the noise should cover only the ice area

    Returns
    -------
    ndarray of noise
    """
    ref_ice_mask = np.load(gdir.get_filepath('ref_ice_mask'))
    noise = std * np.random.standard_normal(ref_ice_mask.shape)
    if zoom > 0:
        noise = interpolation.zoom(noise, zoom)[0:ref_ice_mask.shape[0],
                                                0:ref_ice_mask.shape[1]]
    if glacier_only:
        noise = noise * ref_ice_mask

    return noise


def create_perlin_noise(gdir, desired_rmse=5., octaves=1, base=1., freq=8.0,
                        glacier_only=True, use=True):
    """
    TODO: Documentation

    Parameters
    ----------
    gdir
    desired_rmse
    octaves
    base
    freq
    glacier_only

    Returns
    -------

    """

    ref_ice_mask = np.load(gdir.get_filepath('ref_ice_mask'))
    max_y, max_x = ref_ice_mask.shape

    noise = np.zeros((max_y, max_x))
    for y in range(max_y):
        for x in range(max_x):
            # use pnoise here, but snoise would be a reasonable choice as well
            noise[y, x] = pnoise2(x / freq, y / freq, octaves=octaves,
                                  base=base)

    if glacier_only:
        noise = noise * ref_ice_mask
        rmse = RMSE(noise, 0, ref_ice_mask)
    else:
        rmse = RMSE(noise, 0)

    noise *= desired_rmse / rmse

    return noise


def add_bed_measurements(gdir, bed_measurements):
    # TODO: typecheck: if type(bed_measurements) is not np.ma.m
    bed_measurements.dump(gdir.get_filepath('bed_measurements'))


def generate_bed_measurements(gdir, bed_measurements_mask, std=0):
    true_bed = salem.GeoTiff(gdir.get_filepath('dem')).get_vardata()
    noise = std * np.random.randn(*true_bed.shape)
    bed_measurements = (true_bed + noise) * bed_measurements_mask
    bed_measurements[np.logical_not(bed_measurements_mask)] = -np.inf
    bed_measurements = np.ma.masked_array(bed_measurements,
                                          mask=np.logical_not(
                                              bed_measurements_mask))
    print('Actual RMSE of bed measurements: {:g}'.format(
        RMSE(bed_measurements, true_bed)))
    # TODO: apply std scaling after masking ...?
    # TODO: investigate deviations of RMSE from numpys std
    return bed_measurements
