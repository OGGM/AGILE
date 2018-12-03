import os
from copy import deepcopy

import numpy as np

from cobbi.core import gis
from cobbi.core.data_manipulation import add_bed_measurements
from cobbi.core.data_manipulation import add_noise_to_first_guess
from cobbi.core.data_manipulation import create_perlin_noise, add_surface_noise
from cobbi.core.data_manipulation import generate_bed_measurements
from cobbi.core.data_manipulation import take_true_bed_as_first_guess
from cobbi.core.dynamics import create_glacier
from cobbi.core.first_guess import compile_biased_first_guess
from cobbi.core.first_guess import compile_first_guess
from cobbi.core.first_guess import apply_bed_measurements_to_first_guess
from cobbi.core.inversion import InversionDirectory
from cobbi.core.table_creation import create_case_table, eval_identical_twin
from cobbi.core.utils import NonRGIGlacierDirectory
default_minimize_options = {'maxiter': 1000, 'ftol': 1e-7, 'gtol': 1e-8,
                            'disp': True, 'maxcor': 100, 'maxls': 50}
default_biased_fg_dict={'use': False, 'desired_mean_bias':None}
default_rmsed_fg_dict={'use': False, 'desired_rmse': None, 'octaves': 4,
                            'base': 2, 'freq': 3, 'glacier_only': True}
default_surface_noise_dict={'use': False, 'desired_rmse': None, 'octaves': 4,
                            'base': 2, 'freq': 3, 'glacier_only': True}
default_bed_measurement_dict={'use': False, 'measurement_mask': None, 'std': 30,
                              'seed': 0}
default_Giluwe_inversion_settings = {
    'mb_spinup': None,
    'yrs_spinup': 2000,
    'yrs_forward_run': 200,
    'reg_parameters': None,
    'solver': 'L-BFGS-B',
    'minimize_options': deepcopy(default_minimize_options),
    'inversion_subdir': None,
    'fg_shape_factor': 1,
    'fg_slope_cutoff_angle': 5,
    'fg_interp_boundary': False,
    'bounds_min_max': (None, 600)
    }
default_Giluwe_inversion_settings2 = {
    'mb_spinup': None,
    'yrs_spinup': 2000,
    'yrs_forward_run': 200,
    'reg_parameters': None,
    'solver': 'L-BFGS-B',
    'minimize_options': deepcopy(default_minimize_options),
    'inversion_subdir': None,
    'fg_shape_factor': 1,
    'fg_slope_cutoff_angle': 5,
    'fg_interp_boundary': True,
    'bounds_min_max': (None, 600)
    }
default_Borden_inversion_settings = {
    'mb_spinup': None,
    'yrs_spinup': 2000,
    'yrs_forward_run': 200,
    'reg_parameters': None,
    'solver': 'L-BFGS-B',
    'minimize_options': deepcopy(default_minimize_options),
    'inversion_subdir': None,
    'fg_shape_factor': 1,
    'fg_slope_cutoff_angle': 2.5,
    'fg_interp_boundary': True,
    'fg_min_height': -30,
    'bounds_min_max': (None, 1000)
    }

reg_parameters_set_A = np.array([0.2, 1e-4, 100, 1e6, 0.0, 0.0])
reg_parameters_set_B = np.array([0.2, 1e-1, 1000, 1e7, 0.0, 0.0])
reg_parameters_set_C = np.array([0.2, 1e-1, 1000, 1e7, 1e5, 0.0])

def get_giluwe_inversion_settings(inversion_subdir, reg_parameters):
    return get_inversion_settings(inversion_subdir, reg_parameters,
                                  deepcopy(default_Giluwe_inversion_settings))


def get_borden_inversion_settings(inversion_subdir, reg_parameters):
    return get_inversion_settings(inversion_subdir, reg_parameters,
                                  deepcopy(default_Borden_inversion_settings))


def get_inversion_settings(inversion_subdir, reg_parameters, base_inv_settings):
    base_inv_settings['inversion_subdir'] = inversion_subdir
    base_inv_settings['reg_parameters'] = reg_parameters
    return base_inv_settings


def get_fg_bias_dict(bias=0):
   return {'use': True, 'desired_mean_bias': bias}


def get_fg_rmse_dict(rmse=0, base=2):
   return {'use': True, 'desired_rmse': rmse, 'octaves': 4,
                            'base': base, 'freq': 3, 'glacier_only': True}


def get_surf_rmse_dict(rmse=0, base=2):
   return {'use': True, 'desired_rmse': rmse, 'octaves': 4,
                            'base': base, 'freq': 3, 'glacier_only': True}


def perform_run(case, basedir, inversion_settings,
                create_synthetic_glacier=True,
                biased_fg_dict=None,
                rmsed_fg_dict=None,
                surface_noise_dict=None,
                bed_measurements_dict=None,
                use_preexisting_fg=False,
                use_measurements_in_fg=False):
    if bed_measurements_dict is None:
        bed_measurements_dict = deepcopy(default_bed_measurement_dict)
    if surface_noise_dict is None:
        surface_noise_dict = deepcopy(default_surface_noise_dict)
    if biased_fg_dict is None:
        biased_fg_dict = deepcopy(default_biased_fg_dict)
    if rmsed_fg_dict is None:
        rmsed_fg_dict = deepcopy(default_rmsed_fg_dict)
    gdir = NonRGIGlacierDirectory(case, basedir)
    gdir.write_inversion_settings(**inversion_settings)
    if create_synthetic_glacier:
        gis.define_nonrgi_glacier_region(gdir)
        create_glacier(gdir)
        create_case_table(gdir)

    # potentially need to cleanup
    if os.path.exists(gdir.get_filepath('bed_measurements')):
        os.remove(gdir.get_filepath('bed_measurements'))
    if os.path.exists(gdir.get_filepath('dem_noise')):
        os.remove(gdir.get_filepath('dem_noise'))

    # Think about bed_measurements now to be potentially able to introduce them
    #  to first guess
    if bed_measurements_dict['use']:
        np.random.seed(bed_measurements_dict['seed'])
        bed_measurements = generate_bed_measurements(gdir,
                                                     bed_measurements_dict[
                                                         'measurement_mask'],
                                                     std=bed_measurements_dict[
                                                         'std'])
        add_bed_measurements(gdir, bed_measurements)

    # First guess options
    if not use_preexisting_fg:
        compile_first_guess(gdir)
    if biased_fg_dict['use']:
        compile_biased_first_guess(gdir, biased_fg_dict['desired_mean_bias'])
    if rmsed_fg_dict['use']:
        # desired_rmse = 5
        # desired_rmse = 20.2038        # results in actual rmse of 20
        noise = create_perlin_noise(gdir, **rmsed_fg_dict)
        take_true_bed_as_first_guess(gdir)
        add_noise_to_first_guess(gdir, noise)

    if use_measurements_in_fg:
        apply_bed_measurements_to_first_guess(gdir)

    # Maybe some surface noise?
    if surface_noise_dict['use']:
        noise = create_perlin_noise(gdir, **surface_noise_dict)
        add_surface_noise(gdir, noise)

    idir = InversionDirectory(gdir)
    res = idir.run_minimize()
    eval_identical_twin(idir)

    # copy this script to inversion directory for reproducibility
    #path_to_file = '/home/philipp/COBBI/cobbi/sandbox/Giluwe/bed_measurements.py'
    #fname = os.path.split(path_to_file)[-1]
    #if not os.path.exists(idir.get_current_basedir()):
    #    os.makedirs(idir.get_current_basedir(), exist_ok=True)
    #shutil.copy(path_to_file, os.path.join(idir.get_current_basedir(), fname))
    return idir
