from copy import deepcopy
import numpy as np
import os
import shutil
import salem
import rasterio
#import matplotlib
#matplotlib.use('GTK3Agg')
import oggm.cfg; oggm.cfg.initialize()
from combine2d.sandbox.perform_run import perform_run
from combine2d.sandbox.perform_run import get_borden_inversion_settings
from combine2d.sandbox.perform_run import get_giluwe_inversion_settings
from combine2d.sandbox.perform_run import get_fg_bias_dict
from combine2d.sandbox.perform_run import get_fg_rmse_dict
from combine2d.sandbox.perform_run import get_surf_rmse_dict
from combine2d.sandbox.perform_run import default_bed_measurement_dict
from combine2d.sandbox.perform_run import default_biased_fg_dict
from combine2d.sandbox.perform_run import default_rmsed_fg_dict
from combine2d.sandbox.perform_run import default_surface_noise_dict
# from combine2d.sandbox.perform_run import reg_parameters_set_A
# from combine2d.sandbox.perform_run import reg_parameters_set_B
# from combine2d.sandbox.perform_run import reg_parameters_set_C
from combine2d.sandbox.run_additional_setting_dicts import *
from combine2d.core.utils import NonRGIGlacierDirectory
from combine2d.core.test_cases import Borden, Giluwe
from combine2d.core.visualization import plot_iterative_behaviour
from scipy.signal import convolve2d


basedir = '/home/philipp/extra2'
case = Giluwe
case = Borden
if case.name == 'Borden Peninsula':
    get_my_inversion_settings = get_borden_inversion_settings
    reg_parameters_set = np.array([0.2, 1.25, 2e4, 2e7, 0, 0]) # final
    #reg_parameters_set = np.array([0.2, 1.25, 1e3, 1e6, 0, 0]) # final2
    #reg_parameters_set = np.array([0.2, 1.25, 5e3, 5e6, 0, 0]) # final3
    reg_parameters_set = np.array([0.2, 1.25, 1e3, 1e6, 0, 0])  # improvement1
    reg_parameters_set = np.array([0.2, 1.25, 1e2, 1e5, 0, 0])  # improvement2
    reg_parameters_set = np.array([0.2, 0.2, 1e1, 1e4, 0, 0])  # improvement3
elif case.name == 'Giluwe':
    get_my_inversion_settings = get_giluwe_inversion_settings
    reg_parameters_set = np.array([0.2, 1.25, 1e3, 1e6, 0, 0])
    reg_parameters_set = np.array([1.0, 0, 0, 0, 0, 0])
    reg_parameters_set = np.array([0.2, 1.25, 2e2, 2e5, 0, 0]) # improvement1
    reg_parameters_set = np.array([0.2, 0.75, 1e2, 1e5, 0, 0]) # improvement2

fg_path = '/home/philipp/final/{:s}/fg bias   0/inverted_bed.tiff'.format(
    case.name)
fg_path = os.path.join(basedir, case.name, 'improvement2', 'inverted_bed.tiff')
new_fg_path = os.path.join(basedir, case.name, 'first_guessed_bed.tiff')
shutil.copy(fg_path, new_fg_path)


#gdir = NonRGIGlacierDirectory(case, basedir)
#ref_ice_mask = np.load(gdir.get_filepath('ref_ice_mask'))
#with rasterio.open(gdir.get_filepath('first_guessed_bed')) as src:
#    first_guessed_bed = src.read(1)
#    profile = src.profile#

#smoothed = first_guessed_bed.copy()
#shape = 5
#conv_array = np.ones((shape, shape)) / shape**2  # TODO
#smoothed = convolve2d(first_guessed_bed, conv_array, mode='same')
#first_guessed_bed[ref_ice_mask] = smoothed[ref_ice_mask]
#with rasterio.open(gdir.get_filepath('first_guessed_bed'),
#                   'w', **profile) as dst:
#    dst.write(first_guessed_bed, 1)

identical_twin_inversion_settings = get_my_inversion_settings(
    'improvement3', reg_parameters_set)
idir = perform_run(case, basedir, identical_twin_inversion_settings,
                   create_synthetic_glacier=False,  # True,
                   use_preexisting_fg=True)
# create_synthetic_glacier=True)
