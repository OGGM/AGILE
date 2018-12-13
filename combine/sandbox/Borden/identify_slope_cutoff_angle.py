import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from combine.core import gis, test_cases
from combine.core.utils import NonRGIGlacierDirectory
from combine.core.first_guess import compile_first_guess
from combine.core.inversion import InversionDirectory
from combine.core.dynamics import create_glacier
from combine.core.cost_function import create_cost_func
from combine.core.inversion import InversionDirectory
from combine.core import data_logging
import salem
from combine.core.dynamics import run_forward_core
from oggm import cfg

np.random.seed(0)  # needs to be fixed for reproducible results with noise

cfg.initialize()

basedir = '/path/to/example'
basedir = '/data/philipp/thesis_test2/Borden/identify_sca'

# TODO: think about IceThicknesses for case Giluwe
# Choose a case
case = test_cases.Borden
gdir = NonRGIGlacierDirectory(case, basedir)
# only needed once:
gis.define_nonrgi_glacier_region(gdir)

# create settings for inversion
lambdas = np.zeros(4)

minimize_options = {
    'maxiter': 300,
    'ftol': 0.5e-3,
    #'xtol': 1e-30,
    'gtol': 1e-4,
    #'maxcor': 5,
    #'maxls': 10,
    'disp': True
}

gdir.write_inversion_settings(mb_spinup=None,
                              yrs_spinup=2000,
                              yrs_forward_run=200,
                              reg_parameters=lambdas,
                              solver='L-BFGS-B',
                              minimize_options=minimize_options,
                              inversion_subdir='0',
                              fg_shape_factor=1.,
                              fg_slope_cutoff_angle=2.5,
                              fg_min_height=-30,
                              bounds_min_max=(2, 1000)
                              )

# Optional, if not reset=True and already ran once
# only needed once:
#create_glacier(gdir)
fg = compile_first_guess(gdir)

#idir = InversionDirectory(gdir)

mb = gdir.inversion_settings['mb_forward_run']
if mb is None:
    mb = case.get_mb_model()

guessed_bed = torch.tensor(fg, dtype=torch.float, requires_grad=False)
spinup_surf = salem.GeoTiff(gdir.get_filepath('spinup_dem')).get_vardata()
ref_surf = salem.GeoTiff(gdir.get_filepath('ref_dem')).get_vardata()
ice_mask = np.load(gdir.get_filepath('ref_ice_mask'))
# run model forward
init_ice_thick = spinup_surf - guessed_bed

model_surf = run_forward_core(gdir.inversion_settings['yrs_forward_run'],
                              guessed_bed, case.dx, case.get_mb_model(),
                              init_ice_thick).detach().numpy()
print('Bias: ' + str(np.sum(model_surf - ref_surf)/ice_mask.sum()))
plt.imshow(model_surf - ref_surf)
plt.show()
#dl = data_logging.load_pickle(idir.get_current_basedir() + '/data_logger.pkl')

bed = salem.GeoTiff(gdir.get_filepath('dem')).get_vardata()
print('Bias: ' + str(np.sum(fg - bed)/ice_mask.sum()))
plt.imshow(fg - bed)
plt.show()

print('end')