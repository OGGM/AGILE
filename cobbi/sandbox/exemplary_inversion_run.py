import torch
import matplotlib.pyplot as plt
import numpy as np

from cobbi.core import gis, test_cases
from cobbi.core.utils import NonRGIGlacierDirectory
from cobbi.core.first_guess import compile_first_guess
from cobbi.core.inversion import InversionDirectory
from cobbi.core.dynamics import create_glacier
from oggm import cfg

cfg.initialize()

basedir = '/path/to/example'
basedir = '/media/philipp/Daten/test/Giluwe1'


# Choose a case
case = test_cases.Giluwe
gdir = NonRGIGlacierDirectory(case, basedir)
gis.define_nonrgi_glacier_region(gdir)

# create settings for out inversion
lambdas = np.zeros(10)
minimize_options = {
    'maxiter': 300,
    'ftol': 1e-3,
    #'xtol': 1e-30,
    'gtol': 1e-4,
    #'maxcor': 5,
    #'maxls': 10,
    'disp': True
}
# TODO: bounds would be an additional option for minimization?
gdir.write_inversion_settings(mb_spinup=None,
                              yrs_spinup=2000,
                              yrs_forward_run=200,
                              reg_parameters=lambdas,
                              solver='L-BFGS-B',
                              minimize_options=minimize_options)

# Optional, if not reset=True and already ran once
create_glacier(gdir)