import numpy as np

from combine1d.core.type_conversions import to_numpy_array

from oggm.core.massbalance import LinearMassBalance as oggm_MassBalance
from oggm.core.flowline import FluxBasedModel as oggm_FluxModel
from oggm.core.flowline import RectangularBedFlowline
from oggm.core.flowline import ParabolicBedFlowline
from oggm.core.flowline import TrapezoidalBedFlowline
from oggm import GlacierDirectory
from oggm.tasks import define_glacier_region
import geopandas as gpd
from oggm.utils import get_demo_file
from oggm.core import climate, inversion, centerlines
from oggm import utils
import shapely.geometry as shpg
import copy
from oggm import cfg


def get_first_guess(data_logger):
    fl = data_logger.flowline_init
    parameter_indices = data_logger.parameter_indices
    ice_mask = data_logger.ice_mask

    first_guess = np.empty(data_logger.len_unknown_parameter)
    first_guess[:] = np.nan

    for ind in parameter_indices:
        if ind in ['bed_h', 'w0_m', 'surface_h', 'lambdas']:
            if ind in ['w0_m', 'lambdas']:
                prefix = '_'
            else:
                prefix = ''
            first_guess[parameter_indices[ind]] = getattr(fl, prefix + ind)[ice_mask]
        else:
            raise NotImplementedError('This parameter var is not implemented!')

    return first_guess
