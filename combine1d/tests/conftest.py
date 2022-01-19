"""Pytest fixtures to be used in other test modules"""
import shutil
import pytest
import os
import numpy as np

from oggm.utils import mkdir
from oggm import cfg, utils, workflow, tasks, global_tasks
from oggm.workflow import execute_entity_task

from combine1d.core.inversion import prepare_for_combine_inversion, get_default_inversion_settings
from combine1d.core.data_logging import initialise_DataLogger
from combine1d.core.cost_function import get_indices_for_unknown_parameters, get_known_parameters


@pytest.fixture(scope='session')
def test_dir():
    tmp_test_dir = os.path.join(cfg.PATHS['test_dir'], 'combine')
    mkdir(tmp_test_dir)
    yield tmp_test_dir
    shutil.rmtree(tmp_test_dir)


@pytest.fixture(scope='class')
def hef_gdir(test_dir):
    cfg.initialize(logging_level='CRITICAL')
    cfg.PARAMS['use_multiple_flowlines'] = False
    cfg.PARAMS['use_rgi_area'] = False
    cfg.PATHS['working_dir'] = test_dir

    rgi_ids = ['RGI60-11.00897']
    gl = utils.get_rgi_glacier_entities(rgi_ids)
    gdirs = workflow.init_glacier_directories(gl, from_prepro_level=1)

    global_tasks.gis_prepro_tasks(gdirs)

    cfg.PARAMS['baseline_climate'] = 'CRU'
    global_tasks.climate_tasks(gdirs,
                               base_url='https://cluster.klima.uni-bremen.de/~oggm/ref_mb_params'
                                        '/oggm_v1.4/RGIV62/CRU/centerlines/qc3/pcp2.5')

    execute_entity_task(tasks.prepare_for_inversion, gdirs,
                        invert_all_trapezoid=True)
    execute_entity_task(tasks.mass_conservation_inversion, gdirs)

    execute_entity_task(tasks.init_present_time_glacier, gdirs)

    return gdirs[0]


@pytest.fixture(scope='session')
def all_supported_control_vars():
    return ['bed_h', 'lambdas', 'w0_m']


@pytest.fixture(scope='function')
def observations():
    return {'fl_surface_h:m': {'2003': []},
            'fl_widths:m': {'2003': []},
            'fl_total_area:m2': {'2003': []},
            'fl_total_area:km2': {'2003': []},
            'area:m2': {'2000': [],
                        '2005': [],
                        '2009': []},
            'area:km2': {'2002': [],
                         '2007': []},
            'dh:m': {'2000-2005': [],
                     '2005-2010': []}
            }


@pytest.fixture(params=[['bed_h'], ['bed_h', 'w0_m'], 'all'],
                ids=['bed_h', 'bed_h & w0_m', 'all'])
def control_vars(request):
    return request.param


@pytest.fixture(params=[None,
                        {'surface_h': {'mb_model': {'type': 'constant',
                                                    'years': np.array([1980, 2000]),
                                                    't_bias': -2}
                                       }
                         },
                        {'height_shift': {'mb_model': {'type': 'constant',
                                                       'years': np.array([1980, 2000]),
                                                       'fg_height_shift': 100}
                                          }
                         }
                        ],
                ids=['No_spinup', 'sfc_h_spinup', 'height_shift_spinup'])
def spinup_option(request):
    return request.param


@pytest.fixture(scope='function')
def data_logger_init(hef_gdir, control_vars, spinup_option,
                     all_supported_control_vars):
    # data_logger after initialisation (before creation of cost function)
    inversion_settings = get_default_inversion_settings(get_doc=False)
    if control_vars == 'all':
        inversion_settings['control_vars'] = all_supported_control_vars
    else:
        inversion_settings['control_vars'] = control_vars
    inversion_settings['spinup_options'] = spinup_option
    prepare_for_combine_inversion(hef_gdir, inversion_settings=inversion_settings,
                                  filesuffix='_combine')
    data_logger = initialise_DataLogger(hef_gdir, inversion_input_filesuffix='_combine')
    return data_logger


@pytest.fixture(scope='function')
def data_logger(data_logger_init):
    data_logger = data_logger_init
    # data_logger to be used in cost function
    known_parameters = get_known_parameters(data_logger)
    data_logger.known_parameters = known_parameters

    parameter_indices = get_indices_for_unknown_parameters(data_logger)
    data_logger.parameter_indices = parameter_indices

    return data_logger
