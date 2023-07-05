"""Pytest fixtures to be used in other test modules"""
import shutil
import pytest
import os
import numpy as np

from oggm.utils import mkdir
from oggm import cfg, utils, workflow, tasks, global_tasks
from oggm.workflow import execute_entity_task
from oggm.shop.bedtopo import add_consensus_thickness

from combine1d.core.inversion import prepare_for_combine_inversion, get_default_inversion_settings, \
    get_control_var_bounds
from combine1d.core.data_logging import initialise_DataLogger
from combine1d.core.cost_function import get_indices_for_unknown_parameters, get_known_parameters, \
    define_scaling_terms
from distutils import dir_util


def pytest_configure(config):
    for marker in ["test_env"]:
        config.addinivalue_line("markers", marker)


def pytest_addoption(parser):
    parser.addoption("--run-test-env", metavar="ENVNAME", default="",
                     help="Run only specified test env")


def pytest_collection_modifyitems(config, items):
    run_test_env = config.getoption("--run-test-env")

    test_env_marker = pytest.mark.skip(reason="only test_env=%s tests are run"
                                              % run_test_env)

    for item in items:
        if run_test_env:
            test_env = item.get_closest_marker("test_env")
            if not test_env or test_env.args[0] != run_test_env:
                item.add_marker(test_env_marker)


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
    cfg.PARAMS['border'] = 160
    cfg.PARAMS['continue_on_error'] = False
    cfg.PARAMS['baseline_climate'] = 'W5E5'

    rgi_ids = ['RGI60-11.00897']
    gdirs = workflow.init_glacier_directories(
            rgi_ids, from_prepro_level=3,
            prepro_base_url='https://cluster.klima.uni-bremen.de/~oggm/gdirs/'
                            'oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5/')

    cfg.PARAMS['downstream_line_shape'] = 'parabola'
    execute_entity_task(tasks.init_present_time_glacier, gdirs,
                        filesuffix='_parabola')

    cfg.PARAMS['downstream_line_shape'] = 'trapezoidal'
    execute_entity_task(tasks.init_present_time_glacier, gdirs,
                        filesuffix='_trapezoidal')

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
            'dmdtda:kg m-2 yr-1': {'2000-2005': [],
                                   '2005-2010': []},
            'dmdtda:kg yr-1': {'2010-2015': []},
            'us:myr-1': {'2003': []},
            }


@pytest.fixture(params=[['bed_h'], ['bed_h', 'w0_m'], ['area_bed_h'], 'all'],
                ids=['bed_h', 'bed_h & w0_m', 'area_bed_h', 'all'])
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
                                                       'fg_height_shift': -100},
                                          'spinup_length_yrs': 20
                                          }
                         },
                        {'section': {'extra_grid_points': 10,
                                     'limits': (0.75, 1.25),
                                     'fg_years': 0,
                                     }}
                        ],
                ids=['No_spinup', 'sfc_h_spinup', 'height_shift_spinup',
                     'section'])
def spinup_option(request):
    return request.param


@pytest.fixture(params=['flux_based', 'implicit'],
                ids=['flux_based', 'implicit'])
def dynamic_models(request):
    return request.param


@pytest.fixture(scope='function')
def data_logger_init(hef_gdir, control_vars, spinup_option, dynamic_models,
                     all_supported_control_vars):
    # data_logger after initialisation (before creation of cost function)
    inversion_settings = get_default_inversion_settings(get_doc=False)
    if control_vars == 'all':
        inversion_settings['control_vars'] = all_supported_control_vars
    else:
        inversion_settings['control_vars'] = control_vars
    inversion_settings['dynamic_model'] = dynamic_models
    inversion_settings['spinup_options'] = spinup_option
    if spinup_option == 'section':
        inversion_settings['regularisation_terms'] = {'smoothed_bed': 1.,
                                                      'smoothed_flux': 10.}
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

    define_scaling_terms(data_logger)

    get_control_var_bounds(data_logger)

    return data_logger


@pytest.fixture(scope='function')
def data_dir(tmpdir, request):
    """
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    """
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir
