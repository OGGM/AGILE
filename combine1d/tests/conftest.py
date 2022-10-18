"""Pytest fixtures to be used in other test modules"""
import shutil
import pytest
import os
import numpy as np

from oggm.utils import mkdir
from oggm import cfg, utils, workflow, tasks, global_tasks
from oggm.workflow import execute_entity_task
from oggm.shop.bedtopo import add_consensus_thickness

from combine1d.core.inversion import prepare_for_combine_inversion, get_default_inversion_settings
from combine1d.core.data_logging import initialise_DataLogger
from combine1d.core.cost_function import get_indices_for_unknown_parameters, get_known_parameters
from distutils import dir_util


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
    gl = utils.get_rgi_glacier_entities(rgi_ids)
    gdirs = workflow.init_glacier_directories(
        gl, from_prepro_level=1,
        prepro_base_url='https://cluster.klima.uni-bremen.de/~oggm/gdirs/'
                        'oggm_v1.6/L1-L2_files/elev_bands/')

    # create elevation band flowline
    execute_entity_task(tasks.simple_glacier_masks, gdirs)
    execute_entity_task(add_consensus_thickness, gdirs)
    vn = 'consensus_ice_thickness'
    execute_entity_task(tasks.elevation_band_flowline,
                        gdirs, bin_variables=vn)
    execute_entity_task(tasks.fixed_dx_elevation_band_flowline,
                        gdirs, bin_variables=vn)

    # downstream line
    task_list = [
        tasks.compute_downstream_line,
        tasks.compute_downstream_bedshape,
    ]
    for task in task_list:
        workflow.execute_entity_task(task, gdirs)

    # climate
    execute_entity_task(tasks.process_climate_data, gdirs)
    if cfg.PARAMS['climate_qc_months'] > 0:
        execute_entity_task(tasks.historical_climate_qc, gdirs)

    # mb model calibration to geodetic mass balance
    utils.get_geodetic_mb_dataframe()  # Small optim to avoid concurrency
    execute_entity_task(tasks.mu_star_calibration_from_geodetic_mb, gdirs)
    execute_entity_task(tasks.apparent_mb_from_any_mb, gdirs)

    # Inversion
    workflow.calibrate_inversion_from_consensus(gdirs,
                                                apply_fs_on_mismatch=True,
                                                error_on_mismatch=False,
                                                filter_inversion_output=True)

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
            'dh:m': {'2000-2005': [],
                     '2005-2010': []},
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
                                                       'fg_height_shift': -100}
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
