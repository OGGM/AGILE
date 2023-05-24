import os
import pickle
import numpy as np
import xarray as xr
import pytest

from combine1d.core.first_guess import get_first_guess
from combine1d.core.inversion import (prepare_for_combine_inversion,
                                      get_control_var_bounds,
                                      get_default_inversion_settings,
                                      combine_inversion)
from combine1d.core.data_logging import initialise_DataLogger
from combine1d.core.cost_function import create_cost_fct

pytestmark = pytest.mark.filterwarnings("ignore:<class "
                                        "'combine1d.core.torch_interp1d.Interp1d'> "
                                        "should not be instantiated.:DeprecationWarning")


class TestInversion:
    def test_default_initialisation(self, hef_gdir):
        # check if error is thrown if gdir is not prepared
        with pytest.raises(AttributeError,
                           match='inversion_input_combine file not found!'):
            initialise_DataLogger(hef_gdir)

        # add default inversion_input_combine and try again
        prepare_for_combine_inversion(hef_gdir, inversion_settings=None,
                                      filesuffix='_combine')
        data_logger = initialise_DataLogger(
            hef_gdir, inversion_input_filesuffix='_combine')

        assert len(data_logger.is_trapezoid) == (sum(data_logger.is_trapezoid) +
                                                 sum(data_logger.is_rectangular) +
                                                 sum(data_logger.is_parabolic))
        cost_fct = create_cost_fct(data_logger)

        first_guess = get_first_guess(data_logger)

        cost, grad = cost_fct(first_guess)
        assert cost.size == 1
        assert type(cost) == np.ndarray
        assert type(grad) == np.ndarray

    def test_get_control_var_bounds(self, data_logger):
        create_cost_fct(data_logger)

        first_guess = get_first_guess(data_logger)

        bounds = get_control_var_bounds(data_logger)

        assert len(bounds) == data_logger.len_unknown_parameter
        for i, val in enumerate(first_guess):
            assert bounds[i] != 0
            if bounds[i][0] is not None:
                assert val >= bounds[i][0]
            if bounds[i][1] is not None:
                assert val <= bounds[i][1]

    @pytest.mark.parametrize('control_vars', [['bed_h'], ['bed_h', 'w0_m'],
                                              ['area_bed_h'],
                                              'all'],
                             ids=['bed_h', 'bed_h & w0_m', 'area_bed_h',
                                  'all'])
    @pytest.mark.parametrize(
        'spinup_options', [None,
                           {'surface_h': {'mb_model':
                                              {'type': 'constant',
                                               'years': np.array(
                                                   [1980, 2000]),
                                               't_bias': -2}
                                          }
                            },
                           {'height_shift': {'mb_model':
                                                 {'type': 'constant',
                                                  'years': np.array(
                                                      [1980, 2000]),
                                                  'fg_height_shift': -100},
                                             'spinup_length_yrs': 20
                                             }
                            }
                           ],
        ids=['No_spinup', 'sfc_h_spinup', 'height_shift_spinup'])
    @pytest.mark.parametrize('dynamic_model', ['flux_based', 'implicit'],
                             ids=['flux_based', 'implicit'])
    def test_combine_inversion(self, hef_gdir, control_vars, spinup_options,
                               dynamic_model, all_supported_control_vars):
        # Final integration test that the inversion runs with no errors, also
        # test the saving of the final past evolution
        inversion_settings = get_default_inversion_settings(get_doc=False)
        inversion_settings['minimize_options'] = {'maxiter': 2,
                                                  'ftol': 1e-7,
                                                  'gtol': 1e-8,
                                                  'disp': False,
                                                  'maxcor': 50,
                                                  'maxls': 50,
                                                  'maxfun': 10}
        if control_vars == 'all':
            control_vars = all_supported_control_vars
        inversion_settings['control_vars'] = control_vars
        inversion_settings['spinup_options'] = spinup_options
        inversion_settings['dynamic_model'] = dynamic_model

        # add an artificial measurement for dmdtda
        measure_key = 'dmdtda:kg m-2 yr-1'
        inversion_settings['observations'][measure_key]['2000-2019'] = -1.

        prepare_for_combine_inversion(hef_gdir,
                                      inversion_settings=inversion_settings,
                                      filesuffix='_combine')

        data_logger = combine_inversion(hef_gdir, give_data_logger_back=True,
                                        save_past_evolution=True)

        # test if data_logger contains data
        assert data_logger.minimize_message is not None
        assert data_logger.minimize_status is not None
        assert data_logger.costs is not None
        assert data_logger.c_terms is not None
        assert data_logger.c_terms_description is not None
        assert data_logger.time_needed is not None
        assert data_logger.grads is not None
        assert data_logger.flowlines is not None
        assert data_logger.end_time is not None
        assert data_logger.known_parameters is not None
        assert data_logger.len_unknown_parameter is not None
        assert data_logger.parameter_indices is not None
        assert data_logger.unknown_parameters is not None
        assert data_logger.observations_mdl is not None

        # test if data during minimisation is saved to disk
        # look for xarray version and check
        fp = os.path.join(hef_gdir.dir,
                          'Hintereisferner_COMBINE_inversion_results_'
                          'xr_version.txt')
        with open(fp, 'r') as f:
            saved_version = f.readlines()[1].split(' ')[2]
        assert xr.__version__ == saved_version

        fp = os.path.join(hef_gdir.dir,
                          'Hintereisferner_COMBINE_inversion_results.pkl')
        with open(fp, 'rb') as handle:
            ds_saved = pickle.load(handle)
        assert type(ds_saved) == xr.core.dataset.Dataset
        assert ds_saved.costs[0] > ds_saved.costs[-1]

        # test past evolution
        assert hef_gdir.has_file(
            'model_diagnostics',
            filesuffix='Hintereisferner_COMBINE_inversion_results')
        assert not hef_gdir.has_file('model_diagnostics',
                                     filesuffix='_temporary_run')
        fp = hef_gdir.get_filepath(
            'model_diagnostics',
            filesuffix='Hintereisferner_COMBINE_inversion_results')
        with xr.open_dataset(fp) as ds:
            ds = ds.load()

        # include some test that all the years are there
        assert ds.time[0] == 1980
        assert ds.time[-1] == 2020
        assert np.all(np.isfinite(ds.volume_m3))
        assert np.all(np.isfinite(ds.area_m2))
        assert np.all(np.isfinite(ds.length_m))
