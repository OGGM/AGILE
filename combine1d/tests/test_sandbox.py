import matplotlib.pyplot as plt
import pytest
import xarray as xr
import numpy as np
import os
import pickle

from combine1d.core.inversion import get_default_inversion_settings
from combine1d.sandbox.create_glaciers_with_measurements import create_idealized_experiments
from combine1d.sandbox.define_idealized_experiment import idealized_experiment
from combine1d.sandbox.create_glaciers_with_measurements import StackedMassBalance

from oggm import cfg

do_plot = False


class TestSandbox:
    def test_create_idealized_experiments(self, test_dir):
        cfg.initialize(logging_level='WARNING')
        cfg.PATHS['working_dir'] = test_dir
        cfg.PARAMS['use_multiprocessing'] = False

        # Size of the map around the glacier.
        prepro_border = 160
        # Degree of processing level.
        from_prepro_level = 2
        # URL of the preprocessed gdirs
        # we use elevation bands flowlines here
        base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/' \
                   'L1-L2_files/elev_bands/'

        glacier_names = ['Baltoro',
                         'Aletsch',
                         'Artesonraju',
                         'Peyto'
                         ]

        cfg.PARAMS['use_multiprocessing'] = False
        cfg.PARAMS['cfl_number'] = 0.5
        gdirs = create_idealized_experiments(glacier_names,
                                             prepro_border=prepro_border,
                                             from_prepro_level=from_prepro_level,
                                             base_url=base_url, )

        def resulting_run_test(filesuffix, ys, ye):
            assert gdir.has_file('model_diagnostics',
                                 filesuffix=filesuffix)
            fp = gdir.get_filepath('model_diagnostics',
                                   filesuffix=filesuffix)
            with xr.open_dataset(fp) as ds_diag:
                ds_diag = ds_diag.load()

            assert ds_diag.time[0] == ys
            assert ds_diag.time[-1] == ye
            assert np.all(np.isfinite(ds_diag.volume_m3))
            assert np.all(np.isfinite(ds_diag.area_m2))
            assert np.all(np.isfinite(ds_diag.length_m))

        # test that the file of the creation spinup exist
        for gdir in gdirs:
            gdir.has_file('model_diagnostics',
                          filesuffix='_combine_creation_spinup')

        # test that resulting true run dataset contain the complete period
        for gdir in gdirs:
            resulting_run_test(filesuffix='_combine_true_total_run',
                               ys=1980,
                               ye=2020)

        # test that the resulting gdirs contain the future climate file
        for gdir in gdirs:
            assert gdir.has_file('gcm_data', filesuffix='_BCC-CSM2-MR_ssp370')
            resulting_run_test(filesuffix='_combine_true_future',
                               ys=2020,
                               ye=2101)

        # test that the resulting gdirs contain the oggm default run files
        for gdir in gdirs:
            resulting_run_test(filesuffix='_oggm_dynamic_past',
                               ys=1980,
                               ye=2020)
            assert gdir.has_file('fl_diagnostics',
                                 filesuffix='_oggm_dynamic_past')
            resulting_run_test(filesuffix='_oggm_dynamic_future',
                               ys=2020,
                               ye=2101)

            resulting_run_test(filesuffix='_oggm_static_past',
                               ys=1980,
                               ye=2020)
            assert gdir.has_file('fl_diagnostics',
                                 filesuffix='_oggm_static_past')
            resulting_run_test(filesuffix='_oggm_static_future',
                               ys=2020,
                               ye=2101)

        # test that the resulting gdirs contain the oggm default statistics
        def all_stats_finite(ds, use_year, is_static=False):
            for var in ds.keys():
                for year in ds[var].keys():
                    if is_static:
                        # in static case no velocity is available at rgi date
                        if var == 'us:myr-1':
                            if year == '2000':
                                continue
                    if use_year:
                        for stat in ds[var][year].keys():
                            assert np.all(np.isfinite(ds[var][year][stat]))
                    else:
                        # here year actually is used as stat (e.g. 'rmsd')
                        assert np.all(np.isfinite(ds[var][year]))

        for gdir in gdirs:
            fp = os.path.join(gdir.dir,
                              'default_oggm_statistics.pkl')
            with open(fp, 'rb') as handle:
                ds_default_stats = pickle.load(handle)

            for stat_key in ds_default_stats.keys():
                stat_suffixes = stat_key.split('_')[-1]
                pure_key = stat_key.removesuffix('_' + stat_suffixes)
                if pure_key in ['observations_stats', 'controls_stats',
                                'past_evol_stats', 'today_state_stats',
                                'future_evol_stats', 'past_state_stats']:
                    use_year = False
                    if pure_key in ['observations_stats']:
                        use_year = True
                    all_stats_finite(ds_default_stats[stat_key], use_year,
                                     stat_suffixes=='static')
                else:
                    raise NotImplementedError(f'{stat_key}')

        if do_plot:
            for gdir in gdirs:
                fl_oggm = gdir.read_pickle('model_flowlines')[0]
                fl_consensus = gdir.read_pickle('model_flowlines',
                                                filesuffix='_consensus')[0]
                fl_spinup = gdir.read_pickle('model_flowlines',
                                             filesuffix='_creation_spinup')[0]
                fl_combine_init = \
                    gdir.read_pickle('model_flowlines',
                                     filesuffix='_combine_true_init')[0]
                fl_combine_end = \
                    gdir.read_pickle('model_flowlines',
                                     filesuffix='_combine_true_end')[0]
                fl_combine_first_guess = \
                    gdir.read_pickle('model_flowlines',
                                     filesuffix='_combine_first_guess')[0]

                def get_fl_diagnostics(filesuffix):
                    f = gdir.get_filepath('fl_diagnostics',
                                          filesuffix=filesuffix)
                    with xr.open_dataset(f, group=f'fl_0') as ds:
                        ds = ds.load()
                    return ds

                fl_diag_init = get_fl_diagnostics('_combine_true_dmdt_start')
                fl_diag_start = get_fl_diagnostics('_combine_true_init')
                fl_diag_end = get_fl_diagnostics('_combine_true_end')

                fig = plt.figure(figsize=(15, 10))
                (ax, ax2, ax3, ax4, ax5, ax6) = fig.subplots(6, 1)
                # ax.plot(fl_oggm.dis_on_line, fl_oggm.bed_h,
                # label='OGGM Flowline')
                ax.plot(fl_spinup.dis_on_line, fl_spinup.bed_h,
                        label='Spinup Flowline')
                ax.plot(fl_oggm.dis_on_line, fl_oggm.surface_h,
                        label='OGGM surface_h')
                ax.plot(fl_spinup.dis_on_line, fl_spinup.surface_h,
                        label='Spinup surface_h')
                # ax.plot(fl_consensus.dis_on_line, fl_consensus.surface_h,
                # label='Consensus surface_h')
                ax.plot(fl_combine_init.dis_on_line, fl_combine_init.surface_h,
                        label='Init surface_h')
                ax.plot(fl_combine_end.dis_on_line, fl_combine_end.surface_h,
                        label='END surface_h')
                ax.legend();

                ax2.axhline(0, color='black')
                ax2.plot(fl_spinup.surface_h - fl_combine_init.surface_h,
                         label='positive means retreating '
                               '(spinup to rgi _date)')
                ax2.plot(fl_combine_init.surface_h - fl_combine_end.surface_h,
                         label='rgi_date to end')
                ax2.set_title('delta surface_h')
                ax2.legend()

                ax3.axhline(0, color='black')
                ax3.plot(fl_spinup.bed_h - fl_combine_first_guess.bed_h,
                         label='positive means overdeepening')
                ax3.set_title('delta bed_h')
                ax3.legend()

                ax4.plot(fl_combine_first_guess.is_trapezoid, label='trapez')
                ax4.plot(fl_combine_first_guess.is_rectangular, label='rect')
                ax4.legend()

                ax5.plot(fl_oggm.thick > 0, label='OGGM thick > 0')
                ax5.plot(fl_combine_init.thick > 0,
                         label='COMBINE init thick > 0')
                ax5.legend()

                ax6.plot(fl_diag_init.ice_velocity_myr[-1],
                         label=f'year {fl_diag_init.time[-1].values}')
                ax6.plot(fl_diag_start.ice_velocity_myr[-1],
                         label=f'year {fl_diag_start.time[-1].values} '
                               f'(rgi date)')
                ax6.plot(fl_diag_end.ice_velocity_myr[-1],
                         label=f'year {fl_diag_end.time[-1].values}')
                ax6.set_title('velocity')
                ax6.legend()

                fig.suptitle(gdir.name, fontsize=16)
                fig.tight_layout(pad=1.0)
                plt.show()

    @pytest.mark.parametrize('control_vars',
                             [['area_bed_h', 'lambdas', 'w0_m'],
                              ['bed_h', 'lambdas', 'w0_m']],
                             ids=['area_bed_h', 'bed_h'])
    def test_run_idealized_experiment(self, test_dir, control_vars):

        experiment_glacier = ['Aletsch']

        inversion_settings = get_default_inversion_settings()
        inversion_settings['minimize_options']['maxiter'] = 3

        # add all possible observations to test everything
        inversion_settings['observations']['fl_widths:m'] = {}
        inversion_settings['obs_reg_parameters']['uncertainty']['fl_widths:m'] = 1.
        inversion_settings['observations']['fl_total_area:m2'] = {}
        inversion_settings['obs_reg_parameters']['uncertainty']['fl_total_area:m2'] = 1.
        # extracted from experiment creation
        inversion_settings['observations']['area:km2'] = {}
        inversion_settings['obs_reg_parameters']['uncertainty']['area:km2'] = 1.

        # add all possible control variables
        inversion_settings['control_vars'] = control_vars

        gdirs = idealized_experiment(use_experiment_glaciers=experiment_glacier,
                                     inversion_settings_all=[inversion_settings],
                                     working_dir=test_dir,
                                     output_folder=test_dir,
                                     override_params={'border': 160,
                                                      'cfl_number': 0.5})

        # open final dataset
        fp = os.path.join(test_dir,
                          'Aletsch_COMBINE_inversion_results.pkl')
        with open(fp, 'rb') as handle:
            ds = pickle.load(handle)

        # open default oggm statistics
        fp = os.path.join(gdirs[0].dir,
                          'default_oggm_statistics.pkl')
        with open(fp, 'rb') as handle:
            ds_default_stats = pickle.load(handle)

        # filesuffixes of two different oggm experiments
        for oggm_run_suffix in ['_dynamic', '_static']:
            # test for observation statistics
            ds_key = 'observations_stats'
            ds_key_oggm = ds_key + oggm_run_suffix
            assert ds_key in ds.attrs.keys()
            assert ds_key_oggm in ds_default_stats.keys()
            for obs_key in ds.attrs[ds_key].keys():
                obs_key_name = obs_key.split(':')[0]
                for year_key in ds.attrs[ds_key][obs_key].keys():
                    if obs_key_name in ['fl_surface_h', 'fl_widths']:
                        test_metrics = ['rmsd', 'mean_ad', 'max_ad']
                    elif obs_key_name in ['fl_total_area', 'area', 'dmdtda']:
                        test_metrics = ['diff', 'abs_diff']
                    else:
                        raise NotImplementedError()
                    for metric in test_metrics:
                        assert metric in ds.attrs[ds_key][obs_key][year_key].keys()
                        assert metric in ds_default_stats[ds_key_oggm][obs_key][year_key].keys()
                        assert isinstance(ds.attrs[ds_key][obs_key][year_key][metric],
                                          float)
                        assert isinstance(ds_default_stats[ds_key_oggm][obs_key][year_key][metric],
                                          float)

            # test for control statistics
            ds_key = 'controls_stats'
            ds_key_oggm = ds_key + oggm_run_suffix
            assert ds_key in ds.attrs.keys()
            assert ds_key_oggm in ds_default_stats.keys()
            for control_key in ds.attrs[ds_key].keys():
                if control_key in ['bed_h', 'area_bed_h', 'lambdas', 'w0_m']:
                    test_metrics = ['rmsd', 'mean_ad', 'max_ad']
                    test_default = True
                elif control_key in ['height_shift_spinup']:
                    test_metrics = ['diff', 'abs_diff']
                    test_default = False
                else:
                    raise NotImplementedError()
                for metric in test_metrics:
                    assert metric in ds.attrs[ds_key][control_key].keys()
                    assert isinstance(ds.attrs[ds_key][control_key][metric],
                                      float)
                    if test_default:
                        assert metric in ds_default_stats[ds_key_oggm][control_key].keys()
                        assert isinstance(ds_default_stats[ds_key_oggm][control_key][metric],
                                          float)

            # test the past glacier state statistics
            ds_key = 'past_state_stats'
            ds_key_oggm = ds_key + oggm_run_suffix
            assert ds_key in ds.attrs.keys()
            assert ds_key_oggm in ds_default_stats.keys()
            for var in ['thick', 'area_m2', 'volume_m3']:
                test_metrics = ['rmsd', 'mean_ad', 'max_ad', 'diff']
                for metric in test_metrics:
                    if metric == 'diff':
                        if var != 'thick':
                            assert metric in ds.attrs[ds_key][var].keys()
                            assert metric in ds_default_stats[ds_key_oggm][var].keys()
                            assert isinstance(ds.attrs[ds_key][var][metric],
                                              np.ndarray)
                            assert isinstance(
                                ds_default_stats[ds_key_oggm][var][metric],
                                np.ndarray)
                    else:
                        assert metric in ds.attrs[ds_key][var].keys()
                        assert metric in ds_default_stats[ds_key_oggm][var].keys()
                        assert isinstance(ds.attrs[ds_key][var][metric],
                                          float)
                        assert isinstance(ds_default_stats[ds_key_oggm][var][metric],
                                          float)

            # test the past evolution statistics
            ds_key = 'past_evol_stats'
            ds_key_oggm = ds_key + oggm_run_suffix
            assert ds_key in ds.attrs.keys()
            assert ds_key_oggm in ds_default_stats.keys()
            for var in ['volume_m3', 'area_m2']:
                test_metrics = ['rmsd', 'mean_ad', 'max_ad']
                for metric in test_metrics:
                    assert metric in ds.attrs[ds_key][var].keys()
                    assert metric in ds_default_stats[ds_key_oggm][var].keys()
                    assert isinstance(ds.attrs[ds_key][var][metric],
                                      float)
                    assert isinstance(ds_default_stats[ds_key_oggm][var][metric],
                                      float)

            # test todays glacier state statistics
            ds_key = 'today_state_stats'
            ds_key_oggm = ds_key + oggm_run_suffix
            assert ds_key in ds.attrs.keys()
            assert ds_key_oggm in ds_default_stats.keys()
            for var in ['thick', 'area_m2', 'volume_m3']:
                test_metrics = ['rmsd', 'mean_ad', 'max_ad', 'diff']
                for metric in test_metrics:
                    if metric == 'diff':
                        if var != 'thick':
                            assert metric in ds.attrs[ds_key][var].keys()
                            assert metric in ds_default_stats[ds_key_oggm][var].keys()
                            assert isinstance(ds.attrs[ds_key][var][metric],
                                              np.ndarray)
                            assert isinstance(ds_default_stats[ds_key_oggm][var][metric],
                                              np.ndarray)
                    else:
                        assert metric in ds.attrs[ds_key][var].keys()
                        assert metric in ds_default_stats[ds_key_oggm][var].keys()
                        assert isinstance(ds.attrs[ds_key][var][metric],
                                          float)
                        assert isinstance(ds_default_stats[ds_key_oggm][var][metric],
                                          float)

            # test the future evolution statistics
            ds_key = 'future_evol_stats'
            ds_key_oggm = ds_key + oggm_run_suffix
            assert ds_key in ds.attrs.keys()
            assert ds_key_oggm in ds_default_stats.keys()
            for var in ['volume_m3', 'area_m2']:
                test_metrics = ['rmsd', 'mean_ad', 'max_ad']
                for metric in test_metrics:
                    assert metric in ds.attrs[ds_key][var].keys()
                    assert metric in ds_default_stats[ds_key_oggm][var].keys()
                    assert isinstance(ds.attrs[ds_key][var][metric],
                                      float)
                    assert isinstance(ds_default_stats[ds_key_oggm][var][metric],
                                      float)

    def test_StackedMassBalance(self, test_dir):
        cfg.initialize(logging_level='WARNING')
        cfg.PATHS['working_dir'] = test_dir
        cfg.PARAMS['use_multiprocessing'] = False

        # Size of the map around the glacier.
        prepro_border = 160
        # Degree of processing level.
        from_prepro_level = 2
        # URL of the preprocessed gdirs
        # we use elevation bands flowlines here
        base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/' \
                   'L1-L2_files/elev_bands/'

        glacier_names = ['Aletsch']

        cfg.PARAMS['use_multiprocessing'] = False
        cfg.PARAMS['cfl_number'] = 0.5
        gdirs = create_idealized_experiments(glacier_names,
                                             prepro_border=prepro_border,
                                             from_prepro_level=from_prepro_level,
                                             base_url=base_url, )

        gdir = gdirs[0]

        mb_settings = {'MB1': {'type': 'constant',
                               'years': np.array([1980, 2000])},
                       'MB2': {'type': 'constant',
                               'years': np.array([2000, 2019])}}

        mb_model = StackedMassBalance(gdir=gdir, mb_model_settings=mb_settings)

        assert np.all(np.isin(np.array([1980, 2000]),
                              mb_model._mb_models[0].years))
        assert np.all(np.isin(np.array([2000, 2019]),
                              mb_model._mb_models[1].years))

        test_heights = [3100, 3000, 2500]
        # get_annual
        assert np.all(np.not_equal(
            mb_model.get_annual_mb(test_heights, year=1990),
            mb_model.get_annual_mb(test_heights, year=2010)))
        assert np.all(np.not_equal(
            mb_model.get_annual_mb(test_heights, year=1980),
            mb_model.get_annual_mb(test_heights, year=2000)))
        assert np.all(np.equal(
            mb_model.get_annual_mb(test_heights, year=1980),
            mb_model.get_annual_mb(test_heights, year=1999)))
        assert np.all(np.equal(
            mb_model.get_annual_mb(test_heights, year=2000),
            mb_model.get_annual_mb(test_heights, year=2019)))
        for false_year in [1979, 2020]:
            with pytest.raises(ValueError,
                               match=f'No mb model defined for year '
                                     f'{false_year}'):
                mb_model.get_annual_mb(test_heights, year=false_year)

        # get monthly
        assert np.all(np.not_equal(
            mb_model.get_monthly_mb(test_heights, year=1990),
            mb_model.get_monthly_mb(test_heights, year=2010)))
        assert np.all(np.not_equal(
            mb_model.get_monthly_mb(test_heights, year=1980),
            mb_model.get_monthly_mb(test_heights, year=2000)))
        assert np.all(np.equal(
            mb_model.get_monthly_mb(test_heights, year=1980),
            mb_model.get_monthly_mb(test_heights, year=1999)))
        assert np.all(np.equal(
            mb_model.get_monthly_mb(test_heights, year=2000),
            mb_model.get_monthly_mb(test_heights, year=2019)))
        for false_year in [1979, 2020]:
            with pytest.raises(ValueError,
                               match=f'No mb model defined for year '
                                     f'{false_year}'):
                mb_model.get_monthly_mb(test_heights, year=false_year)
