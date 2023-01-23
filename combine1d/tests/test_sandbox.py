import matplotlib.pyplot as plt
import pytest
import xarray as xr
import numpy as np
import os
import pickle

from combine1d.core.inversion import get_default_inversion_settings
from combine1d.sandbox.create_glaciers_with_measurements import create_idealized_experiments
from combine1d.sandbox.define_idealized_experiment import idealized_experiment

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
                         'Hintereisferner',
                         'Aletsch',
                         'Artesonraju',
                         'Shallap'
                         ]

        cfg.PARAMS['use_multiprocessing'] = False
        cfg.PARAMS['cfl_number'] = 0.5
        gdirs = create_idealized_experiments(glacier_names,
                                             prepro_border=prepro_border,
                                             from_prepro_level=from_prepro_level,
                                             base_url=base_url, )

        # test that resulting dataset contain the complete experiment period
        for gdir in gdirs:
            fp = gdir.get_filepath('model_diagnostics',
                                   filesuffix='_combine_total_run')
            with xr.open_dataset(fp) as ds_diag:
                ds_diag = ds_diag.load()

            assert ds_diag.time[0] == 1980
            assert ds_diag.time[-1] == 2019
            assert np.all(np.isfinite(ds_diag.volume_m3))
            assert np.all(np.isfinite(ds_diag.area_m2))
            assert np.all(np.isfinite(ds_diag.length_m))

        # test that the resulting gdirs contain the future climate file
        for gdir in gdirs:
            assert gdir.has_file('gcm_data', filesuffix='_BCC-CSM2-MR_ssp370')
            assert gdir.has_file('model_diagnostics',
                                 filesuffix='_combine_true_future')

        if do_plot:
            for gdir in gdirs:
                fl_oggm = gdir.read_pickle('model_flowlines')[0]
                fl_consensus = gdir.read_pickle('model_flowlines',
                                                filesuffix='_consensus')[0]
                fl_spinup = gdir.read_pickle('model_flowlines',
                                             filesuffix='_spinup')[0]
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

                fl_diag_init = get_fl_diagnostics('_combine_spinup')
                fl_diag_start = get_fl_diagnostics('_combine_true_init')
                fl_diag_end = get_fl_diagnostics('_combine_end')

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

        experiment_glacier = ['Hintereisferner']

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

        idealized_experiment(use_experiment_glaciers=experiment_glacier,
                             inversion_settings_all=[inversion_settings],
                             working_dir=test_dir,
                             output_folder=test_dir,
                             override_params={'border': 160,
                                              'cfl_number': 0.5})

        # open final dataset
        fp = os.path.join(test_dir,
                          'Hintereisferner_COMBINE_inversion_results.pkl')
        with open(fp, 'rb') as handle:
            ds = pickle.load(handle)

        # test for observation statistics
        ds_key = 'observations_stats'
        assert ds_key in ds.attrs.keys()
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
                    assert isinstance(ds.attrs[ds_key][obs_key][year_key][metric],
                                      float)

        # test for control statistics
        ds_key = 'controls_stats'
        assert ds_key in ds.attrs.keys()
        for control_key in ds.attrs[ds_key].keys():
            if control_key in ['bed_h', 'area_bed_h', 'lambdas', 'w0_m']:
                test_metrics = ['rmsd', 'mean_ad', 'max_ad']
            elif control_key in ['height_shift_spinup']:
                test_metrics = ['diff', 'abs_diff']
            else:
                raise NotImplementedError()
            for metric in test_metrics:
                assert metric in ds.attrs[ds_key][control_key].keys()
                assert isinstance(ds.attrs[ds_key][control_key][metric],
                                  float)

        # test the past evolution statistics
        ds_key = 'past_evol_stats'
        assert ds_key in ds.attrs.keys()
        for var in ['volume_m3', 'area_m2']:
            test_metrics = ['rmsd', 'mean_ad', 'max_ad']
            for metric in test_metrics:
                assert metric in ds.attrs[ds_key][var].keys()
                assert isinstance(ds.attrs[ds_key][var][metric],
                                  float)

        # test todays glacier state statistics
        ds_key = 'today_state_stats'
        assert ds_key in ds.attrs.keys()
        for var in ['thick', 'area_m2', 'volume_m3']:
            test_metrics = ['rmsd', 'mean_ad', 'max_ad', 'diff']
            for metric in test_metrics:
                if metric == 'diff':
                    if var != 'thick':
                        assert metric in ds.attrs[ds_key][var].keys()
                        assert isinstance(ds.attrs[ds_key][var][metric],
                                          np.ndarray)
                else:
                    assert metric in ds.attrs[ds_key][var].keys()
                    assert isinstance(ds.attrs[ds_key][var][metric],
                                      float)

        # test the future evolution statistics
        ds_key = 'future_evol_stats'
        assert ds_key in ds.attrs.keys()
        for var in ['volume_m3', 'area_m2']:
            test_metrics = ['rmsd', 'mean_ad', 'max_ad']
            for metric in test_metrics:
                assert metric in ds.attrs[ds_key][var].keys()
                assert isinstance(ds.attrs[ds_key][var][metric],
                                  float)
