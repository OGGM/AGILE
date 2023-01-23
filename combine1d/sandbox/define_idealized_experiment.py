import copy
import logging
import warnings
import pickle
import os
import numpy as np
import xarray as xr
from combine1d.sandbox.create_glaciers_with_measurements import \
    create_idealized_experiments
from combine1d.core.inversion import prepare_for_combine_inversion, \
    combine_inversion
from oggm import cfg, utils, workflow, tasks
from oggm import entity_task
from oggm.core.flowline import SemiImplicitModel

# Module logger
log = logging.getLogger(__name__)


def idealized_experiment(use_experiment_glaciers,
                         inversion_settings_all,
                         working_dir, output_folder,
                         params_file=None, override_params=None,
                         logging_level='WORKFLOW'):
    # Local paths
    if override_params is None:
        override_params = {}

    utils.mkdir(working_dir)
    override_params['working_dir'] = working_dir

    utils.mkdir(output_folder)

    cfg.initialize(file=params_file, params=override_params,
                   logging_level=logging_level, future=True)

    print('Create glacier directories with idealized glaciers:')
    # Size of the map around the glacier.
    if cfg.PARAMS['border'] != 160:
        msg = (f"cfg.PARAMS['border'] is {cfg.PARAMS['border']} but experiments were "
               f"created with border=160!")
        warnings.warn(msg)
        raise RuntimeError(msg)
    # Degree of processing level.
    from_prepro_level = 2
    # URL of the preprocessed gdirs
    # we use elevation bands flowlines here
    base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/' \
               'L1-L2_files/elev_bands/'

    gdirs = create_idealized_experiments(use_experiment_glaciers,
                                         prepro_border=cfg.PARAMS['border'],
                                         from_prepro_level=from_prepro_level,
                                         base_url=base_url, )

    print('Finished creation of directories.')

    print('Start experiments:')

    all_experiments = []
    for inv_setting in inversion_settings_all:
        for gdir in gdirs:
            all_experiments.append((gdir, dict(inversion_settings=inv_setting,
                                               output_folder=output_folder)
                                    ))

    workflow.execute_entity_task(conduct_sandbox_inversion, all_experiments)

    print('Experiments finished!')
    return gdirs


def add_future_projection_run(gdir, data_logger):
    """add a future projection to analyse the influence of inverison result on
    the future"""

    fls_init = data_logger.flowlines[-1]

    ssp = 'ssp370'
    GCM = 'BCC-CSM2-MR'
    rid = '_{}_{}'.format(GCM, ssp)

    workflow.execute_entity_task(tasks.run_from_climate_data, [gdir],
                                 climate_filename='gcm_data',
                                 climate_input_filesuffix=rid,
                                 init_model_fls=fls_init,
                                 output_filesuffix=data_logger.filename +
                                                   '_future',
                                 evolution_model=SemiImplicitModel,
                                 )


def calculate_result_statistics(gdir, data_logger):
    """calculate some statistics of the result for analysis"""

    # open the dataset of the run to add our calculated statistics
    fp = os.path.join(data_logger.output_filepath,
                      data_logger.filename + '.pkl')
    with open(fp, 'rb') as handle:
        ds = pickle.load(handle)

    # define which statistic we want to compute for different types of data
    def add_1d_stats(x, y):
        rms_deviation = utils.rmsd(x, y)
        mean_absolute_deviation = utils.mad(x, y)
        max_absolute_deviation = np.max(np.abs(x - y))

        return {'rmsd': float(rms_deviation),
                'mean_ad': float(mean_absolute_deviation),
                'max_ad': float(max_absolute_deviation)}

    def add_0d_stats(x, y):
        return {'diff': float(x - y),
                'abs_diff': float(np.abs(x - y))}

    def add_2d_stats(x, y):
        return_dict = add_1d_stats(x, y)
        return_dict['diff'] = x - y
        return return_dict

    # how well do we match the observations -----------------------------------
    obs_mdl = ds.observations_mdl[-1].values.item()
    obs_given = ds.attrs['observations']
    obs_stats = {}
    for obs_key in obs_mdl.keys():
        obs_stats[obs_key] = {}
        # the key consists of name and unit, here extract only name
        obs_key_name = obs_key.split(':')[0]
        for year_key in obs_mdl[obs_key].keys():
            if obs_key_name in ['fl_surface_h', 'fl_widths']:
                # here we compare data along the flowline
                obs_stats[obs_key][year_key] = add_1d_stats(
                    obs_mdl[obs_key][year_key],
                    obs_given[obs_key][year_key])
            elif obs_key_name in ['fl_total_area', 'area', 'dmdtda']:
                obs_stats[obs_key][year_key] = add_0d_stats(
                    obs_mdl[obs_key][year_key],
                    obs_given[obs_key][year_key])
            else:
                raise NotImplementedError(f'Observation {obs_key} not '
                                          f'implemented!')
    ds.attrs['observations_stats'] = obs_stats

    # how well do we match the control variables ------------------------------
    control_indices = ds.attrs['parameter_indices']
    controls_mdl = {}
    for control_var in control_indices.keys():
        controls_mdl[control_var] = \
            ds.unknown_parameters[-1][control_indices[control_var]].values
    controls_true = {}
    fls_true = gdir.read_pickle('model_flowlines',
                                filesuffix='_combine_true_init')[0]
    for control_var in controls_mdl:
        if control_var in ['bed_h', 'area_bed_h']:
            if control_var == 'area_bed_h':
                obs_scal_key = 'observations_for_scaling'
                scaling_fct = \
                    ds.attrs[obs_scal_key]['fl_widths:m'][ds.ice_mask.values]
            else:
                scaling_fct = np.array([1])
            controls_true[control_var] = \
                fls_true.bed_h[ds.ice_mask.values] / scaling_fct * \
                np.mean(scaling_fct)
        elif control_var in ['lambdas', 'w0_m']:
            controls_true[control_var] = \
                getattr(fls_true, f'_{control_var}')[ds.ice_mask.values]
        elif control_var in ['height_shift_spinup']:
            controls_true[control_var] = controls_mdl[control_var]
        else:
            raise NotImplementedError(f'control {control_var} not implemented!')

    controls_stats = {}
    for control_var in controls_mdl.keys():
        if control_var in ['bed_h', 'area_bed_h', 'lambdas', 'w0_m']:
            controls_stats[control_var] = add_1d_stats(
                controls_mdl[control_var],
                controls_true[control_var])
        elif control_var in ['height_shift_spinup']:
            controls_stats[control_var] = add_0d_stats(
                controls_mdl[control_var],
                controls_true[control_var])
        else:
            raise NotImplementedError(f'Control variable {control_var} not '
                                      f'implemented!')

    ds.attrs['controls_stats'] = controls_stats

    # how well do we match the past glacier evolution -------------------------
    fp = gdir.get_filepath('model_diagnostics',
                           filesuffix=data_logger.filename)
    with xr.open_dataset(fp) as ds_diag:
        past_evol_mdl = ds_diag.load()
    fp = gdir.get_filepath('model_diagnostics',
                           filesuffix='_combine_total_run')
    with xr.open_dataset(fp) as ds_diag:
        past_evol_true = ds_diag.load()

    past_evol_stats = {}
    for var in ['volume_m3', 'area_m2']:
        past_evol_stats[var] = add_1d_stats(
                past_evol_mdl[var],
                past_evol_true[var])

    ds.attrs['past_evol_stats'] = past_evol_stats

    # how well do we match today's glacier state ------------------------------
    fls_end_mdl = data_logger.flowlines[-1]
    fls_end_true = gdir.read_pickle('model_flowlines',
                                    filesuffix='_combine_true_end')[0]

    today_state_stats = {}
    for var in ['thick', 'area_m2', 'volume_m3']:
        if var in ['thick']:
            today_state_stats[var] = add_2d_stats(
                getattr(fls_end_mdl, var),
                getattr(fls_end_true, var))
        elif var == 'area_m2':
            def get_area(fl):
                return np.where(fl.thick > 0, fl.widths_m, 0) * fl.dx_meter
            today_state_stats[var] = add_2d_stats(
                get_area(fls_end_mdl),
                get_area(fls_end_true))
        elif var == 'volume_m3':
            def get_volume(fl):
                return fl.section * fl.dx_meter
            today_state_stats[var] = add_2d_stats(
                get_volume(fls_end_mdl),
                get_volume(fls_end_true))
        else:
            raise NotImplementedError('')

    ds.attrs['today_state_stats'] = today_state_stats

    # how well do we match the future glacier evolution -----------------------
    fp = gdir.get_filepath('model_diagnostics',
                           filesuffix=data_logger.filename + '_future')
    with xr.open_dataset(fp) as ds_diag:
        future_evol_mdl = ds_diag.load()
    fp = gdir.get_filepath('model_diagnostics',
                           filesuffix='_combine_true_future')
    with xr.open_dataset(fp) as ds_diag:
        future_evol_true = ds_diag.load()

    future_evol_stats = {}
    for var in ['volume_m3', 'area_m2']:
        future_evol_stats[var] = add_1d_stats(
            future_evol_mdl[var],
            future_evol_true[var])

    ds.attrs['future_evol_stats'] = future_evol_stats

    # save final dataset as pickle again
    out = os.path.join(data_logger.output_filepath,
                       data_logger.filename + '.pkl')
    with open(out, 'wb') as handle:
        pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)


@entity_task(log, writes=['inversion_input', 'model_flowlines'])
def conduct_sandbox_inversion(gdir, inversion_settings=None,
                              output_folder=None,
                              init_model_fls='_combine_first_guess'):
    """TODO"""

    # check if mb_model_settings should be loaded from gdir
    if inversion_settings['mb_models_settings'] == 'load_from_gdir':
        inversion_settings['mb_models_settings'] = \
            gdir.read_pickle('inversion_input',
                             filesuffix='_combine_mb_models')

    # load the observations to use from gdir
    all_measurements = gdir.read_pickle('inversion_input',
                                        filesuffix='_combine_measurements')
    used_measurements = copy.deepcopy(inversion_settings['observations'])
    inversion_settings['observations'] = {}
    for msr in used_measurements:
        if msr in ['fl_surface_h:m', 'fl_widths:m', 'fl_total_area:m2',
                   'fl_total_area:km2']:
            inversion_settings['observations'][msr] = {}
        elif msr in all_measurements.keys():
            inversion_settings['observations'][msr] = all_measurements[msr]
        else:
            raise RuntimeError(f'No observation for {msr} available!')

    prepare_for_combine_inversion(gdir,
                                  inversion_settings=inversion_settings,
                                  filesuffix=
                                  inversion_settings['experiment_description'])

    data_logger = combine_inversion(gdir,
                                    inversion_input_filesuffix=
                                    inversion_settings['experiment_description'],
                                    init_model_filesuffix=None,
                                    init_model_fls=init_model_fls,
                                    climate_filename='climate_historical',
                                    climate_filesuffix='',
                                    output_filesuffix='_combine_output_' +
                                                      inversion_settings['experiment_description'],
                                    output_filepath=output_folder,
                                    save_dataset=True,
                                    give_data_logger_back=True)

    add_future_projection_run(gdir, data_logger=data_logger)

    calculate_result_statistics(gdir,
                                data_logger=data_logger)
