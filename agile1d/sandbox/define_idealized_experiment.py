import copy
import logging
import warnings
import xarray as xr

from agile1d.sandbox.calculate_statistics import calculate_result_statistics
from agile1d.sandbox.create_glaciers_with_measurements import \
    create_idealized_experiments
from agile1d.core.inversion import prepare_for_agile_inversion, \
    agile_inversion
from oggm import cfg, utils, workflow, tasks
from oggm import entity_task
from oggm.core.flowline import SemiImplicitModel

# Module logger
log = logging.getLogger(__name__)


def idealized_experiment(use_experiment_glaciers,
                         use_experiment_glacier_states,
                         inversion_settings_all,
                         working_dir, output_folder,
                         inversion_settings_individual=None,
                         params_file=None, override_params=None,
                         init_model_fls='_oggm_first_guess',
                         logging_level='WORKFLOW',
                         gcm='BCC-CSM2-MR',
                         ssp='ssp370',
                         print_statistic=True):
    # Local paths
    if override_params is None:
        override_params = {}

    if any(state not in ['equilibrium', 'retreating', 'advancing']
           for state in use_experiment_glacier_states):
        raise NotImplementedError("Glacier states could only be one of "
                                  "['equilibrium', 'retreating', 'advancing']"
                                  ", but given was "
                                  f"{use_experiment_glacier_states}.")

    utils.mkdir(working_dir)
    override_params['working_dir'] = working_dir

    utils.mkdir(output_folder)

    cfg.initialize(file=params_file, params=override_params,
                   logging_level=logging_level, future=True)

    # store model geometry for hydro output
    cfg.PARAMS['store_model_geometry'] = True

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
                                         use_experiment_glacier_states,
                                         prepro_border=cfg.PARAMS['border'],
                                         from_prepro_level=from_prepro_level,
                                         base_url=base_url,
                                         gcm=gcm, ssp=ssp)

    print('Finished creation of directories.')

    print('Start experiments:')

    all_experiments = []
    for inv_setting in inversion_settings_all:
        for gdir in gdirs:
            for glacier_state in use_experiment_glacier_states:
                inv_setting_use = copy.deepcopy(inv_setting)
                if inversion_settings_individual is not None:
                    try:
                        individual_setting = \
                            inversion_settings_individual[gdir.rgi_id]
                        for key in individual_setting.keys():
                            inv_setting_use[key] = individual_setting[key]
                    except KeyError:
                        # if no individual settings for this glacier move on
                        pass
                # add glacier state to outputfilesuffix
                inv_setting_use['experiment_description'] = (
                    f"{glacier_state}_"
                    f"{inv_setting_use['experiment_description']}")
                all_experiments.append((gdir,
                                        dict(
                                            glacier_state=glacier_state,
                                            inversion_settings=inv_setting_use,
                                            output_folder=output_folder)
                                        ))

    workflow.execute_entity_task(conduct_sandbox_inversion, all_experiments,
                                 init_model_fls=init_model_fls,
                                 gcm=gcm, ssp=ssp,
                                 print_statistic=print_statistic)

    print('Experiments finished!')
    return gdirs


def add_future_projection_run(gdir, data_logger, gcm='BCC-CSM2-MR',
                              ssp='ssp370'):
    """add a future projection to analyse the influence of inverison result on
    the future"""

    fls_init = data_logger.flowlines[-1]
    fp = gdir.get_filepath('model_diagnostics',
                           filesuffix=data_logger.filename)
    with xr.open_dataset(fp) as ds_diag:
        past_evol_mdl = ds_diag.load()

    rid = '_{}_{}'.format(gcm, ssp)

    workflow.execute_entity_task(tasks.run_with_hydro, [gdir],
                                 run_task=tasks.run_from_climate_data,
                                 climate_filename='gcm_data',
                                 climate_input_filesuffix=rid,
                                 init_model_fls=fls_init,
                                 ys=past_evol_mdl.time[-1].values,
                                 output_filesuffix=data_logger.filename +
                                 '_future',
                                 evolution_model=SemiImplicitModel,
                                 )


@entity_task(log, writes=['inversion_input', 'model_flowlines'])
def conduct_sandbox_inversion(gdir, glacier_state, inversion_settings=None,
                              output_folder=None,
                              init_model_fls='_oggm_first_guess',
                              gcm='BCC-CSM2-MR',
                              ssp='ssp370',
                              print_statistic=True,
                              return_data_logger=False):
    """TODO"""

    # add glacier state to first guess
    init_model_fls = f'{init_model_fls}_{glacier_state}'

    # load mb_model settings from gdir depending on glacier state
    inversion_settings['mb_models_settings'] = \
        gdir.read_pickle('inversion_input',
                         filesuffix='_agile_mb_models_'
                                    f'{glacier_state}')

    # if perfect spinup we must add glacier state to string
    spn_option = list(inversion_settings['spinup_options'].keys())[0]
    if spn_option in ['perfect_sfc_h', 'perfect_thickness', 'perfect_section']:
        inversion_settings['spinup_options'][spn_option] = \
            inversion_settings['spinup_options'][spn_option] + f'_{glacier_state}'

    # also for perfect_bed_h we need to add the glacier state
    if 'perfect_bed_h' in list(inversion_settings['spinup_options'].keys()):
        inversion_settings['spinup_options']['perfect_bed_h'] = \
            inversion_settings['spinup_options']['perfect_bed_h'] + f'_{glacier_state}'

    # load the observations to use from gdir
    all_measurements = gdir.read_pickle('inversion_input',
                                        filesuffix='_agile_measurements_'
                                                   f'{glacier_state}')
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

    inversion_filesuffix = f"_{inversion_settings['experiment_description']}"

    prepare_for_agile_inversion(
        gdir, inversion_settings=inversion_settings,
        filesuffix=inversion_filesuffix)

    data_logger = agile_inversion(gdir,
                                  inversion_input_filesuffix=
                                  inversion_filesuffix,
                                  init_model_filesuffix=None,
                                  init_model_fls=init_model_fls,
                                  climate_filename='climate_historical',
                                  climate_filesuffix='',
                                  output_filesuffix=
                                  f'_agile_output{inversion_filesuffix}',
                                  output_filepath=output_folder,
                                  save_dataset=True,
                                  give_data_logger_back=True)

    add_future_projection_run(gdir, data_logger=data_logger, gcm=gcm, ssp=ssp)

    calculate_result_statistics(gdir,
                                glacier_state=glacier_state,
                                data_logger=data_logger,
                                print_statistic=print_statistic)

    if return_data_logger:
        return data_logger
