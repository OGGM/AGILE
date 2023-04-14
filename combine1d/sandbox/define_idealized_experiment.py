import copy
import logging
import warnings
import xarray as xr

from combine1d.sandbox.calculate_statistics import calculate_result_statistics
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
                         logging_level='WORKFLOW',
                         gcm='BCC-CSM2-MR',
                         ssp='ssp370'):
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
                                         base_url=base_url,
                                         gcm=gcm, ssp=ssp)

    print('Finished creation of directories.')

    print('Start experiments:')

    all_experiments = []
    for inv_setting in inversion_settings_all:
        for gdir in gdirs:
            all_experiments.append((gdir, dict(inversion_settings=inv_setting,
                                               output_folder=output_folder)
                                    ))

    workflow.execute_entity_task(conduct_sandbox_inversion, all_experiments,
                                 gcm=gcm, ssp=ssp)

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

    workflow.execute_entity_task(tasks.run_from_climate_data, [gdir],
                                 climate_filename='gcm_data',
                                 climate_input_filesuffix=rid,
                                 init_model_fls=fls_init,
                                 ys=past_evol_mdl.time[-1].values,
                                 output_filesuffix=data_logger.filename +
                                 '_future',
                                 evolution_model=SemiImplicitModel,
                                 )


@entity_task(log, writes=['inversion_input', 'model_flowlines'])
def conduct_sandbox_inversion(gdir, inversion_settings=None,
                              output_folder=None,
                              init_model_fls='_combine_first_guess',
                              gcm='BCC-CSM2-MR',
                              ssp='ssp370'):
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

    add_future_projection_run(gdir, data_logger=data_logger, gcm=gcm, ssp=ssp)

    calculate_result_statistics(gdir,
                                data_logger=data_logger)
