import copy
import logging
from combine1d.sandbox.create_glaciers_with_measurements import \
    create_idealized_experiments
from combine1d.core.inversion import prepare_for_combine_inversion, \
    combine_inversion
from oggm import cfg, utils, workflow
from oggm import entity_task

# Module logger
log = logging.getLogger(__name__)


def idealized_experiment(use_experiment_glaciers=None,
                         inversion_settings_all=None,
                         working_dir='', output_folder='',
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
    prepro_border = 160
    cfg.PARAMS['border'] = prepro_border
    # Degree of processing level.
    from_prepro_level = 3
    # URL of the preprocessed gdirs
    # we use elevation bands flowlines here
    base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/' \
               'L3-L5_files/CRU/elev_bands/qc3/pcp2.5/no_match/'

    gdirs = create_idealized_experiments(use_experiment_glaciers,
                                         prepro_border=prepro_border,
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

    workflow.execute_entity_task(conduct_combine_inversion, all_experiments)

    print('Experiments finished!')


@entity_task(log, writes=['inversion_input', 'model_flowlines'])
def conduct_combine_inversion(gdir, inversion_settings=None,
                              output_folder=None):
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

    combine_inversion(gdir,
                      inversion_input_filesuffix=
                      inversion_settings['experiment_description'],
                      init_model_filesuffix=None,
                      init_model_fls='_combine_first_guess',
                      climate_filename='climate_historical',
                      climate_filesuffix='',
                      output_filesuffix='_combine_output_' +
                                        inversion_settings['experiment_description'],
                      output_filepath=output_folder,
                      save_dataset=True)
