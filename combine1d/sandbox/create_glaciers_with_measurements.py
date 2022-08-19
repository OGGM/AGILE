import copy
import logging
import pandas as pd
import numpy as np
from oggm.cfg import SEC_IN_YEAR
from scipy.optimize import brentq

from oggm import cfg, workflow, tasks, entity_task
from oggm.core.flowline import MixedBedFlowline, FluxBasedModel
from oggm.core.massbalance import ConstantMassBalance, MultipleFlowlineMassBalance, \
    PastMassBalance
from oggm.shop import bedtopo

from combine1d.sandbox.glaciers_for_idealized_experiments import experiment_glaciers

# Module logger
log = logging.getLogger(__name__)


def create_idealized_experiments(all_glaciers,
                                 prepro_border=None,
                                 from_prepro_level=None,
                                 base_url=None,
                                 yr_spinup=1980,
                                 yr_start_run=2000,
                                 yr_end_run=2019,
                                 used_mb_models='constant'):
    '''
    Create idealized experiment (with geometry and measurements) and return
    glacier directory from which combine run can be started and true glacier
    geometry for validation of combine run
    TODO
    Parameters
    ----------

    Returns
    -------

    '''

    cfg.PARAMS['min_ice_thick_for_length'] = 0.01
    cfg.PARAMS['glacier_length_method'] = 'consecutive'

    # check if glaciers are already defined
    for glacier in all_glaciers:
        if glacier not in experiment_glaciers.keys():
            raise ValueError(f'{glacier} not defined for idealized experiments')

    # create a dict which translate rgi_ids to glacier name
    rgi_id_to_name = dict([(experiment_glaciers[glacier]['rgi_id'],
                            glacier) for glacier in all_glaciers])

    # initialise glacier directorys from OGGM prepo level
    gdirs = workflow.init_glacier_directories(list(rgi_id_to_name.keys()),
                                              from_prepro_level=from_prepro_level,
                                              prepro_base_url=base_url,
                                              prepro_border=prepro_border)

    # add names of glaciers
    for gdir in gdirs:
        gdir.name = rgi_id_to_name[gdir.rgi_id]

    # add thickness from consensus thickness estimate
    workflow.execute_entity_task(bedtopo.add_consensus_thickness, gdirs)

    # bin the consensus thickness to elevation bands
    workflow.execute_entity_task(tasks.elevation_band_flowline, gdirs,
                                 bin_variables=['consensus_ice_thickness'],
                                 preserve_totals=[True]
                                 )

    # This takes the csv file and prepares new 'inversion_flowlines.pkl' and
    # created a new csv file with regular spacing
    workflow.execute_entity_task(tasks.fixed_dx_elevation_band_flowline, gdirs,
                                 bin_variables=['consensus_ice_thickness'],
                                 preserve_totals=[True]
                                 )

    # add consensus flowline
    workflow.execute_entity_task(initialise_consensus_flowline, gdirs)

    # create spinup glacier with consensus flowline, try to match rgi_date length
    # if one t_bias_spinup_limits is None the gdirs with spinup function are
    # returned to define limits (with different sign) by hand
    all_t_bias_spinup_limits = \
        [experiment_glaciers[glacier]['t_bias_spinup_limits']
         for glacier in all_glaciers]
    if np.any([limit is None for limit in all_t_bias_spinup_limits]):
        out = []
        for gdir in gdirs:
            out.append(create_spinup_glacier(gdir,
                                             rgi_id_to_name=rgi_id_to_name,
                                             yr_start_run=yr_start_run,
                                             yr_end_run=yr_end_run,
                                             yr_spinup=yr_spinup,
                                             used_mb_models=used_mb_models))
        return out

    else:
        workflow.execute_entity_task(create_spinup_glacier, gdirs,
                                     rgi_id_to_name=rgi_id_to_name,
                                     yr_start_run=yr_start_run,
                                     yr_end_run=yr_end_run,
                                     yr_spinup=yr_spinup,
                                     used_mb_models=used_mb_models)


    # create glacier with experiment measurements
    workflow.execute_entity_task(evolve_glacier_and_create_measurements, gdirs,
                                 used_mb_models=used_mb_models,
                                 yr_start_run=yr_start_run,
                                 yr_spinup=yr_spinup,
                                 yr_end_run=yr_end_run)

    # now OGGM inversion from idealized glacier surface for first guess
    workflow.execute_entity_task(oggm_inversion_for_first_guess, gdirs)

    print('Experiment creation finished')
    return gdirs


@entity_task(log, writes=['model_flowlines'])
def initialise_consensus_flowline(gdir):
    """TODO
    """

    # initialise the consensus flowline (with consensus thickness and update w0,
    # so that surface_h and widths_m is unchanged)
    # and save as 'model_flowlines_consensus'
    df_regular = pd.read_csv(gdir.get_filepath('elevation_band_flowline',
                                               filesuffix='_fixed_dx'),
                             index_col=0)
    ice_thick_consensus = df_regular['consensus_ice_thickness'].values

    fl_consensus = copy.deepcopy(gdir.read_pickle('model_flowlines')[0])
    widths_m_original = copy.deepcopy(fl_consensus.widths_m)
    fl_consensus.bed_h[:len(ice_thick_consensus)] = \
        fl_consensus.surface_h[:len(ice_thick_consensus)] - ice_thick_consensus
    fl_consensus.thick[:len(ice_thick_consensus)] = ice_thick_consensus
    fl_consensus._w0_m[:len(ice_thick_consensus)] = \
        np.clip(widths_m_original[:len(ice_thick_consensus)] -
                fl_consensus._lambdas[:len(ice_thick_consensus)] *
                ice_thick_consensus,
                10, None)

    gdir.write_pickle([fl_consensus], 'model_flowlines', filesuffix='_consensus')


@entity_task(log, writes=['model_flowlines'])
def create_spinup_glacier(gdir, rgi_id_to_name, yr_start_run, yr_end_run,
                          yr_spinup, used_mb_models):
    """TODO
    """
    # now creat spinup glacier with consensus flowline starting from ice free,
    # try to match length at rgi_date for 'realistic' experiment setting
    fl_consensus = gdir.read_pickle('model_flowlines',
                                    filesuffix='_consensus')[0]
    length_m_ref = fl_consensus.length_m

    fls_spinup = copy.deepcopy([fl_consensus])
    fls_spinup[0].thick = np.zeros(len(fls_spinup[0].thick))
    halfsize = (yr_start_run - yr_spinup) / 2
    yr_rgi = gdir.rgi_date

    mb_spinup = MultipleFlowlineMassBalance(gdir,
                                            fls=fls_spinup,
                                            mb_model_class=ConstantMassBalance,
                                            filename='climate_historical',
                                            input_filesuffix='',
                                            y0=yr_spinup + halfsize,
                                            halfsize=halfsize)

    mb_historical = MultipleFlowlineMassBalance(gdir,
                                                fls=fls_spinup,
                                                mb_model_class=PastMassBalance,
                                                filename='climate_historical',
                                                input_filesuffix='')

    def spinup_run(t_bias):
        # with t_bias the glacier state after spinup is changed between iterations
        mb_spinup.temp_bias = t_bias
        # run the spinup
        model_spinup = FluxBasedModel(copy.deepcopy(fls_spinup),
                                      mb_spinup,
                                      y0=0)
        model_spinup.run_until_equilibrium(max_ite=1000)

        # Now conduct the actual model run to the rgi date
        model_historical = FluxBasedModel(model_spinup.fls,
                                          mb_historical,
                                          y0=yr_spinup)
        model_historical.run_until(yr_rgi)

        cost = (model_historical.length_m - length_m_ref) / length_m_ref * 100

        return cost

    glacier_name = rgi_id_to_name[gdir.rgi_id]
    if experiment_glaciers[glacier_name]['t_bias_spinup_limits'] is None:
        print('returned spinup_run function for searching for the t_bias_limits!')
        return gdir, spinup_run
    else:
        t_bias_spinup_limits = \
            experiment_glaciers[glacier_name]['t_bias_spinup_limits']

    if experiment_glaciers[glacier_name]['t_bias_spinup'] is None:
        # search for it by giving back
        t_bias_spinup = brentq(spinup_run,
                               t_bias_spinup_limits[0],
                               t_bias_spinup_limits[1],
                               maxiter=20, disp=False)
    else:
        t_bias_spinup = experiment_glaciers[glacier_name]['t_bias_spinup']

    print(f'{gdir.name} spinup tbias: {t_bias_spinup} with L mismatch at rgi_date:'
          f' {spinup_run(t_bias_spinup)} m')

    mb_spinup.temp_bias = t_bias_spinup
    model_spinup = FluxBasedModel(copy.deepcopy(fls_spinup),
                                  mb_spinup,
                                  y0=0)
    model_spinup.run_until_equilibrium(max_ite=1000)

    gdir.write_pickle(model_spinup.fls, 'model_flowlines', filesuffix='_spinup')


@entity_task(log, writes=['model_flowlines', 'inversion_input'])
def evolve_glacier_and_create_measurements(gdir, used_mb_models, yr_start_run,
                                           yr_spinup, yr_end_run):
    """TODO
    """
    fls_spinup = gdir.read_pickle('model_flowlines', filesuffix='_spinup')
    yr_rgi = gdir.rgi_date
    # now start actual experiment run for the creation of measurements
    if used_mb_models == 'constant':
        halfsize_spinup = (yr_start_run - yr_spinup) / 2
        mb_spinup = MultipleFlowlineMassBalance(gdir,
                                                fls=fls_spinup,
                                                mb_model_class=ConstantMassBalance,
                                                filename='climate_historical',
                                                input_filesuffix='',
                                                y0=yr_spinup + halfsize_spinup,
                                                halfsize=halfsize_spinup)
        halfsize_run = (yr_end_run - yr_start_run) / 2
        mb_run = MultipleFlowlineMassBalance(gdir,
                                             fls=fls_spinup,
                                             mb_model_class=ConstantMassBalance,
                                             filename='climate_historical',
                                             input_filesuffix='',
                                             y0=yr_start_run + halfsize_run,
                                             halfsize=halfsize_run)

        # save used massbalance models for combine
        mb_models_combine = {'MB1': {'type': 'constant',
                                     'years': np.array([yr_spinup,
                                                        yr_start_run])},
                             'MB2': {'type': 'constant',
                                     'years': np.array([yr_start_run,
                                                        yr_end_run])}
                             }
        gdir.write_pickle(mb_models_combine, 'inversion_input',
                          filesuffix='_combine_mb_models')

    else:
        raise NotImplementedError(f'{used_mb_models}')

    # do spinup period before first measurement
    model = FluxBasedModel(copy.deepcopy(fls_spinup),
                           mb_spinup,
                           y0=yr_spinup)
    model.run_until_and_store(yr_start_run,
                              diag_path=gdir.get_filepath('model_diagnostics',
                                                          filesuffix='_combine_spinup'))

    # get measurements for dhdt
    dh_volume = [None, None]
    dh_area = [None, None]
    dh_volume[0] = model.volume_m3
    dh_area[0] = model.area_m2

    # switch to mb_run and run to rgi_date and save measurements and flowline
    model = FluxBasedModel(copy.deepcopy(model.fls),
                           mb_run,
                           y0=yr_start_run)
    model.run_until_and_store(yr_rgi,
                              diag_path=gdir.get_filepath('model_diagnostics',
                                                          filesuffix='_combine_true_init')
                              )
    gdir.write_pickle(model.fls, 'model_flowlines',
                      filesuffix='_combine_true_init')
    rgi_date_area_km2 = model.area_km2
    rgi_date_volume_km3 = model.volume_km3
    rgi_date_us_myr = model.u_stag[0] * model._surf_vel_fac * SEC_IN_YEAR

    # now run to the end for dhdt
    model.run_until_and_store(yr_end_run,
                              diag_path=gdir.get_filepath('model_diagnostics',
                                                          filesuffix='_combine_end'),
                              geom_path=gdir.get_filepath('model_geometry',
                                                          filesuffix='_combine_end',
                                                          delete=True)
                              )
    gdir.write_pickle(model.fls, 'model_flowlines',
                      filesuffix='_combine_true_end')
    dh_volume[1] = model.volume_m3
    dh_area[1] = model.area_m2

    # calculate dh
    dh_m = (dh_volume[1] - dh_volume[0]) / \
           ((dh_area[1] + dh_area[0]) / 2.)

    # save measurements in gdir
    all_measurements = {'dh:m': {'2000-2019': dh_m},
                        'area:km2': {str(yr_rgi): rgi_date_area_km2},
                        'volume:km3': {str(yr_rgi): rgi_date_volume_km3},
                        'us:myr-1': {str(yr_rgi): rgi_date_us_myr},
                        }
    gdir.write_pickle(all_measurements, 'inversion_input',
                      filesuffix='_combine_measurements')


@entity_task(log, writes=['model_flowlines'])
def oggm_inversion_for_first_guess(gdir):
    """TODO
    """
    # now use _combine_true_init for an OGGM inversion for the first guess
    fls_inversion = gdir.read_pickle('inversion_flowlines')[0]
    fls_experiment = gdir.read_pickle('model_flowlines',
                                      filesuffix='_combine_true_init')[0]

    # get a ice mask for only the uppermost coherent ice mass
    ice_mask = np.where(fls_experiment.thick > 0.01, True, False)
    ice_mask_index_diff = np.diff(np.where(fls_experiment.thick > 0.01)[0])
    if not np.all(ice_mask_index_diff == 1):
        ice_mask[np.argmax(ice_mask_index_diff > 1) + 1:] = False

    # now use ice mask to create new inversion_flowlines
    fls_inversion.nx = int(sum(ice_mask))
    fls_inversion.surface_h = fls_experiment.surface_h[ice_mask]
    fls_inversion.widths = fls_experiment.widths[ice_mask]
    fls_inversion.is_trapezoid = np.full(fls_inversion.nx, True)
    fls_inversion.is_rectangular = np.full(fls_inversion.nx, False)
    fls_inversion.flux = None
    fls_inversion.flux_needs_correction = False
    fls_inversion.apparent_mb = None
    gdir.write_pickle([fls_inversion], 'inversion_flowlines')

    # now calculate flux for inversion
    # depending on used base url
    # no match
    # Climate period
    df = gdir.read_json('local_mustar')
    t_star = df['t_star']
    mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
    yr_range = [t_star - mu_hp, t_star + mu_hp]

    workflow.execute_entity_task(tasks.apparent_mb_from_any_mb, gdir,
                                 mb_years=yr_range)

    # e.g. for per glacier, UNTESTED
    # per_glacier_tasks = [tasks.mu_star_calibration_from_geodetic_mb,
    #                      tasks.apparent_mb_from_any_mb]

    # now do the inversion
    inversion_tasks = [tasks.prepare_for_inversion,
                       tasks.mass_conservation_inversion,
                       tasks.filter_inversion_output]
    for task in inversion_tasks:
        workflow.execute_entity_task(task, gdir)

    # Here initilise the new flowline after inversion (combine first guess)
    inv = gdir.read_pickle('inversion_output')[0]
    bed_h_inv = inv['hgt'] - inv['thick']
    bed_h_new = copy.deepcopy(fls_experiment.bed_h)
    bed_h_new[ice_mask] = bed_h_inv
    section_inv = inv['volume'] / (fls_experiment.dx * fls_experiment.map_dx)
    section_new = np.zeros(fls_experiment.nx)
    section_new[ice_mask] = section_inv
    surface_h_new = fls_experiment.surface_h
    surface_h_new[~ice_mask] = bed_h_new[~ice_mask]
    widths_m_new = fls_experiment.widths_m
    widths_m_new[~ice_mask & ~fls_experiment.is_trapezoid] = 0.

    fls_first_guess = MixedBedFlowline(line=fls_experiment.line,
                                       dx=fls_experiment.dx,
                                       map_dx=fls_experiment.map_dx,
                                       surface_h=surface_h_new,
                                       bed_h=bed_h_new,
                                       section=section_new,
                                       bed_shape=fls_experiment.bed_shape,
                                       is_trapezoid=fls_experiment.is_trapezoid,
                                       lambdas=fls_experiment._lambdas,
                                       widths_m=widths_m_new,
                                       rgi_id=fls_experiment.rgi_id,
                                       gdir=gdir)
    gdir.write_pickle([fls_first_guess], 'model_flowlines',
                      filesuffix='_combine_first_guess')
