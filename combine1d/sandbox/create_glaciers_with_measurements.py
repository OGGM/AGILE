import copy
import logging
import pandas as pd
import numpy as np
from oggm.cfg import SEC_IN_YEAR
from scipy.optimize import brentq
from itertools import tee

from oggm import cfg, workflow, tasks, entity_task, utils
from oggm.core.flowline import MixedBedFlowline, SemiImplicitModel
from oggm.core.massbalance import ConstantMassBalance, MultipleFlowlineMassBalance, \
    PastMassBalance, RandomMassBalance, MassBalanceModel
from oggm.shop import bedtopo, gcm_climate

from combine1d.sandbox.calculate_statistics import calculate_default_oggm_statistics
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
                                 used_mb_models='constant',
                                 gcm='BCC-CSM2-MR',
                                 ssp='ssp370'):
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

    # climate
    workflow.execute_entity_task(tasks.process_climate_data, gdirs)
    if cfg.PARAMS['climate_qc_months'] > 0:
        workflow.execute_entity_task(tasks.historical_climate_qc, gdirs)
    utils.get_geodetic_mb_dataframe()  # Small optim to avoid concurrency
    workflow.execute_entity_task(tasks.mu_star_calibration_from_geodetic_mb, gdirs)
    workflow.execute_entity_task(tasks.apparent_mb_from_any_mb, gdirs)

    # inversion, for each glacier separated to always use the same total volume
    for gdir in gdirs:
        workflow.calibrate_inversion_from_consensus([gdir],
                                                    apply_fs_on_mismatch=False,
                                                    error_on_mismatch=True,
                                                    filter_inversion_output=False)

    # initialise original flowline
    cfg.PARAMS['downstream_line_shape'] = 'parabola'
    workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)

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

    # initialise trapezoidal flowline
    workflow.execute_entity_task(tasks.compute_downstream_bedshape, gdirs)
    cfg.PARAMS['downstream_line_shape'] = 'trapezoidal'
    workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs,
                                 filesuffix='_trapezoidal')
    assert np.all(gdir.read_pickle('model_flowlines',
                                   filesuffix='_trapezoidal')[0].is_trapezoid)

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

    # finally use oggm default initialisation for comparisons
    workflow.execute_entity_task(oggm_default_initialisation, gdirs,
                                 ys=yr_spinup, ye=yr_end_run, gcm=gcm, ssp=ssp)

    # change back to default
    cfg.PARAMS['cfl_number'] = 0.02

    print('Experiment creation finished')
    return gdirs


@entity_task(log, writes=['model_flowlines'])
def initialise_consensus_flowline(gdir):
    """ Initialise the consensus flowline. Update w0 so that surface_h and
    widhts_m are unchanged. Saved as 'model_flowlines_consensus'.
    """

    df_regular = pd.read_csv(gdir.get_filepath('elevation_band_flowline',
                                               filesuffix='_fixed_dx'),
                             index_col=0)
    ice_thick_consensus = df_regular['consensus_ice_thickness'].values

    fl_consensus = copy.deepcopy(
        gdir.read_pickle('model_flowlines', filesuffix='_trapezoidal')[0])
    widths_m_original = copy.deepcopy(fl_consensus.widths_m)
    fl_consensus.bed_h[:len(ice_thick_consensus)] = \
        fl_consensus.surface_h[:len(ice_thick_consensus)] - ice_thick_consensus
    fl_consensus.thick[:len(ice_thick_consensus)] = ice_thick_consensus
    fl_consensus._w0_m[:len(ice_thick_consensus)] = \
        np.clip(widths_m_original[:len(ice_thick_consensus)] -
                fl_consensus._lambdas[:len(ice_thick_consensus)] *
                ice_thick_consensus,
                10, None)

    gdir.write_pickle([fl_consensus], 'model_flowlines',
                      filesuffix='_consensus')


@entity_task(log, writes=['model_flowlines'])
def create_spinup_glacier(gdir, rgi_id_to_name, yr_start_run, yr_end_run,
                          yr_spinup, used_mb_models):
    """
    Finds the tbias used for the spinup mb if not given in experiment glaciers,
    also saves the glacier state after spinup at yr_start_run (1980) as
    'model_flowlines_spinup'. Method inspired from model creation of
    Eis et al. 2019/2021
    """
    # define how many years the random climate spinup should be
    years_to_run_random = 50

    # now creat spinup glacier with consensus flowline starting from ice free,
    # try to match area at rgi_date for 'realistic' experiment setting
    fl_consensus = gdir.read_pickle('model_flowlines',
                                    filesuffix='_consensus')[0]
    length_m_ref = fl_consensus.length_m

    yr_rgi = gdir.rgi_date

    fls_spinup = copy.deepcopy([fl_consensus])

    mb_random = MultipleFlowlineMassBalance(gdir,
                                            fls=fls_spinup,
                                            mb_model_class=RandomMassBalance,
                                            seed=1,
                                            filename='climate_historical',
                                            y0=1930,
                                            halfsize=20)

    mb_past = MultipleFlowlineMassBalance(gdir,
                                            fls=fls_spinup,
                                            mb_model_class=PastMassBalance,
                                            filename='climate_historical')

    halfsize_const_1 = (yr_start_run - yr_spinup) / 2
    mb_const_1 = MultipleFlowlineMassBalance(gdir,
                                                  fls=fls_spinup,
                                                  mb_model_class=ConstantMassBalance,
                                                  filename='climate_historical',
                                                  input_filesuffix='',
                                                  y0=yr_spinup + halfsize_const_1,
                                                  halfsize=halfsize_const_1
                                                  )

    halfsize_const_2 = (yr_end_run - yr_start_run) / 2
    mb_const_2 = MultipleFlowlineMassBalance(gdir,
                                             fls=fls_spinup,
                                             mb_model_class=ConstantMassBalance,
                                             filename='climate_historical',
                                             input_filesuffix='',
                                             y0=yr_start_run + halfsize_const_2,
                                             halfsize=halfsize_const_2
                                             )

    def get_spinup_glacier(t_bias):
        # first run random climate with t_bias
        mb_random.temp_bias = t_bias
        model = SemiImplicitModel(copy.deepcopy(fls_spinup),
                                  mb_random,
                                  y0=0)
        model.run_until(years_to_run_random)

        # from their use a climate run
        model = SemiImplicitModel(copy.deepcopy(model.fls),
                                  mb_past,
                                  y0=1930)
        model.run_until(yr_spinup)

        return model

    def spinup_run(t_bias):

        model_spinup = get_spinup_glacier(t_bias)

        # Now conduct the actual model run to the rgi date
        model_historical_1 = SemiImplicitModel(model_spinup.fls,
                                               mb_const_1,
                                               y0=yr_spinup)
        model_historical_1.run_until(yr_start_run)

        model_historical_2 = SemiImplicitModel(model_historical_1.fls,
                                               mb_const_2,
                                               y0=yr_start_run)
        model_historical_2.run_until(yr_rgi)

        cost = (model_historical_2.length_m - length_m_ref) / length_m_ref * 100

        return cost

    glacier_name = rgi_id_to_name[gdir.rgi_id]
    if experiment_glaciers[glacier_name]['t_bias_spinup_limits'] is None:
        print('returned spinup_run function for searching for the t_bias_limits!')
        return gdir, spinup_run
    else:
        t_bias_spinup_limits = \
            experiment_glaciers[glacier_name]['t_bias_spinup_limits']

    if experiment_glaciers[glacier_name]['t_bias_spinup'] is None:
        # search for it
        t_bias_spinup = brentq(spinup_run,
                               t_bias_spinup_limits[0],
                               t_bias_spinup_limits[1],
                               maxiter=20, disp=False)
    else:
        t_bias_spinup = experiment_glaciers[glacier_name]['t_bias_spinup']

    print(f'{gdir.name} spinup tbias: {t_bias_spinup} with L mismatch at rgi_date:'
          f' {spinup_run(t_bias_spinup) / 100 * length_m_ref} m')

    model_spinup = get_spinup_glacier(t_bias_spinup)

    gdir.write_pickle(model_spinup.fls, 'model_flowlines', filesuffix='_spinup')


@entity_task(log, writes=['model_flowlines', 'inversion_input'])
def evolve_glacier_and_create_measurements(gdir, used_mb_models, yr_start_run,
                                           yr_spinup, yr_end_run,
                                           gcm='BCC-CSM2-MR', ssp='ssp370'):
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
    model = SemiImplicitModel(copy.deepcopy(fls_spinup),
                              mb_spinup,
                              y0=yr_spinup)
    model.run_until_and_store(
        yr_start_run,
        diag_path=gdir.get_filepath('model_diagnostics',
                                    filesuffix='_combine_spinup',
                                    delete=True),
        fl_diag_path=gdir.get_filepath('fl_diagnostics',
                                       filesuffix='_combine_spinup',
                                       delete=True)
    )

    # get measurements for dmdtda
    dmdtda_volume = [None, None]
    dmdtda_area = [None, None]
    dmdtda_volume[0] = model.volume_m3
    dmdtda_area[0] = model.area_m2

    # switch to mb_run and run to rgi_date and save measurements and flowline
    model = SemiImplicitModel(copy.deepcopy(model.fls),
                              mb_run,
                              y0=yr_start_run)
    model.run_until_and_store(
        yr_rgi,
        diag_path=gdir.get_filepath('model_diagnostics',
                                    filesuffix='_combine_true_init',
                                    delete=True),
        fl_diag_path=gdir.get_filepath('fl_diagnostics',
                                       filesuffix='_combine_true_init',
                                       delete=True)
    )
    gdir.write_pickle(model.fls, 'model_flowlines',
                      filesuffix='_combine_true_init')
    rgi_date_area_km2 = model.area_km2
    rgi_date_volume_km3 = model.volume_km3
    rgi_date_fl_surface_h = model.fls[0].surface_h
    rgi_date_fl_widths = model.fls[0].widths_m

    # need to convert stag velocity to nonstaggerd grid points
    var = model.u_stag[0]
    val = (var[1:model.fls[0].nx + 1] + var[:model.fls[0].nx]) / 2 * \
          model._surf_vel_fac
    rgi_date_us_myr = val * cfg.SEC_IN_YEAR

    # now run to the end for dmdtda
    model.run_until_and_store(
        yr_end_run,
        diag_path=gdir.get_filepath('model_diagnostics',
                                    filesuffix='_combine_end',
                                    delete=True),
        geom_path=gdir.get_filepath('model_geometry',
                                    filesuffix='_combine_end',
                                    delete=True),
        fl_diag_path=gdir.get_filepath('fl_diagnostics',
                                       filesuffix='_combine_end',
                                       delete=True)
                              )
    gdir.write_pickle(model.fls, 'model_flowlines',
                      filesuffix='_combine_true_end')
    dmdtda_volume[1] = model.volume_m3
    dmdtda_area[1] = model.area_m2
    # combine model diagnostics to one file for the whole period
    tasks.merge_consecutive_run_outputs(
        gdir,
        input_filesuffix_1='_combine_spinup',
        input_filesuffix_2='_combine_true_init',
        output_filesuffix='_combine_total_run',
        delete_input=False)
    tasks.merge_consecutive_run_outputs(
        gdir,
        input_filesuffix_1='_combine_total_run',
        input_filesuffix_2='_combine_end',
        output_filesuffix='_combine_total_run',
        delete_input=False)

    # calculate dmdtda
    dmdtda = (
        # mass change
        (dmdtda_volume[1] - dmdtda_volume[0]) *
        cfg.PARAMS['ice_density']
        # divided by mean area
        / ((dmdtda_area[1] + dmdtda_area[0]) / 2.)
        # divided by period
        / (yr_end_run - yr_start_run)
    )

    # save measurements in gdir
    all_measurements = {'dmdtda:kg m-2 yr-1': {f'{yr_start_run}-{yr_end_run}':
                                               dmdtda},
                        'area:km2': {str(yr_rgi): rgi_date_area_km2},
                        'area:m2': {str(yr_rgi): rgi_date_area_km2 * 1e-6},
                        'volume:km3': {str(yr_rgi): rgi_date_volume_km3},
                        'volume:m3': {str(yr_rgi): rgi_date_volume_km3 * 1e-9},
                        'us:myr-1': {str(yr_rgi): rgi_date_us_myr},
                        'fl_surface_h:m': {str(yr_rgi): rgi_date_fl_surface_h},
                        'fl_widths:m': {str(yr_rgi): rgi_date_fl_widths},
                        'fl_total_area:m2': {str(yr_rgi):
                                             rgi_date_area_km2 * 1e-6},
                        'fl_total_area:km2': {str(yr_rgi): rgi_date_area_km2},
                        }
    gdir.write_pickle(all_measurements, 'inversion_input',
                      filesuffix='_combine_measurements')

    # conduct one future simulation for comparison
    # download locations for precipitation and temperature
    bp = 'https://cluster.klima.uni-bremen.de/~oggm/cmip6/GCM/{}/{}_{}_r1i1p1f1_pr.nc'
    bt = 'https://cluster.klima.uni-bremen.de/~oggm/cmip6/GCM/{}/{}_{}_r1i1p1f1_tas.nc'
    # 'Download' the files
    ft = utils.file_downloader(bt.format(gcm, gcm, ssp))
    fp = utils.file_downloader(bp.format(gcm, gcm, ssp))
    # bias correct them
    workflow.execute_entity_task(gcm_climate.process_cmip_data, [gdir],
                                 filesuffix='_{}_{}'.format(gcm, ssp),
                                 # recognize the climate file for later
                                 fpath_temp=ft,  # temperature projections
                                 fpath_precip=fp,  # precip projections
                                 )
    # actual projection runs
    rid = '_{}_{}'.format(gcm, ssp)
    workflow.execute_entity_task(tasks.run_from_climate_data, [gdir],
                                 climate_filename='gcm_data',
                                 climate_input_filesuffix=rid,
                                 init_model_fls=model.fls,
                                 ys=yr_end_run,
                                 output_filesuffix='_combine_true_future',
                                 evolution_model=SemiImplicitModel,
                                 )


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

    workflow.execute_entity_task(tasks.apparent_mb_from_any_mb, gdir)

    # now do the inversion
    inversion_tasks = [tasks.prepare_for_inversion,
                       tasks.mass_conservation_inversion]
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
    # force trapezoidal bed shape everywhere, could cause a problem with rect
    # shapes, which are used for too thick glacier parts during the inversion
    # where lambda is not steep enough to have w0 > 0
    new_lambdas = np.zeros(fls_experiment.nx) + fls_experiment._lambdas[0]

    fls_first_guess = MixedBedFlowline(line=fls_experiment.line,
                                       dx=fls_experiment.dx,
                                       map_dx=fls_experiment.map_dx,
                                       surface_h=surface_h_new,
                                       bed_h=bed_h_new,
                                       section=section_new,
                                       bed_shape=fls_experiment.bed_shape,
                                       is_trapezoid=fls_experiment.is_trapezoid,
                                       lambdas=new_lambdas,
                                       widths_m=widths_m_new,
                                       rgi_id=fls_experiment.rgi_id,
                                       gdir=gdir)
    gdir.write_pickle([fls_first_guess], 'model_flowlines',
                      filesuffix='_combine_first_guess')


def pairwise(iterable):
    """ In Python 3.10 this is available in itertools.pairwise"""
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class StackedMassBalance(MassBalanceModel):
    """Define different MassBalanceModels for different time periods.
    """

    def __init__(self, gdir, mb_model_settings, filename='climate_historical',
                 input_filesuffix=''):

        super(StackedMassBalance, self).__init__()

        self.hemisphere = gdir.hemisphere

        # set periods of different mb_models
        all_periods = []
        for key_mb in mb_model_settings:
            all_periods.append(mb_model_settings[key_mb]['years'])
        self._periods = np.sort(np.unique(np.concatenate(all_periods)))

        # set mb_model for all specific sorted periods
        mb_models = {}
        for p_nr, (p_start, p_end) in enumerate(pairwise(self._periods)):
            # check if their is a mb_model defined for the current period
            found_mb_model = False
            for key_mb in mb_model_settings:
                if np.all([p_start, p_end] ==
                          mb_model_settings[key_mb]['years']):
                    if mb_model_settings[key_mb]['type'] == 'constant':
                        halfsize_run = (p_end - p_start) / 2
                        mb_models[p_nr] = ConstantMassBalance(
                            gdir, y0=p_start + halfsize_run,
                            halfsize=halfsize_run, filename=filename,
                            input_filesuffix=input_filesuffix)
                        found_mb_model = True
                        break
                    else:
                        raise NotImplementedError('')

            if not found_mb_model:
                raise ValueError(f'No mb model defined for period {p_start} '
                                 f'to {p_end}!')

        self._mb_models = mb_models

    def get_period_nr(self, year):
        current_period = np.searchsorted(self._periods, year, side='right')

        # the given year is smaller than minimum defined in periods
        if current_period == 0:
            raise ValueError(f'No mb model defined for year {year}')

        if current_period >= len(self._periods):
            if year == self._periods[-1]:
                # ok at the upper limit we use again the last mb_model
                current_period -= 1
            else:
                # the given year is larger than the maximum defined in periods
                raise ValueError(f'No mb model defined for year {year}')

        return current_period - 1  # because mb_model index starts with 0

    def get_monthly_mb(self, heights, year=None, fl_id=None, fls=None):
        return self._mb_models[self.get_period_nr(year)].get_monthly_mb(
            heights=heights, year=year, fl_id=fl_id, fls=fls)

    def get_annual_mb(self, heights, year=None, fl_id=None, fls=None):
        return self._mb_models[self.get_period_nr(year)].get_annual_mb(
            heights=heights, year=year, fl_id=fl_id, fls=fls)


def get_experiment_mb_model(gdir):
    """Getting the experiment mb_model defined earlier """
    mb_model_settings = gdir.read_pickle('inversion_input',
                                         filesuffix='_combine_mb_models')

    return StackedMassBalance(gdir=gdir, mb_model_settings=mb_model_settings)


def oggm_default_initialisation(gdir, ys, ye, gcm='BCC-CSM2-MR', ssp='ssp370'):
    """ Use oggm default initialisation method and do a projection
    """
    mb_model = get_experiment_mb_model(gdir)

    # do dynamic initialisation
    dynamic_spinup_model = tasks.run_dynamic_spinup(
        gdir, spinup_start_yr=ys, ye=ye, evolution_model=SemiImplicitModel,
        model_flowline_filesuffix='_combine_first_guess',
        mb_model_historical=mb_model, precision_absolute=0.1,
        output_filesuffix='_oggm_default_past', store_model_geometry=True,
        store_fl_diagnostics=True, store_model_evolution=True,
        ignore_errors=False, add_fixed_geometry_spinup=True)

    # conduct projection run
    rid = '_{}_{}'.format(gcm, ssp)
    workflow.execute_entity_task(tasks.run_from_climate_data, [gdir],
                                 climate_filename='gcm_data',
                                 climate_input_filesuffix=rid,
                                 init_model_fls=dynamic_spinup_model.fls,
                                 ys=ye,
                                 output_filesuffix='_oggm_default_future',
                                 evolution_model=SemiImplicitModel,
                                 )

    # calculate statistics
    calculate_default_oggm_statistics(gdir)
