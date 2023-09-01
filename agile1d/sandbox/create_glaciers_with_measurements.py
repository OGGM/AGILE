import copy
import logging
import pandas as pd
import numpy as np
from oggm.cfg import SEC_IN_YEAR
from scipy.optimize import brentq

from oggm import cfg, workflow, tasks, entity_task, utils
from oggm.core.flowline import MixedBedFlowline, SemiImplicitModel
from oggm.core.massbalance import ConstantMassBalance, MultipleFlowlineMassBalance, \
    MonthlyTIModel, RandomMassBalance, MassBalanceModel
from oggm.shop import bedtopo, gcm_climate

from agile1d.core.massbalance import StackedMassBalance
from agile1d.core.inversion import get_adaptive_upper_ice_thickness_limit
from agile1d.sandbox.calculate_statistics import calculate_default_oggm_statistics
from agile1d.sandbox.glaciers_for_idealized_experiments import experiment_glaciers

# Module logger
log = logging.getLogger(__name__)


def create_idealized_experiments(all_glaciers,
                                 glacier_states,
                                 prepro_border=None,
                                 from_prepro_level=None,
                                 base_url=None,
                                 yr_run_start=1980,
                                 yr_dmdt_start=2000,
                                 yr_run_end=2020,
                                 used_mb_models='TIModel',
                                 gcm='BCC-CSM2-MR',
                                 ssp='ssp370'):
    '''
    Create idealized experiment (with geometry and measurements) and return
    glacier directory from which agile run can be started and true glacier
    geometry for validation of agile run
    TODO
    Parameters
    ----------

    Returns
    -------

    '''

    # store model geometry for hydro output
    cfg.PARAMS['store_model_geometry'] = True

    # check if glaciers are already defined
    for glacier in all_glaciers:
        if glacier not in experiment_glaciers.keys():
            raise ValueError(f'{glacier} not defined for idealized experiments')

    # check if glacier states are valid
    if any(state not in ['equilibrium', 'retreating', 'advancing']
           for state in glacier_states):
        raise NotImplementedError("Glacier states could only be one of "
                                  "['equilibrium', 'retreating', 'advancing']"
                                  ", but given was "
                                  f"{glacier_states}.")

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
    utils.get_geodetic_mb_dataframe()  # Small optim to avoid concurrency
    workflow.execute_entity_task(tasks.mb_calibration_from_geodetic_mb, gdirs,
                                 informed_threestep=True)
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
    for gdir in gdirs:
        assert np.all(
            gdir.read_pickle('model_flowlines',
                             filesuffix='_trapezoidal')[0].is_trapezoid)

    # add consensus flowline
    workflow.execute_entity_task(initialise_consensus_flowline, gdirs)

    # change rgi_date for all to 2000
    for gdir in gdirs:
        gdir.rgi_date = 2000

    # create glacier with experiment measurements, for all states
    for glacier_state in glacier_states:
        # create spinup glacier starting from consensus flowline
        workflow.execute_entity_task(create_spinup_glacier, gdirs,
                                     glacier_state=glacier_state,
                                     yr_run_start=yr_run_start)

        workflow.execute_entity_task(evolve_glacier_and_create_measurements,
                                     gdirs,
                                     glacier_state=glacier_state,
                                     used_mb_models=used_mb_models,
                                     yr_dmdt_start=yr_dmdt_start,
                                     yr_run_start=yr_run_start,
                                     yr_run_end=yr_run_end,
                                     gcm=gcm,
                                     ssp=ssp)

        # now OGGM inversion from idealized glacier surface for first guess
        workflow.execute_entity_task(oggm_inversion_for_first_guess,
                                     gdirs, glacier_state=glacier_state)

        # now add a second bed inversion
        workflow.execute_entity_task(glabtop_inversion_for_first_guess,
                                     gdirs, glacier_state=glacier_state)

        # finally use oggm default initialisation for comparisons
        workflow.execute_entity_task(oggm_default_initialisation,
                                     gdirs, glacier_state=glacier_state,
                                     ys=yr_run_start, ye=yr_run_end, gcm=gcm,
                                     ssp=ssp)

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
    ice_free_start = len(df_regular['consensus_ice_thickness'].values)

    fl_ref = copy.deepcopy(
        gdir.read_pickle('model_flowlines', filesuffix='_trapezoidal')[0])
    assert ice_free_start == sum(fl_ref.thick > 0)
    widths_m_original = copy.deepcopy(fl_ref.widths_m)
    surface_h_original = copy.deepcopy(fl_ref.surface_h)
    bed_h_consensus = copy.deepcopy(fl_ref.bed_h)
    ice_thick_consensus = copy.deepcopy(fl_ref.thick)
    ice_thick_consensus[:ice_free_start] = df_regular['consensus_ice_thickness'].values

    # calculate bed height from surface height and thickness
    bed_h_consensus[:ice_free_start] = (surface_h_original[:ice_free_start] -
                                        ice_thick_consensus[:ice_free_start])

    # smooth bed_h_consensus taking downstream line into account
    n_smoothing = 5  # number of grid points to smooth
    n_smoothing = -abs(n_smoothing)
    downstream_sfc_h = bed_h_consensus[ice_free_start: ice_free_start + 5]
    down_slope_avg = np.average(np.abs(np.diff(downstream_sfc_h)))
    new_last_bed_h = downstream_sfc_h[0] + down_slope_avg
    all_bed_h_changes = np.linspace(0, new_last_bed_h -
                                    bed_h_consensus[:ice_free_start][-1],
                                    -n_smoothing)
    bed_h_consensus[:ice_free_start][n_smoothing:] = \
        bed_h_consensus[:ice_free_start][n_smoothing:] + all_bed_h_changes
    ice_thick_consensus[:ice_free_start] = (
            surface_h_original[:ice_free_start] -
            bed_h_consensus[:ice_free_start])

    # adapt surface height to make sure min ice thick of 1 m
    min_ice_thick = 1
    surface_h_use = copy.deepcopy(surface_h_original)
    surface_h_use[:ice_free_start] = np.where(
        surface_h_original[:ice_free_start] -
        bed_h_consensus[:ice_free_start] < min_ice_thick,
        bed_h_consensus[:ice_free_start] + min_ice_thick,
        surface_h_original[:ice_free_start])
    ice_thick_consensus = surface_h_use - bed_h_consensus

    # make sure that everything is trapezoidal
    max_thick_for_trap = ((widths_m_original -
                           cfg.PARAMS['trapezoid_min_bottom_width']) /
                          cfg.PARAMS['trapezoid_lambdas'])
    ice_thick_consensus = np.where(ice_thick_consensus > max_thick_for_trap,
                                   max_thick_for_trap,
                                   ice_thick_consensus)
    surface_h_use = ice_thick_consensus + bed_h_consensus

    # calculate new section
    w0_m_consensus = (widths_m_original - cfg.PARAMS['trapezoid_lambdas'] *
                      ice_thick_consensus)
    section_consensus = ((widths_m_original + w0_m_consensus) / 2 *
                         ice_thick_consensus)

    assert np.all(section_consensus >= 0)

    fl_consensus = MixedBedFlowline(line=fl_ref.line,
                                    dx=fl_ref.dx,
                                    map_dx=fl_ref.map_dx,
                                    surface_h=surface_h_use,
                                    bed_h=bed_h_consensus,
                                    section=section_consensus,
                                    bed_shape=fl_ref.bed_shape,
                                    is_trapezoid=
                                    np.full_like(fl_ref.is_trapezoid, True),
                                    lambdas=
                                    np.full_like(fl_ref._lambdas,
                                                 cfg.PARAMS['trapezoid_lambdas']),
                                    widths_m=widths_m_original,
                                    rgi_id=fl_ref.rgi_id,
                                    gdir=gdir)

    assert np.allclose(w0_m_consensus, fl_consensus._w0_m)
    assert np.all(fl_consensus._w0_m >=
                  cfg.PARAMS['trapezoid_min_bottom_width'])
    assert np.all(~fl_consensus.is_rectangular)
    gdir.write_pickle([fl_consensus], 'model_flowlines',
                      filesuffix='_consensus')


def get_equilibrium_year_and_halfsize(gdir):
    return ({'RGI60-11.01450': 1993,  # Aletsch
             'RGI60-16.02444': 1982,  # Artesonraju
             'RGI60-14.06794': 1987,  # Baltoro
             'RGI60-02.05098': 1957,  # Peyto
             }[gdir.rgi_id],
            {'RGI60-11.01450': 5,  # Aletsch
             'RGI60-16.02444': 1,  # Artesonraju
             'RGI60-14.06794': 6,  # Baltoro
             'RGI60-02.05098': 9,  # Peyto
             }[gdir.rgi_id])


def get_spinup_mb_model(gdir, glacier_state, fls_spinup):
    if glacier_state in ['retreating']:
        t_bias_default = gdir.read_json('mb_calib')['temp_bias']
        if gdir.rgi_id == 'RGI60-16.02444':
            t_bias_use = t_bias_default + 0
        elif gdir.rgi_id in ['RGI60-02.05098', 'RGI60-14.06794']:
            t_bias_use = t_bias_default - 0
        elif gdir.rgi_id == 'RGI60-11.01450':
            t_bias_use = t_bias_default + 1.1
        else:
            raise NotImplementedError(f'{gdir.rgi_id}')
        mb_model = MultipleFlowlineMassBalance(gdir,
                                               fls=fls_spinup,
                                               mb_model_class=RandomMassBalance,
                                               seed=2,
                                               filename='climate_historical',
                                               y0=1950,
                                               halfsize=30,
                                               temp_bias=t_bias_use
                                               )
    elif glacier_state in ['advancing']:
        t_bias_default = gdir.read_json('mb_calib')['temp_bias']
        if gdir.rgi_id == 'RGI60-16.02444':
            t_bias_use = t_bias_default + 0.4
        elif gdir.rgi_id in ['RGI60-02.05098']:
            t_bias_use = t_bias_default - 0.5
        elif gdir.rgi_id in ['RGI60-14.06794']:
            t_bias_use = t_bias_default - 0
        elif gdir.rgi_id == 'RGI60-11.01450':
            t_bias_use = t_bias_default + 1.1
        else:
            raise NotImplementedError(f'{gdir.rgi_id}')
        mb_model = MultipleFlowlineMassBalance(gdir,
                                               fls=fls_spinup,
                                               mb_model_class=RandomMassBalance,
                                               seed=2,
                                               filename='climate_historical',
                                               y0=1950,
                                               halfsize=30,
                                               temp_bias=t_bias_use)
    elif glacier_state in ['equilibrium']:
        y0, halfsize = get_equilibrium_year_and_halfsize(gdir)
        mb_model = MultipleFlowlineMassBalance(gdir,
                                               fls=fls_spinup,
                                               mb_model_class=ConstantMassBalance,
                                               filename='climate_historical',
                                               y0=y0,
                                               halfsize=halfsize)
    else:
        raise NotImplementedError(f'{glacier_state}')

    return mb_model


@entity_task(log, writes=['model_flowlines'])
def create_spinup_glacier(gdir, glacier_state, yr_run_start):
    """
    Creates and saves the glacier state at yr_run_start (e.g. 1980) as
    'model_flowlines_creation_spinup'. Method inspired from model creation of
    Eis et al. 2019/2021.
    """
    # define how many years the random climate spinup should be
    years_to_run_random = 60
    if glacier_state == 'equilibrium':
        years_to_run_random = 120

    # use flow-parameters from default oggm inversion
    diag = gdir.get_diagnostics()
    fs = diag.get('inversion_fs', cfg.PARAMS['fs'])
    glen_a = diag.get('inversion_glen_a', cfg.PARAMS['glen_a'])
    kwarg_dyn = {'fs': fs, 'glen_a': glen_a}

    # now creat spinup glacier starting from consensus flowline
    fls_spinup = gdir.read_pickle('model_flowlines',
                                  filesuffix='_consensus')

    mb_random = get_spinup_mb_model(gdir, glacier_state, fls_spinup)

    model = SemiImplicitModel(fls_spinup,
                              mb_random,
                              y0=yr_run_start - years_to_run_random,
                              **kwarg_dyn)
    model.run_until_and_store(
        yr_run_start,
        diag_path=gdir.get_filepath('model_diagnostics',
                                    filesuffix='_agile_creation_spinup_'
                                               f'{glacier_state}',
                                    delete=True))

    print(f'{gdir.name} L mismatch at rgi_date:'
          f' {model.fls[-1].length_m - fls_spinup[-1].length_m} m')

    gdir.write_pickle(model.fls, 'model_flowlines',
                      filesuffix='_creation_spinup_'
                                 f'{glacier_state}')


def get_glacier_state_mb_model(gdir, glacier_state, save_to_gdir=False,
                               fls_spinup=None):
    if fls_spinup is None:
        fls_spinup = gdir.read_pickle('model_flowlines',
                                      filesuffix='_creation_spinup_'
                                                 f'{glacier_state}')
    model_type = None
    model_args = {}
    if glacier_state == 'equilibrium':
        y0, halfsize = get_equilibrium_year_and_halfsize(gdir)
        mb_model = MultipleFlowlineMassBalance(
            gdir,
            fls=fls_spinup,
            mb_model_class=ConstantMassBalance,
            y0=y0,
            halfsize=halfsize,
            filename='climate_historical',
            )
        model_type = 'ConstantModel'
        model_args['y0'] = y0
        model_args['halfsize'] = halfsize
        model_args['filename'] = 'climate_historical'

    elif glacier_state == 'advancing':
        t_bias_default = gdir.read_json('mb_calib')['temp_bias']
        repeat = False
        ys = None
        ye = None
        if gdir.rgi_id == 'RGI60-16.02444':
            t_bias_use = t_bias_default - 0.35
            repeat = True
            ys = 1979
            ye = 2012
        elif gdir.rgi_id in ['RGI60-02.05098']:
            t_bias_use = t_bias_default - 1.8
        elif gdir.rgi_id in ['RGI60-14.06794']:
            t_bias_use = t_bias_default - 2.5
        elif gdir.rgi_id == 'RGI60-11.01450':
            t_bias_use = t_bias_default - 2
        else:
            raise NotImplementedError(f'{gdir.rgi_id}')
        mb_model = MultipleFlowlineMassBalance(
            gdir,
            fls=fls_spinup,
            mb_model_class=MonthlyTIModel,
            temp_bias=t_bias_use,
            repeat=repeat,
            ys=ys, ye=ye,
            filename='climate_historical')

        model_type = 'TIModel'
        model_args['temp_bias'] = t_bias_use
        model_args['repeat'] = repeat
        model_args['ys'] = ys
        model_args['ye'] = ye
        model_args['filename'] = 'climate_historical'

    elif glacier_state == 'retreating':
        t_bias_default = gdir.read_json('mb_calib')['temp_bias']
        if gdir.rgi_id == 'RGI60-16.02444':
            t_bias_use = t_bias_default + 0
        elif gdir.rgi_id in ['RGI60-02.05098']:
            t_bias_use = t_bias_default + 0
        elif gdir.rgi_id in ['RGI60-14.06794']:
            t_bias_use = t_bias_default + 1.2
        elif gdir.rgi_id == 'RGI60-11.01450':
            t_bias_use = t_bias_default + 1.3
        else:
            raise NotImplementedError(f'{gdir.rgi_id}')
        mb_model = MultipleFlowlineMassBalance(
            gdir,
            fls=fls_spinup,
            mb_model_class=MonthlyTIModel,
            temp_bias=t_bias_use,
            filename='climate_historical')

        model_type = 'TIModel'
        model_args['temp_bias'] = t_bias_use
        model_args['filename'] = 'climate_historical'

    if save_to_gdir:
        mb_models_agile = {'MB': {'type': model_type,
                                  'model_args': model_args,
                                  'years': np.array([1980, 2020])}
                           }

        # write the mb_model settings for later into gdir
        gdir.write_pickle(mb_models_agile, 'inversion_input',
                          filesuffix=f'_agile_mb_models_{glacier_state}')

    return mb_model


@entity_task(log, writes=['model_flowlines', 'inversion_input'])
def evolve_glacier_and_create_measurements(gdir, glacier_state, used_mb_models,
                                           yr_dmdt_start, yr_run_start,
                                           yr_run_end,
                                           gcm='BCC-CSM2-MR', ssp='ssp370'):
    """TODO
    """
    fls_spinup = gdir.read_pickle('model_flowlines',
                                  filesuffix='_creation_spinup_'
                                             f'{glacier_state}')
    yr_rgi = gdir.rgi_date

    # use flow-parameters from default oggm inversion
    diag = gdir.get_diagnostics()
    fs = diag.get('inversion_fs', cfg.PARAMS['fs'])
    glen_a = diag.get('inversion_glen_a', cfg.PARAMS['glen_a'])
    kwarg_dyn = {'fs': fs, 'glen_a': glen_a}

    # now start actual experiment run for the creation of measurements
    if used_mb_models == 'constant':
        '''
            mb_models_agile = {'MB1': {'type': 'constant',
                                       'years': np.array([yr_run_start,
                                                          yr_dmdt_start])},
                               'MB2': {'type': 'constant',
                                       'years': np.array([yr_dmdt_start,
                                                          yr_run_end])}
                               }
            mb_run = StackedMassBalance(gdir=gdir,
                                        mb_model_settings=mb_models_agile)
        '''

        raise NotImplementedError('Not implemented for different glacier'
                                  'states!')

    elif used_mb_models == 'TIModel':
        mb_run = get_glacier_state_mb_model(gdir, glacier_state,
                                            save_to_gdir=True)
    else:
        raise NotImplementedError(f'{used_mb_models}')
    # do spinup period before first measurement
    model = SemiImplicitModel(copy.deepcopy(fls_spinup),
                              mb_run,
                              y0=yr_run_start,
                              **kwarg_dyn)
    model.run_until_and_store(
        yr_dmdt_start,
        diag_path=gdir.get_filepath('model_diagnostics',
                                    filesuffix='_agile_true_dmdt_start_'
                                               f'{glacier_state}',
                                    delete=True),
        fl_diag_path=gdir.get_filepath('fl_diagnostics',
                                       filesuffix='_agile_true_dmdt_start_'
                                                  f'{glacier_state}',
                                       delete=True)
    )

    # get measurements for dmdtda
    dmdtda_volume = [None, None]
    dmdtda_area = [None, None]
    dmdtda_volume[0] = model.volume_m3
    dmdtda_area[0] = model.area_m2

    model.run_until_and_store(
        yr_rgi,
        diag_path=gdir.get_filepath('model_diagnostics',
                                    filesuffix='_agile_true_init_'
                                               f'{glacier_state}',
                                    delete=True),
        fl_diag_path=gdir.get_filepath('fl_diagnostics',
                                       filesuffix='_agile_true_init_'
                                                  f'{glacier_state}',
                                       delete=True)
    )
    gdir.write_pickle(model.fls, 'model_flowlines',
                      filesuffix=f'_agile_true_init_{glacier_state}')
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
        yr_run_end,
        diag_path=gdir.get_filepath('model_diagnostics',
                                    filesuffix='_agile_true_end_'
                                               f'{glacier_state}',
                                    delete=True),
        geom_path=gdir.get_filepath('model_geometry',
                                    filesuffix='_agile_true_end_'
                                               f'{glacier_state}',
                                    delete=True),
        fl_diag_path=gdir.get_filepath('fl_diagnostics',
                                       filesuffix='_agile_true_end_'
                                                  f'{glacier_state}',
                                       delete=True)
                              )
    gdir.write_pickle(model.fls, 'model_flowlines',
                      filesuffix=f'_agile_true_end_{glacier_state}')
    dmdtda_volume[1] = model.volume_m3
    dmdtda_area[1] = model.area_m2

    # agile model diagnostics to one file for the whole period
    tasks.merge_consecutive_run_outputs(
        gdir,
        input_filesuffix_1=f'_agile_true_dmdt_start_{glacier_state}',
        input_filesuffix_2=f'_agile_true_init_{glacier_state}',
        output_filesuffix=f'_agile_true_total_run_{glacier_state}',
        delete_input=False)
    tasks.merge_consecutive_run_outputs(
        gdir,
        input_filesuffix_1=f'_agile_true_total_run_{glacier_state}',
        input_filesuffix_2=f'_agile_true_end_{glacier_state}',
        output_filesuffix=f'_agile_true_total_run_{glacier_state}',
        delete_input=False)

    # calculate dmdtda
    dmdtda = (
        # mass change
        (dmdtda_volume[1] - dmdtda_volume[0]) *
        cfg.PARAMS['ice_density']
        # divided by mean area
        / ((dmdtda_area[1] + dmdtda_area[0]) / 2.)
        # divided by period
        / (yr_run_end - yr_dmdt_start)
    )

    # save measurements in gdir
    all_measurements = {'dmdtda:kg m-2 yr-1': {f'{yr_dmdt_start}-{yr_run_end}':
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
                      filesuffix=f'_agile_measurements_{glacier_state}')

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
    workflow.execute_entity_task(tasks.run_with_hydro, [gdir],
                                 run_task=tasks.run_from_climate_data,
                                 climate_filename='gcm_data',
                                 climate_input_filesuffix=rid,
                                 init_model_fls=model.fls,
                                 ys=yr_run_end,
                                 output_filesuffix='_agile_true_future_'
                                                   f'{glacier_state}',
                                 evolution_model=SemiImplicitModel,
                                 store_fl_diagnostics=True
                                 )


@entity_task(log, writes=['model_flowlines'])
def oggm_inversion_for_first_guess(gdir, glacier_state):
    """TODO
    """
    # now use _agile_true_init for an OGGM inversion for the first guess
    fls_inversion = gdir.read_pickle('inversion_flowlines')[0]
    fls_experiment = gdir.read_pickle('model_flowlines',
                                      filesuffix='_agile_true_init_'
                                                 f'{glacier_state}')[0]

    # load the oggm default dynamic parameters
    diag = gdir.get_diagnostics()
    fs = diag['inversion_fs']
    glen_a = diag['inversion_glen_a']

    # get a ice mask for only the uppermost coherent ice mass
    ice_mask = np.where(fls_experiment.thick > 0.01, True, False)
    ice_mask_index_diff = np.diff(np.where(fls_experiment.thick > 0.01)[0])
    # if their is an ice-free gap raise an assertion
    assert np.all(ice_mask_index_diff == 1), (f'Their is a ice-free gap, check!'
                                              f' (rgi_id: {gdir.rgi_id}, '
                                              f'glacier_state: {glacier_state})')

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
    workflow.execute_entity_task(tasks.prepare_for_inversion, gdir)
    workflow.execute_entity_task(tasks.mass_conservation_inversion, gdir,
                                 fs=fs, glen_a=glen_a)

    # Here initilise the new flowline after inversion (agile first guess)
    inv = gdir.read_pickle('inversion_output')[0]
    # calculate max thickness and section to force trapezoidal everywhere
    max_thick_for_trap = ((inv['width'] -
                           cfg.PARAMS['trapezoid_min_bottom_width']) /
                          cfg.PARAMS['trapezoid_lambdas'])
    max_section_for_trap = (cfg.PARAMS['trapezoid_min_bottom_width'] +
                            cfg.PARAMS['trapezoid_lambdas'] *
                            max_thick_for_trap / 2) * max_thick_for_trap
    inv_thick_use = np.where(inv['is_rectangular'],
                             max_thick_for_trap,
                             inv['thick'])
    bed_h_inv = inv['hgt'] - inv_thick_use
    bed_h_new = copy.deepcopy(fls_experiment.bed_h)
    bed_h_new[ice_mask] = bed_h_inv

    section_inv = np.where(inv['is_rectangular'],
                           max_section_for_trap,
                           inv['volume'] / (fls_experiment.dx *
                                            fls_experiment.map_dx))
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

    assert np.allclose(fls_first_guess.widths_m, fls_experiment.widths_m)
    assert np.allclose(fls_first_guess.thick[ice_mask], inv_thick_use)
    gdir.write_pickle([fls_first_guess], 'model_flowlines',
                      filesuffix=f'_oggm_first_guess_{glacier_state}')


@entity_task(log, writes=['model_flowlines'])
def glabtop_inversion_for_first_guess(gdir, glacier_state):
    """TODO
    """
    fls_experiment = gdir.read_pickle('model_flowlines',
                                      filesuffix='_agile_true_init_'
                                                 f'{glacier_state}')[0]
    ice_mask = np.where(fls_experiment.thick > 0.01, True, False)

    fg_thick = get_adaptive_upper_ice_thickness_limit(
        fls_experiment, additional_ice_thickness=0)
    fg_bed_h = fls_experiment.surface_h[fls_experiment.thick > 0] - fg_thick

    bed_h_new = copy.deepcopy(fls_experiment.bed_h)
    bed_h_new[ice_mask] = fg_bed_h

    section_new = np.zeros(fls_experiment.nx)
    widths = fls_experiment.widths_m[ice_mask]
    lam = fls_experiment._lambdas[ice_mask]
    section_new[ice_mask] = (2*widths - lam * fg_thick) / 2 * fg_thick

    fl_fg = MixedBedFlowline(line=fls_experiment.line,
                             dx=fls_experiment.dx,
                             map_dx=fls_experiment.map_dx,
                             surface_h=fls_experiment.surface_h,
                             bed_h=bed_h_new,
                             section=section_new,
                             bed_shape=fls_experiment.bed_shape,
                             is_trapezoid=fls_experiment.is_trapezoid,
                             lambdas=fls_experiment._lambdas,
                             widths_m=fls_experiment.widths_m,
                             rgi_id=fls_experiment.rgi_id,
                             gdir=gdir)

    assert np.allclose(fl_fg.widths_m, fls_experiment.widths_m)
    assert np.all(fl_fg._w0_m >= 9.9)
    assert np.all(fl_fg.is_trapezoid)

    gdir.write_pickle([fl_fg], 'model_flowlines',
                      filesuffix=f'_glabtop_first_guess_{glacier_state}')


def get_experiment_mb_model(gdir):
    """Getting the experiment mb_model defined earlier """
    mb_model_settings = gdir.read_pickle('inversion_input',
                                         filesuffix='_agile_mb_models')

    return StackedMassBalance(gdir=gdir, mb_model_settings=mb_model_settings)


def oggm_default_initialisation(gdir, glacier_state, ys, ye,
                                gcm='BCC-CSM2-MR', ssp='ssp370'):
    """ Use oggm default initialisation method and do a projection
    """

    # define mb_model depening on glacier_state
    mb_model = get_glacier_state_mb_model(gdir, glacier_state)

    # do dynamic initialisation
    spinup_model = tasks.run_dynamic_spinup(
        gdir, spinup_start_yr=ys, ye=ye, evolution_model=SemiImplicitModel,
        model_flowline_filesuffix=f'_oggm_first_guess_{glacier_state}',
        #precision_absolute=0.1,
        mb_model_historical=mb_model,
        output_filesuffix=f'_oggm_dynamic_past_{glacier_state}',
        store_model_geometry=True,
        store_fl_diagnostics=True, store_model_evolution=True,
        ignore_errors=False, add_fixed_geometry_spinup=True)

    # do static initialisation
    fls_init = gdir.read_pickle('model_flowlines',
                                filesuffix='_oggm_first_guess_'
                                           f'{glacier_state}')
    static_model = tasks.run_from_climate_data(
        gdir, ye=ye,
        fixed_geometry_spinup_yr=ys,
        init_model_fls=fls_init,
        mb_model=mb_model,
        output_filesuffix=f'_oggm_static_past_{glacier_state}',
        store_model_geometry=True, store_fl_diagnostics=True)

    # conduct projection run for dynamic initialisation
    rid = '_{}_{}'.format(gcm, ssp)
    workflow.execute_entity_task(tasks.run_with_hydro, [gdir],
                                 run_task=tasks.run_from_climate_data,
                                 climate_filename='gcm_data',
                                 climate_input_filesuffix=rid,
                                 init_model_fls=spinup_model.fls,
                                 ys=ye,
                                 output_filesuffix='_oggm_dynamic_future_'
                                                   f'{glacier_state}',
                                 evolution_model=SemiImplicitModel,
                                 )

    # conduct projection run for static initialisation
    rid = '_{}_{}'.format(gcm, ssp)
    workflow.execute_entity_task(tasks.run_with_hydro, [gdir],
                                 run_task=tasks.run_from_climate_data,
                                 climate_filename='gcm_data',
                                 climate_input_filesuffix=rid,
                                 init_model_fls=static_model.fls,
                                 ys=ye,
                                 output_filesuffix='_oggm_static_future_'
                                                   f'{glacier_state}',
                                 evolution_model=SemiImplicitModel,
                                 )

    # calculate statistics
    calculate_default_oggm_statistics(gdir, glacier_state)
