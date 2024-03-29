import os
import pickle
import copy
import numpy as np
import xarray as xr
from oggm import utils, cfg


# define which statistic we want to compute for different types of data
def bias(a1, a2):
    return (a1 - a2).mean().item()


def add_1d_stats(x, y):
    diff = x - y
    rms_deviation = utils.rmsd(x, y)
    mean_absolute_deviation = utils.mad(x, y)
    max_absolute_deviation = np.max(np.abs(x - y))
    bias_deviation = bias(x, y)

    return {'rmsd': float(rms_deviation),
            'mean_ad': float(mean_absolute_deviation),
            'max_ad': float(max_absolute_deviation),
            'bias': float(bias_deviation),
            'diff': diff}


def add_0d_stats(x, y):
    return {'diff': float(x - y),
            'abs_diff': float(np.abs(x - y))}


def calculate_result_statistics(gdir, glacier_state, data_logger,
                                print_statistic=False):
    """calculate some statistics of the result for analysis"""

    # open the dataset of the run to add our calculated statistics
    fp = os.path.join(data_logger.output_filepath,
                      data_logger.filename + '.pkl')
    with open(fp, 'rb') as handle:
        ds = pickle.load(handle)

    # how well do we match the observations -----------------------------------
    obs_mdl = ds.observations_mdl[-1].values.item()
    obs_given = ds.attrs['observations']
    obs_stats = {}
    for obs_key in obs_mdl.keys():
        obs_stats[obs_key] = {}
        # the key consists of name and unit, here extract only name
        obs_key_name = obs_key.split(':')[0]
        for year_key in obs_mdl[obs_key].keys():
            if obs_key_name in ['fl_surface_h', 'fl_widths', 'us']:
                # here we compare data along the flowline
                obs_stats[obs_key][year_key] = add_1d_stats(
                    obs_mdl[obs_key][year_key],
                    obs_given[obs_key][year_key])
            elif obs_key_name in ['fl_total_area', 'area', 'dmdtda', 'volume']:
                obs_stats[obs_key][year_key] = add_0d_stats(
                    obs_mdl[obs_key][year_key],
                    obs_given[obs_key][year_key])
            else:
                raise NotImplementedError(f'{obs_key_name}')
    ds.attrs['observations_stats'] = obs_stats

    # how well do we match the control variables ------------------------------
    control_indices = ds.attrs['parameter_indices']
    controls_mdl = {}
    for control_var in control_indices.keys():
        controls_mdl[control_var] = \
            ds.unknown_parameters[-1][control_indices[control_var]].values
    controls_true = {}
    fls_true = gdir.read_pickle('model_flowlines',
                                filesuffix='_agile_true_init_'
                                           f'{glacier_state}')[0]
    for control_var in controls_mdl:
        if control_var in ['bed_h', 'area_bed_h']:
            controls_true[control_var] = \
                fls_true.bed_h[ds.ice_mask.values]
        elif control_var in ['lambdas', 'w0_m']:
            controls_true[control_var] = \
                getattr(fls_true, f'_{control_var}')[ds.ice_mask.values]
        elif control_var in ['height_shift_spinup']:
            controls_true[control_var] = controls_mdl[control_var]
        elif control_var in ['section']:
            fls_1980_true = gdir.read_pickle('model_flowlines',
                                             filesuffix='_creation_spinup_'
                                                        f'{glacier_state}')[0]
            controls_true[control_var] = \
                fls_1980_true.section[:len(controls_mdl['section'])]
        else:
            raise NotImplementedError(f'{control_var}')

    controls_stats = {}
    for control_var in controls_mdl.keys():
        if control_var in ['bed_h', 'area_bed_h', 'lambdas', 'w0_m']:
            if control_var == 'area_bed_h':
                obs_scal_key = 'observations_for_scaling'
                scaling_fct = \
                    ds.attrs[obs_scal_key]['fl_widths:m'][ds.ice_mask.values]
            else:
                scaling_fct = np.array([1])
            controls_stats[control_var] = add_1d_stats(
                controls_mdl[control_var] / scaling_fct,
                controls_true[control_var])
        elif control_var in ['height_shift_spinup']:
            controls_stats[control_var] = add_0d_stats(
                controls_mdl[control_var],
                controls_true[control_var])
        elif control_var in ['section']:
            controls_stats[control_var] = add_1d_stats(
                controls_mdl[control_var],
                controls_true[control_var])
        else:
            raise NotImplementedError(f'{control_var}')

    ds.attrs['controls_stats'] = controls_stats

    # how well do we match the past glacier state -----------------------------
    fls_start_mdl = copy.deepcopy(data_logger.flowlines[-1])
    sfc_h_start = copy.deepcopy(data_logger.sfc_h_start[-1])
    fls_start_mdl.surface_h = sfc_h_start
    fls_start_true = gdir.read_pickle('model_flowlines',
                                      filesuffix='_creation_spinup_'
                                                 f'{glacier_state}')[0]

    past_state_stats = {}
    for var in ['thick', 'area_m2', 'volume_m3']:
        if var in ['thick']:
            past_state_stats[var] = add_1d_stats(
                getattr(fls_start_mdl, var),
                getattr(fls_start_true, var))
        elif var == 'area_m2':
            def get_area(fl):
                return np.where(fl.thick > 0, fl.widths_m, 0) * fl.dx_meter

            past_state_stats[var] = add_1d_stats(
                get_area(fls_start_mdl),
                get_area(fls_start_true))
        elif var == 'volume_m3':
            def get_volume(fl):
                return fl.section * fl.dx_meter

            past_state_stats[var] = add_1d_stats(
                get_volume(fls_start_mdl),
                get_volume(fls_start_true))
        else:
            raise NotImplementedError(f'{var}')

    ds.attrs['past_state_stats'] = past_state_stats

    # how well do we match the past glacier evolution -------------------------
    fp = gdir.get_filepath('model_diagnostics',
                           filesuffix=data_logger.filename)
    with xr.open_dataset(fp) as ds_diag:
        past_evol_mdl = ds_diag.load()
    fp = gdir.get_filepath('model_diagnostics',
                           filesuffix='_agile_true_total_run_'
                                      f'{glacier_state}')
    with xr.open_dataset(fp) as ds_diag:
        past_evol_true = ds_diag.load()

    past_evol_stats = {}
    for var in ['volume_m3', 'area_m2']:
        past_evol_stats[var] = add_1d_stats(
                past_evol_mdl[var],
                past_evol_true[var])

    ds.attrs['past_evol_stats'] = past_evol_stats

    # how well do we match today's glacier state ------------------------------
    fls_end_mdl = copy.deepcopy(data_logger.flowlines[-1])
    fls_end_true = gdir.read_pickle('model_flowlines',
                                    filesuffix='_agile_true_end_'
                                               f'{glacier_state}')[0]

    today_state_stats = {}
    for var in ['thick', 'area_m2', 'volume_m3']:
        if var in ['thick']:
            today_state_stats[var] = add_1d_stats(
                getattr(fls_end_mdl, var),
                getattr(fls_end_true, var))
        elif var == 'area_m2':
            def get_area(fl):
                return np.where(fl.thick > 0, fl.widths_m, 0) * fl.dx_meter
            today_state_stats[var] = add_1d_stats(
                get_area(fls_end_mdl),
                get_area(fls_end_true))
        elif var == 'volume_m3':
            def get_volume(fl):
                return fl.section * fl.dx_meter
            today_state_stats[var] = add_1d_stats(
                get_volume(fls_end_mdl),
                get_volume(fls_end_true))
        else:
            raise NotImplementedError(f'{var}')

    ds.attrs['today_state_stats'] = today_state_stats

    # how well do we match the future glacier evolution -----------------------
    fp = gdir.get_filepath('model_diagnostics',
                           filesuffix=data_logger.filename + '_future')
    with xr.open_dataset(fp) as ds_diag:
        future_evol_mdl = ds_diag.load()
    fp = gdir.get_filepath('model_diagnostics',
                           filesuffix='_agile_true_future_'
                                      f'{glacier_state}')
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

    if print_statistic:
        # Here print the statistics in comparision to the default run

        # open default statistics
        fp_default = os.path.join(gdir.dir, f'default_oggm_statistics_'
                                            f'{glacier_state}.pkl')
        with open(fp_default, 'rb') as handle:
            default_stats = pickle.load(handle)

        # loop through the default statistics and show current
        print('')
        print('Statistic overview, values given as: current (oggm_dynamic, '
              'oggm_static)')
        for stat in default_stats.keys():
            if '_dynamic' in stat:
                stat_clean = stat.replace('_dynamic', '')
                if stat_clean == 'observations_stats':
                    print(f'{stat_clean}:')
                    for obs in default_stats[stat].keys():
                        for year in default_stats[stat][obs].keys():
                            if obs not in ds.attrs[stat_clean].keys():
                                continue
                            print(f'    {obs} ({year}): ')
                            for measure in default_stats[stat][obs][year].keys():
                                if measure == 'diff':
                                    continue
                                current_val = ds.attrs[stat_clean][obs][year][measure]
                                dynamic_val = default_stats[stat][obs][year][measure]
                                static_val = \
                                    default_stats[stat_clean + '_static'][obs][year][measure]
                                print(f'        '
                                      f'{measure}: {current_val:.2e} '
                                      f'({dynamic_val:.2e}, {static_val:.2e})')
                elif stat_clean in ['controls_stats', 'past_state_stats',
                                    'today_state_stats', 'past_evol_stats',
                                    'future_evol_stats']:
                    if stat_clean not in ds.attrs.keys():
                        continue
                    print(f'{stat_clean}:')
                    for var in default_stats[stat].keys():
                        if var not in ds.attrs[stat_clean].keys():
                            continue
                        print(f'    {var}: ')
                        for measure in default_stats[stat][var].keys():
                            if measure == 'diff':
                                continue
                            current_val = ds.attrs[stat_clean][var][measure]
                            dynamic_val = default_stats[stat][var][measure]
                            static_val = \
                                default_stats[stat_clean + '_static'][var][measure]
                            print(f'        '
                                  f'{measure}: {current_val:.2e} '
                                  f'({dynamic_val:.2e}, {static_val:.2e})')

                else:
                    raise NotImplementedError(f'{stat_clean}')


def calculate_default_oggm_statistics(gdir, glacier_state):

    default_oggm_statistics = {}

    for reali in ['dynamic', 'static']:
        # open the run files
        with xr.open_dataset(
                gdir.get_filepath('model_diagnostics',
                                  filesuffix=f'_oggm_{reali}_past_'
                                             f'{glacier_state}')) as ds:
            diag_past = ds.load()
        f = gdir.get_filepath('fl_diagnostics',
                              filesuffix=f'_oggm_{reali}_past_'
                                         f'{glacier_state}')
        with xr.open_dataset(f, group=f'fl_0') as ds:
            fl_diag_past = ds.load()
        with xr.open_dataset(
                gdir.get_filepath('model_diagnostics',
                                  filesuffix=f'_oggm_{reali}_future_'
                                             f'{glacier_state}')) as ds:
            diag_future = ds.load()

        # how well do we match the observations -------------------------------
        obs_given = gdir.read_pickle('inversion_input',
                                     filesuffix='_agile_measurements_'
                                                f'{glacier_state}')
        obs_stats = {}
        for obs_key in obs_given.keys():
            obs_stats[obs_key] = {}
            # the key consists of name and unit, here extract only name
            obs_key_name, obs_key_unit = obs_key.split(':')
            for year_key in obs_given[obs_key].keys():
                if obs_key_name in ['fl_surface_h', 'fl_widths', 'us']:
                    # here we compare data along the flowline
                    fl = fl_diag_past.sel(time=year_key)
                    if obs_key_name == 'fl_surface_h':
                        bed_h = fl.bed_h.values
                        thick = fl.thickness_m.values
                        obs_mdl = bed_h + thick
                        if obs_key_unit not in ['m']:
                            raise NotImplementedError(f'{obs_key_unit}')
                    elif obs_key_name == 'fl_widths':
                        area = fl.area_m2.values
                        obs_mdl = area / (fl.dx * fl.map_dx)
                        if obs_key_unit not in ['m']:
                            raise NotImplementedError(f'{obs_key_unit}')
                    elif obs_key_name == 'us':
                        obs_mdl = fl.ice_velocity_myr.values
                        if obs_key_unit not in ['myr-1']:
                            raise NotImplementedError(f'{obs_key_unit}')
                    else:
                        raise NotImplementedError(f'{obs_key_name}')
                    obs_stats[obs_key][year_key] = add_1d_stats(
                        obs_mdl,
                        obs_given[obs_key][year_key])

                elif obs_key_name in ['fl_total_area', 'area', 'volume', 'dmdtda']:
                    if obs_key_name in ['fl_total_area', 'area']:
                        obs_mdl = diag_past.sel(time=year_key).area_m2.values
                        if obs_key_unit == 'm2':
                            pass
                        elif obs_key_unit == 'km2':
                            obs_mdl *= 1e-6
                        else:
                            raise NotImplementedError(f'{obs_key_unit}')

                    elif obs_key_name == 'volume':
                        obs_mdl = diag_past.sel(time=year_key).volume_m3.values
                        if obs_key_unit == 'm3':
                            pass
                        elif obs_key_unit == 'km3':
                            obs_mdl *= 1e-9
                        else:
                            raise NotImplementedError(f'{obs_key_unit}')

                    elif obs_key_name == 'dmdtda':
                        ys, ye = year_key.split('-')
                        obs_mdl = (
                            # mass change
                            (diag_past.sel(time=ye).volume_m3 -
                             diag_past.sel(time=ys).volume_m3) *
                            cfg.PARAMS['ice_density']
                            # divided by mean area
                            / ((diag_past.sel(time=ye).area_m2 +
                                diag_past.sel(time=ys).area_m2) / 2.)
                            # divided by period
                            / (int(ye) - int(ys))
                        )
                        if obs_key_unit == 'kg m-2 yr-1':
                            pass
                        else:
                            raise NotImplementedError(f'{obs_key_unit}')
                    else:
                        raise NotImplementedError(f'{obs_key_name}')
                    obs_stats[obs_key][year_key] = add_0d_stats(
                        obs_mdl,
                        obs_given[obs_key][year_key])
                else:
                    raise NotImplementedError(f'{obs_key_name}')
        default_oggm_statistics[f'observations_stats_{reali}'] = obs_stats

        # how well do we match the control variables --------------------------
        all_control_vars = ['bed_h', 'area_bed_h', 'lambdas', 'w0_m']

        controls_mdl = {}
        controls_true = {}
        fls_mdl = gdir.read_pickle('model_flowlines',
                                   filesuffix='_oggm_first_guess_'
                                              f'{glacier_state}')[0]
        fls_true = gdir.read_pickle('model_flowlines',
                                    filesuffix='_agile_true_init_'
                                               f'{glacier_state}')[0]

        for control_var in all_control_vars:
            if control_var in ['bed_h', 'area_bed_h']:
                controls_mdl[control_var] = fls_mdl.bed_h
                controls_true[control_var] = fls_true.bed_h

            elif control_var in ['lambdas', 'w0_m']:
                controls_mdl[control_var] = getattr(fls_mdl, f'_{control_var}')
                controls_true[control_var] = getattr(fls_true, f'_{control_var}')

            else:
                raise NotImplementedError(f'{control_var}')

        controls_stats = {}
        for control_var in controls_mdl.keys():
            if control_var in ['bed_h', 'area_bed_h', 'lambdas', 'w0_m']:
                controls_stats[control_var] = add_1d_stats(
                    controls_mdl[control_var],
                    controls_true[control_var])

            else:
                raise NotImplementedError(f'{control_var}')

        default_oggm_statistics[f'controls_stats_{reali}'] = controls_stats

        # how well do we match the past glacier state -----------------------------
        fls_start_mdl = fl_diag_past.sel(time=fl_diag_past.time[0])
        fls_start_true = gdir.read_pickle('model_flowlines',
                                          filesuffix='_creation_spinup_'
                                                     f'{glacier_state}')[0]

        past_state_stats = {}
        for var in ['thick', 'area_m2', 'volume_m3']:
            if var in ['thick']:
                past_state_stats[var] = add_1d_stats(
                    getattr(fls_start_mdl, 'thickness_m').values,
                    getattr(fls_start_true, var))
            elif var == 'area_m2':
                def get_area(fl):
                    return np.where(fl.thick > 0, fl.widths_m, 0) * fl.dx_meter

                past_state_stats[var] = add_1d_stats(
                    getattr(fls_start_mdl, 'area_m2').values,
                    get_area(fls_start_true))
            elif var == 'volume_m3':
                def get_volume(fl):
                    return fl.section * fl.dx_meter

                past_state_stats[var] = add_1d_stats(
                    getattr(fls_start_mdl, 'volume_m3').values,
                    get_volume(fls_start_true))
            else:
                raise NotImplementedError(f'{var}')

        default_oggm_statistics[f'past_state_stats_{reali}'] = past_state_stats

        # how well do we match the past glacier evolution ---------------------
        past_evol_mdl = diag_past
        fp = gdir.get_filepath('model_diagnostics',
                               filesuffix='_agile_true_total_run_'
                                          f'{glacier_state}')
        with xr.open_dataset(fp) as ds_diag:
            past_evol_true = ds_diag.load()

        past_evol_stats = {}
        for var in ['volume_m3', 'area_m2']:
            past_evol_stats[var] = add_1d_stats(
                past_evol_mdl[var],
                past_evol_true[var])

        default_oggm_statistics[f'past_evol_stats_{reali}'] = past_evol_stats

        # how well do we match today's glacier state --------------------------
        fls_end_mdl = fl_diag_past.sel(time=fl_diag_past.time[-1])
        fls_end_true = gdir.read_pickle('model_flowlines',
                                        filesuffix='_agile_true_end_'
                                                   f'{glacier_state}')[0]

        today_state_stats = {}
        for var in ['thick', 'area_m2', 'volume_m3']:
            if var in ['thick']:
                today_state_stats[var] = add_1d_stats(
                    getattr(fls_end_mdl, 'thickness_m').values,
                    getattr(fls_end_true, var))
            elif var == 'area_m2':
                def get_area(fl):
                    return np.where(fl.thick > 0, fl.widths_m, 0) * fl.dx_meter

                today_state_stats[var] = add_1d_stats(
                    getattr(fls_end_mdl, 'area_m2').values,
                    get_area(fls_end_true))
            elif var == 'volume_m3':
                def get_volume(fl):
                    return fl.section * fl.dx_meter

                today_state_stats[var] = add_1d_stats(
                    getattr(fls_end_mdl, 'volume_m3').values,
                    get_volume(fls_end_true))
            else:
                raise NotImplementedError(f'{var}')

        default_oggm_statistics[f'today_state_stats_{reali}'] = today_state_stats

        # how well do we match the future glacier evolution -------------------
        future_evol_mdl = diag_future
        fp = gdir.get_filepath('model_diagnostics',
                               filesuffix='_agile_true_future_'
                                          f'{glacier_state}')
        with xr.open_dataset(fp) as ds_diag:
            future_evol_true = ds_diag.load()

        future_evol_stats = {}
        for var in ['volume_m3', 'area_m2']:
            future_evol_stats[var] = add_1d_stats(
                future_evol_mdl[var].values,
                future_evol_true[var].values)

        default_oggm_statistics[f'future_evol_stats_{reali}'] = future_evol_stats

    # save final default statistics as pickle
    out = os.path.join(gdir.dir,
                       f'default_oggm_statistics_{glacier_state}.pkl')
    with open(out, 'wb') as handle:
        pickle.dump(default_oggm_statistics, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
