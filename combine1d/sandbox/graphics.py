from oggm import cfg, utils
from combine1d.sandbox.create_glaciers_with_measurements import create_idealized_experiments
import matplotlib.pyplot as plt
from matplotlib import colors as m_colors
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
from collections import OrderedDict
import os
from oggm import graphics
from oggm.graphics import _plot_map, truncate_colormap
from oggm.core.massbalance import MonthlyTIModel, MultipleFlowlineMassBalance, RandomMassBalance
from oggm.core.flowline import FlowlineModel
import salem
from oggm.core import gis
import matplotlib


# define help functions for the plots
def get_fl_diagnostics(gdir, filesuffix):
    f = gdir.get_filepath('fl_diagnostics',
                          filesuffix=filesuffix)
    with xr.open_dataset(f, group=f'fl_0') as ds:
        ds = ds.load()
    return ds


def surf_to_nan(surf_h, thick):

    t1 = thick[:-2]
    t2 = thick[1:-1]
    t3 = thick[2:]
    pnan = ((t1 == 0) & (t2 == 0)) & ((t2 == 0) & (t3 == 0))
    surf_h[np.where(pnan)[0] + 1] = np.NaN
    return surf_h


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


# outline and DEM map
@_plot_map
def plot_domain_adapted(gdirs, ax=None, smap=None, use_netcdf=False):
    """Plot the glacier directory.

    Parameters
    ----------
    gdirs
    ax
    smap
    use_netcdf : bool
        use output of glacier_masks instead of geotiff DEM
    """
    gdir = gdirs[0]
    # Files
    if use_netcdf:
        with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
            topo = nc.variables['topo'][:]
    else:
        topo = gis.read_geotiff_dem(gdir)
    try:
        smap.set_data(topo, crs=gdir.grid.center_grid)
    except ValueError:
        pass

    cm = truncate_colormap(matplotlib.colormaps['terrain'], minval=0.25, maxval=1.0)
    smap.set_plot_params(cmap=cm)

    for gdir in gdirs:
        crs = gdir.grid.center_grid

        try:
            geom = gdir.read_pickle('geometries')

            # Plot boundaries
            poly_pix = geom['polygon_pix']
            smap.set_geometry(poly_pix, crs=crs, fc='white',
                              alpha=0.3, zorder=2, linewidth=.2)
            poly_pix = utils.tolist(poly_pix)
            for _poly in poly_pix:
                for l in _poly.interiors:
                    smap.set_geometry(l, crs=crs, color='black', linewidth=0.5)
        except FileNotFoundError:
            smap.set_shapefile(gdir.read_shapefile('outlines'))

    smap.plot(ax)

    return dict(cbar_label='Alt. [m]')


def reduce_grid_extend(gdir, factor=2):
    # template for new grid
    new_grid = {'proj': gdir.grid.proj, 'nxny': None,
                'dxdy': (gdir.grid.dx, gdir.grid.dy), 'x0y0': None,
                'pixel_ref': gdir.grid.pixel_ref}
    # unchanged grid properties

    # define new number of grid points by dividing through factor
    new_nx = np.floor_divide(gdir.grid.nx, factor)
    new_ny = np.floor_divide(gdir.grid.ny, factor)
    new_grid['nxny'] = (new_nx, new_ny)

    # calculate new starting point
    (left, right, bottom, top) = gdir.grid.extent_in_crs(new_grid['proj'])

    diff_nx = np.floor_divide(gdir.grid.nx - new_nx, 2)
    new_left = left + diff_nx * gdir.grid.dx
    new_right = right - diff_nx * gdir.grid.dx

    diff_ny = np.floor_divide(gdir.grid.ny - new_ny, 2)
    if np.sign(gdir.grid.dy) < 0:
        new_top = top + diff_ny * gdir.grid.dy
        new_grid['x0y0'] = (new_left, new_top)
    else:
        new_bottom = bottom + diff_ny * gdir.grid.dy
        new_grid['x0y0'] = (new_left, new_bottom)

    return salem.gis.Grid.from_dict(new_grid)


def plot_outline_with_dem(gdir, ax=None, map_extend_factor=3, use_netcdf=False):
    grid = reduce_grid_extend(gdir, factor=map_extend_factor)
    smap = salem.Map(grid,
                     countries=False,
                     nx=grid.nx)
    plot_domain_adapted(gdir, ax=ax, smap=smap, use_netcdf=use_netcdf,
                        add_colorbar=False,
                        lonlat_contours_kwargs={'interval': 0})

    # clear plot
    ax.set_title('')


# heading text
def plot_heading_text(gdir, fig, ax, parameters_text_height=0.5, linespacing=1):

    ax.axis('off')

    ax.text(
        0.5,
        1,
        f'{gdir.rgi_id}: {gdir.name}',
        fontsize=16,
        horizontalalignment='center',
        verticalalignment='top',
        transform=ax.transAxes
    )

    props = dict(boxstyle='round', facecolor='grey', alpha=0.1, pad=0.6)

    ax.text(
        0.5,
        parameters_text_height,
        f"Center Longitude = {gdir.cenlon:.2f}°, "
        f"Center Latitude = {gdir.cenlat:.2f}°"
        "\n"
        fr" $A$ = {gdir.get_diagnostics()['inversion_glen_a']:.1e} "
        r"s$^{-1}$ Pa$^{-3}$, "
        fr"$f_s$ = {gdir.get_diagnostics()['inversion_fs']} "
        "\n"
        fr"$d_f$ = {gdir.read_json('mb_calib')['melt_f']:.1f} "
        r"kg m$^{-2}$ K$^{-1}$ day$^{-1}$, "
        fr"$p_f$ = {gdir.read_json('mb_calib')['prcp_fac']:.1f}, "
        fr"$t_b$ = {gdir.read_json('mb_calib')['temp_bias']:.1f} °C",
        horizontalalignment='center',
        verticalalignment='top',
        transform=ax.transAxes,
        linespacing=linespacing,
        bbox=props
    )


# glacier bed with hypsomentry
# adapted from graphics.plot_modeloutput_section
def plot_bed_hypsometry(gdir, ax, fig, grid_points_added=5, plot_non_trap=True,
                        bins=20, legend_x_position=0.8, legend_y_position=1.0,
                        add_future=True, ye=2050):
    # open flowlines to be plotted (only for single flowline models)
    fls_true_start = gdir.read_pickle('model_flowlines', filesuffix='_creation_spinup')[0]
    fls_true_rgi = gdir.read_pickle('model_flowlines', filesuffix='_combine_true_init')[0]
    fls_true_end = gdir.read_pickle('model_flowlines', filesuffix='_combine_true_end')[0]

    if add_future:
        fls_true_future = get_fl_diagnostics(gdir, '_combine_true_future')
        fls_true_future = fls_true_future.loc[{'time': ye}]
        fls_true_future_thickness = fls_true_future.thickness_m.values
        fls_true_future_surface_h = fls_true_future.bed_h.values + fls_true_future_thickness
        fls_true_future_area = fls_true_future.area_m2.values

    # define colors
    color_start = 'C2'
    color_rgi = 'C1'
    color_end = 'C0'
    color_future = 'C3'

    # define extend to plot according to longest flowline
    max_grid_point = np.max(
        [np.argwhere(fls_true_start.thick > 0)[-1][0] + 1,
         np.argwhere(fls_true_rgi.thick > 0)[-1][0] + 1,
         np.argwhere(fls_true_end.thick > 0)[-1][0] + 1])
    max_grid_point += grid_points_added

    # Compute area histo
    def compute_area_histo(fls):
        area = np.array([])
        height = np.array([])
        bed = np.array([])

        a = fls.widths_m[:max_grid_point] * fls.dx_meter * 1e-6
        a = np.where(fls.thick[:max_grid_point] > 0, a, 0)
        area = np.concatenate((area, a))
        height = np.concatenate((height, fls.surface_h[:max_grid_point]))
        bed = np.concatenate((bed, fls.bed_h[:max_grid_point]))
        return area, height, bed

    area_true_start, height_true_start, bed_true_start = compute_area_histo(fls_true_start)
    area_true_rgi, height_true_rgi, bed_true_rgi = compute_area_histo(fls_true_rgi)
    area_true_end, height_true_end, bed_true_end = compute_area_histo(fls_true_end)

    if add_future:
        area_true_future = np.where(fls_true_future_thickness[:max_grid_point] > 0,
                                    fls_true_future_area[:max_grid_point], 0) * 1e-6
        height_true_future = fls_true_future_surface_h[:max_grid_point]

    ylim = [np.min([bed_true_start.min(),
                    bed_true_rgi.min(),
                    bed_true_end.min()]),
            np.max([height_true_start.max(),
                    height_true_rgi.max(),
                    height_true_end.max()])]

    # Plot histo
    posax = ax.get_position()
    posax = [posax.x0 + 2 * posax.width / 3.0,
             posax.y0, posax.width / 3.0,
             posax.height]
    axh = fig.add_subplot(frameon=False)
    axh.set_position(posax)

    axh.hist(height_true_end, orientation='horizontal', range=ylim, bins=bins,
             alpha=1, weights=area_true_end, color=color_end,
             )
    axh.hist(height_true_rgi, orientation='horizontal', range=ylim, bins=bins,
             alpha=1, weights=area_true_rgi, color=color_rgi,
             histtype='step', lw=2.1
             )
    axh.hist(height_true_start, orientation='horizontal', range=ylim, bins=bins,
             alpha=1, weights=area_true_start, color=color_start, histtype='step', lw=1.4)

    if add_future:
        axh.hist(height_true_future, orientation='horizontal', range=ylim, bins=bins,
                 alpha=1, weights=area_true_future, color=color_future,
                 histtype='step',
                 lw=1.4
                 )

    axh.invert_xaxis()
    axh.xaxis.tick_top()
    axh.set_xlabel('Area [km$^2$]')
    axh.xaxis.set_label_position('top')
    axh.set_ylim(ylim)
    axh.yaxis.set_ticks_position('right')
    axh.set_yticks([])
    axh.axhline(y=ylim[1], color='black', alpha=1)  # qick n dirty trick

    # plot Centerlines
    x = np.arange(max_grid_point) * fls_true_rgi.dx * fls_true_rgi.map_dx

    # Plot the bed
    ax.plot(x, fls_true_rgi.bed_h[:max_grid_point], color='k', linewidth=2.5, label=r'$b_{true}$')

    if plot_non_trap:
        # Where not trapezoid change color
        bed_no_t = fls_true_rgi.bed_h[:max_grid_point] * np.NaN
        pt = (~fls_true_rgi.is_trapezoid[:max_grid_point]) | fls_true_rgi.is_rectangular[
                                                             :max_grid_point]
        bed_no_t[pt] = fls_true_rgi.bed_h[:max_grid_point][pt]
        ax.plot(x, bed_no_t, '*', color='red', linewidth=2.5, ms=10,
                zorder=90)

    def plot_glacier_surface(fl, label, color):
        surfh = surf_to_nan(fl.surface_h[:max_grid_point],
                            fl.thick[:max_grid_point])
        ax.plot(x, surfh,
                color=color, linewidth=2,
                label=label)

    # Plot glacier surface start, rgi, end
    plot_glacier_surface(fls_true_start, label='1980', color=color_start)
    plot_glacier_surface(fls_true_rgi, label=f'{gdir.rgi_date}', color=color_rgi)
    plot_glacier_surface(fls_true_end, label='2020', color=color_end)

    if add_future:
        surfh = surf_to_nan(fls_true_future_surface_h[:max_grid_point],
                            fls_true_future_thickness[:max_grid_point])
        ax.plot(x, surfh,
                color=color_future, linewidth=2,
                label=ye)

    ax.set_ylim(ylim)

    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Distance along flowline [m]')
    ax.set_ylabel('Altitude [m]')

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(list(by_label.values()), list(by_label.keys()),
              bbox_to_anchor=(legend_x_position, legend_y_position),
              frameon=False)


# Mass Balance plot
def plot_specific_mb(gdir, ax,
                     ys=1920,
                     ye=2100,
                     y_switch_spinup_run=1980,
                     y_switch_run_future=2020,
                     text_position=1.1,
                     add_text=True, add_future=True):
    # define mass balance models to use
    fls_spinup = gdir.read_pickle('model_flowlines',
                                  filesuffix='_consensus')
    mb_model_random = MultipleFlowlineMassBalance(gdir,
                                                  fls=fls_spinup,
                                                  mb_model_class=RandomMassBalance,
                                                  seed=2,
                                                  filename='climate_historical',
                                                  y0=1950,
                                                  halfsize=30)
    mb_model_run = MultipleFlowlineMassBalance(gdir,
                                               fls=fls_spinup,
                                               mb_model_class=MonthlyTIModel,
                                               filename='climate_historical')
    gcm = 'BCC-CSM2-MR'
    ssp = 'ssp370'
    rid = '_{}_{}'.format(gcm, ssp)
    mb_model_future = MultipleFlowlineMassBalance(gdir,
                                                  fls=fls_spinup,
                                                  mb_model_class=MonthlyTIModel,
                                                  filename='gcm_data',
                                                  input_filesuffix=rid)
    fls = gdir.read_pickle('model_flowlines',
                           filesuffix='_combine_true_init')

    # read out mb values
    specific_mb = []
    if add_future:
        years = np.arange(ys, ye)
    else:
        years = np.arange(ys, y_switch_run_future)
    for yr in years:
        if yr < y_switch_spinup_run:
            specific_mb.append(mb_model_random.get_specific_mb(fls=fls, year=yr))
        elif yr < y_switch_run_future:
            specific_mb.append(mb_model_run.get_specific_mb(fls=fls, year=yr))
        else:
            if add_future:
                specific_mb.append(mb_model_future.get_specific_mb(fls=fls, year=yr))

    ax.plot(years, specific_mb, lw=2, zorder=1,
            # label='Annual'
            )

    ax.axhline(0, ls='--', color='gray', zorder=0, alpha=0.5)

    if add_text:
        ax.axvline(y_switch_spinup_run, ls='-', color='black', lw=1, zorder=3)
        ylim = ax.get_ylim()
        ax.text(1950,
                ylim[0] + (ylim[1] - ylim[0]) * text_position,
                'Random Forcing',
                horizontalalignment='center',
                verticalalignment='center',
                )
        annual_text_x = 2000
        if add_future:
            annual_text_x = (y_switch_spinup_run + ye) / 2
        ax.text(annual_text_x,
                ylim[0] + (ylim[1] - ylim[0]) * text_position,
                'Annual Forcing',
                horizontalalignment='center',
                verticalalignment='center',
                )

    # add moving average
    window = 7  # should be an odd number
    halfsize_window = int((window - 1) / 2)
    specific_mb_avg = np.concatenate([np.full(halfsize_window, np.nan),
                                      moving_average(np.array(specific_mb), window),
                                      np.full(halfsize_window, np.nan)])
    ax.plot(years, specific_mb_avg, lw=2, label=f'{window}-yr average', zorder=2)

    ax.legend(frameon=True)
    ax.set_ylabel('Mass-Balance\n[kg yr-1]')


# glacier evolution during creation
def plot_glacier_evolution(gdir, ax_v, ax_a, ax_l, text_position=1.1,
                           add_text=True, add_future=True, add_fg=True,
                           ye=2100):
    color = 'C0'
    color_fg = 'C1'

    # open glacier creation
    with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                           filesuffix='_combine_creation_spinup')) as ds:
        ds_spinup = ds.load()

    # open actual historical run
    with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                           filesuffix='_combine_true_total_run')) as ds:
        ds_run = ds.load()

    # open future run
    with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                           filesuffix='_combine_true_future')) as ds:
        ds_future = ds.load()

    if add_fg:
        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_oggm_dynamic_past')) as ds:
            ds_run_fg = ds.load()

        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_oggm_dynamic_future')) as ds:
            ds_future_fg = ds.load()

    if add_future:
        ds_future = ds_future.sel(time=ds_future.time[ds_future.time <= ye])
        ds_future_fg = ds_future_fg.sel(time=ds_future_fg.time[ds_future_fg.time <= ye])

    # plot volume
    ax_v.plot(ds_spinup.time, ds_spinup.volume_m3 * 1e-9, color=color,
              label='true')
    ax_v.plot(ds_run.time, ds_run.volume_m3 * 1e-9, color=color)
    if add_future:
        ax_v.plot(ds_future.time, ds_future.volume_m3 * 1e-9, color=color)
    if add_fg:
        ax_v.plot(ds_run_fg.time, ds_run_fg.volume_m3 * 1e-9, color=color_fg,
                  label='oggm default')
        if add_future:
            ax_v.plot(ds_future_fg.time, ds_future_fg.volume_m3 * 1e-9, color=color_fg)
    ax_v.axvline(ds_spinup.time[-1], ls='-', color='black', lw=1)
    ax_v.set_ylabel('Volume\n[km3]')
    ax_v.yaxis.tick_right()
    ax_v.yaxis.set_label_position("right")
    if add_text:
        ylim = ax_v.get_ylim()
        ax_v.text(1950,
                  ylim[0] + (ylim[1] - ylim[0]) * text_position,
                  'Random Forcing',
                  horizontalalignment='center',
                  verticalalignment='center',
                  )
        annual_text_x = 2000
        if add_future:
            annual_text_x = 2040
        ax_v.text(annual_text_x,
                  ylim[0] + (ylim[1] - ylim[0]) * text_position,
                  'Annual Forcing',
                  horizontalalignment='center',
                  verticalalignment='center',
                  )
    if add_fg:
        ax_v.legend(frameon=True)

    # plot area
    ax_a.plot(ds_spinup.time, ds_spinup.area_m2 * 1e-6, color=color, label='Area')
    ax_a.plot(ds_run.time, ds_run.area_m2 * 1e-6, color=color)
    if add_future:
        ax_a.plot(ds_future.time, ds_future.area_m2 * 1e-6, color=color)
    if add_fg:
        ax_a.plot(ds_run_fg.time, ds_run_fg.area_m2 * 1e-6, color=color_fg,
                  label='First Guess')
        if add_future:
            ax_a.plot(ds_future_fg.time, ds_future_fg.area_m2 * 1e-6, color=color_fg)
    ax_a.axvline(ds_spinup.time[-1], ls='-', color='black', lw=1)
    ax_a.set_ylabel('Area\n[km2]')
    ax_a.yaxis.tick_right()
    ax_a.yaxis.set_label_position("right")
    # ax_a.legend(frameon=True)

    if ax_l is not None:
        # plot length
        ax_l.plot(ds_spinup.time, ds_spinup.length_m)
        ax_l.plot(ds_run.time, ds_run.length_m)
        if add_future:
            ax_l.plot(ds_future.time, ds_future.length_m)
        if add_fg:
            ax_l.plot(ds_run_fg.time, ds_run_fg.length_m, color=color_fg,
                      label='First Guess')
            if add_future:
                ax_l.plot(ds_future_fg.time, ds_future_fg.length_m, color=color_fg)
        ax_l.set_ylabel('length [m]')


# dhdt, bed diff plot
def plot_dhdt_first_guess_db(gdir, ax_dh, ax_db, ax_ds=None, grid_points_added=0):
    # open flowlines to be plotted (only for single flowline models)
    fls_true_rgi = gdir.read_pickle('model_flowlines', filesuffix='_combine_true_init')[0]
    fls_first_guess = gdir.read_pickle('model_flowlines', filesuffix='_combine_first_guess')[0]


    # define extend to plot according to longest flowline
    max_grid_point = np.max(
        [
            np.argwhere(fls_true_rgi.thick > 0)[-1][0] + 1,
        ])
    max_grid_point += grid_points_added

    # open different flowline diagnostics
    ds_start = get_fl_diagnostics(gdir, '_combine_true_dmdt_start')
    ds_init = get_fl_diagnostics(gdir, '_combine_true_init')
    ds_end = get_fl_diagnostics(gdir, '_combine_true_end')
    ds_default = get_fl_diagnostics(gdir, '_oggm_dynamic_past').sel({'time': 2000})

    ds_all = xr.merge([ds_start, ds_init, ds_end])

    # small test
    for year in np.arange(1980, 2001):
        assert np.allclose(ds_start.sel(time=year).thickness_m,
                           ds_all.sel(time=year).thickness_m)
    for year in np.arange(2001, gdir.rgi_date + 1):
        assert np.allclose(ds_init.sel(time=year).thickness_m,
                           ds_all.sel(time=year).thickness_m)
    for year in np.arange(gdir.rgi_date + 1, 2020):
        assert np.allclose(ds_end.sel(time=year).thickness_m,
                           ds_all.sel(time=year).thickness_m)

    periods = [
        (gdir.rgi_date - 1, gdir.rgi_date + 1),
    ]

    dhdts = []
    for period in periods:
        start, end = period
        dhdts.append(ds_all.sel(time=end).thickness_m[:max_grid_point].values -
                     ds_all.sel(time=start).thickness_m[:max_grid_point].values)

    # add first guess bed h difference
    bed_h_diff = fls_first_guess.bed_h[:max_grid_point] - fls_true_rgi.bed_h[:max_grid_point]

    dhdts.append(bed_h_diff)
    dhdts = np.stack(dhdts)

    # caluculate surface height difference
    if ax_ds is not None:
        surface_h_true = fls_true_rgi.surface_h
        surface_h_default = ds_default.bed_h + ds_default.thickness_m
        delta_surface = surface_h_default - surface_h_true

    # actual plotting
    ax_dh.plot(ds_all.dis_along_flowline[:max_grid_point].values,
               dhdts[0],
               )
    ax_dh.axhline(0, ls='--', color='gray', zorder=0, alpha=0.5)
    ax_dh.set_ylabel(r'$\frac{\mathrm{d}h}{\mathrm{d}t}$ 2000' + '\n[m]')
    ax_dh.legend(frameon=False)

    ax_db.plot(ds_all.dis_along_flowline[:max_grid_point].values,
               bed_h_diff,
               )
    ax_db.axhline(0, ls='--', color='gray', zorder=0, alpha=0.5)
    ax_db.set_ylabel(r'$b_{fg}$ - $b_{true}$' + '\n[m]')
    ax_db.legend(frameon=False)

    if ax_ds is not None:
        ax_ds.plot(delta_surface.dis_along_flowline[:max_grid_point].values,
                   delta_surface[:max_grid_point].values,
                   # label='Surface height difference 2000'
                   )
        ax_ds.axhline(0, ls='--', color='gray', zorder=0, alpha=0.5)
        ax_ds.set_ylabel(r'$s_{fg}$ - $s_{true}$' + '\n[m]')
        ax_ds.legend(frameon=False)


# plot velocities
def plot_velocities(gdir, ax, grid_points_added=5):
    # open flowlines to be plotted (only for single flowline models)
    fls_true_start = gdir.read_pickle('model_flowlines', filesuffix='_creation_spinup')[0]
    fls_true_rgi = gdir.read_pickle('model_flowlines', filesuffix='_combine_true_init')[0]
    fls_true_end = gdir.read_pickle('model_flowlines', filesuffix='_combine_true_end')[0]

    # define extend to plot according to longest flowline
    max_grid_point = np.max(
        [np.argwhere(fls_true_start.thick > 0)[-1][0] + 1,
         np.argwhere(fls_true_rgi.thick > 0)[-1][0] + 1,
         np.argwhere(fls_true_end.thick > 0)[-1][0] + 1])
    max_grid_point += grid_points_added

    # open different flowline diagnostics
    ds_start = get_fl_diagnostics(gdir, '_combine_true_dmdt_start')
    ds_init = get_fl_diagnostics(gdir, '_combine_true_init')
    ds_end = get_fl_diagnostics(gdir, '_combine_true_end')

    ds_all = xr.merge([ds_start, ds_init, ds_end])

    # define colors
    color_start = 'green'
    color_rgi = 'blue'
    color_end = 'purple'

    # actual plot
    ax.plot(ds_all.dis_along_flowline[:max_grid_point].values,
            ds_all.sel(time=1981).ice_velocity_myr[:max_grid_point].values,
            label='1981', color=color_start)
    ax.plot(ds_all.dis_along_flowline[:max_grid_point].values,
            ds_all.sel(time=gdir.rgi_date).ice_velocity_myr[:max_grid_point].values,
            label=f'{gdir.rgi_date} (RGI-date)', color=color_rgi)
    ax.plot(ds_all.dis_along_flowline[:max_grid_point].values,
            ds_all.sel(time=2020).ice_velocity_myr[:max_grid_point].values,
            label='2020', color=color_end)

    ax.set_ylabel('ice velocity [m yr-1]')


# create the whole plot
def create_synthetic_glacier_plot(gdir, fig, save_fig=False, fig_dir='',
                                  legend_position=(0.2, 1.2),
                                  map_extend_factor=2,
                                  add_future=False, add_fg=True, ye=2020):
    top_margin = 0.02
    bottom_margin = 0.1

    left_margin_plot = 0.13
    right_margin_plot = 0.13
    left_margin_heading = 0.07
    right_margin_heading = 0.07

    heading_bottom = 0.75
    heading_space = 0.08
    left_right_space = 0.03
    height_v_a_ax = 0.35
    mb_v_a_space = 0.05

    SMALL_SIZE = 13
    MEDIUM_SIZE = 13
    BIGGER_SIZE = 13

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # heading
    gs1 = fig.add_gridspec(nrows=1, ncols=4,
                           left=0 + left_margin_heading,
                           right=1 - right_margin_heading,
                           bottom=heading_bottom,
                           top=1 - top_margin)

    ax_map = fig.add_subplot(gs1[0, 0])
    ax_text = fig.add_subplot(gs1[0, 1:])

    # left column (bed, db, dh, ds)
    gs2 = fig.add_gridspec(nrows=4, ncols=1,
                           left=0 + left_margin_plot,
                           right=0.5 - left_right_space / 2,
                           bottom=0 + bottom_margin,
                           top=heading_bottom - heading_space,
                           hspace=0.1)

    ax_bed = fig.add_subplot(gs2[0:2, :])
    ax_db = fig.add_subplot(gs2[2, :])
    ax_dh = fig.add_subplot(gs2[3, :])
    # ax_ds = fig.add_subplot(gs2[4, :])

    # right column (MB, V, A)
    gs4 = fig.add_gridspec(nrows=3, ncols=1,
                           left=0.5 + left_right_space / 2,
                           right=1 - right_margin_plot,
                           bottom=0 + bottom_margin,
                           top=heading_bottom - heading_space,
                           # 0 + bottom_margin + height_v_a_ax - mb_v_a_space / 2,
                           hspace=0.1)

    ax_mb = fig.add_subplot(gs4[0, 0])
    ax_v = fig.add_subplot(gs4[1, 0])
    ax_a = fig.add_subplot(gs4[2, 0])

    ax_mb.xaxis.set_tick_params(labelbottom=False)

    ax_v.xaxis.set_tick_params(labelbottom=False)
    ax_a.set_xlabel('Time (years)')

    ax_mb.yaxis.tick_right()
    ax_mb.yaxis.set_label_position("right")
    ax_v.yaxis.tick_right()
    ax_v.yaxis.set_label_position("right")
    ax_a.yaxis.tick_right()
    ax_a.yaxis.set_label_position("right")

    # heading plots
    plot_outline_with_dem(gdir, ax=ax_map, map_extend_factor=map_extend_factor)
    plot_heading_text(gdir, fig, ax_text,
                      parameters_text_height=0.7,
                      linespacing=1.3)

    # left column plots
    plot_bed_hypsometry(gdir, ax=ax_bed, fig=fig, grid_points_added=15, bins=15,
                        legend_x_position=legend_position[0],
                        legend_y_position=legend_position[1],
                        add_future=add_future, ye=ye)
    plot_dhdt_first_guess_db(gdir,
                             ax_dh=ax_dh,
                             ax_db=ax_db,
                             # ax_ds=ax_ds,
                             grid_points_added=1)

    xlim_left = ax_bed.get_xlim()
    ax_dh.set_xlim(xlim_left)
    ax_db.set_xlim(xlim_left)

    ax_bed.xaxis.set_tick_params(labelbottom=False)
    ax_bed.set_xlabel('')
    ax_db.xaxis.set_tick_params(labelbottom=False)
    ax_dh.set_xlabel('Distance along flowline [km]')

    # remove last label of left row
    labels = [int(float(label.get_text().replace('−', '-')) / 1e3) for label in
              ax_dh.get_xticklabels()]
    ax_dh.set_xticklabels(labels)

    # right column plots
    plot_specific_mb(gdir, ax=ax_mb, ys=1920, ye=ye,
                     add_text=True, add_future=add_future)
    plot_glacier_evolution(gdir,
                           ax_v=ax_v,
                           ax_a=ax_a,
                           ax_l=None,
                           add_text=False,
                           add_future=add_future,
                           add_fg=add_fg,
                           ye=ye
                           )

    xlim_right = ax_mb.get_xlim()
    ax_v.set_xlim(xlim_right)
    ax_a.set_xlim(xlim_right)

    if add_future:
        x_ticks_time = np.arange(1920, ye, 20)
        x_ticks_time_labels = []
        for tick in x_ticks_time:
            if tick == 1920:
                tick = '0'
            elif tick == 1940:
                tick = '20'
            elif tick == 1960:
                tick = '40'
            x_ticks_time_labels.append(str(tick))
        ax_mb.set_xticks(x_ticks_time)
        ax_v.set_xticks(x_ticks_time)
        ax_a.set_xticks(x_ticks_time, labels=x_ticks_time_labels)
    else:
        ax_a.set_xticklabels(["", "0", "20", "40", "1980", "2000", "2020", "2040"])

    if save_fig:
        result_folder = utils.mkdir(fig_dir)
        fig.savefig(os.path.join(result_folder, f'{gdir.name}.png'))


def create_and_save_all_synthetic_glacier_plots(fig_dir='', **kwargs):

    if fig_dir == '':
        raise ValueError('You should define a directory where to save the '
                         'resulting plots!')

    cfg.initialize(logging_level='WARNING')
    cfg.PATHS['working_dir'] = utils.gettempdir(dirname='Synthetic_glacier_plots',
                                                reset=True)
    cfg.PARAMS['use_multiprocessing'] = False

    prepro_border = 160
    from_prepro_level = 2
    base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/' \
               'L1-L2_files/elev_bands/'

    glacier_names = ['Baltoro',
                     'Aletsch',
                     'Artesonraju',
                     'Peyto',
                     ]

    cfg.PARAMS['cfl_number'] = 0.5
    gdirs = create_idealized_experiments(glacier_names,
                                         prepro_border=prepro_border,
                                         from_prepro_level=from_prepro_level,
                                         base_url=base_url, )

    glacier_kwargs = {'RGI60-14.06794':
                          {'legend_position': (0.3, 1.21),
                           'map_extend_factor': 1.7},
                      'RGI60-11.01450':
                          {'legend_position': (0.3, 1.36),
                           'map_extend_factor': 3},
                      'RGI60-16.02444':
                          {'legend_position': (0.3, 1.21),
                           'map_extend_factor': 2.5},
                      'RGI60-02.05098':
                          {'legend_position': (0.3, 1.3),
                           'map_extend_factor': 2},
                      }

    for gdir in gdirs:
        kwargs_use = glacier_kwargs[gdir.rgi_id]
        save_fig = True

        fig = plt.figure(figsize=(10, 6))

        for key, item in kwargs.items():
            kwargs_use[key] = item

        create_synthetic_glacier_plot(gdir, fig, save_fig=save_fig,
                                      fig_dir=fig_dir,
                                      **kwargs_use)
