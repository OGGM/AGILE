import pickle
import re
from functools import partial
from itertools import compress
import os
import io
import torch
import pandas as pd
import numpy as np
from bokeh.models import HoverTool
import panel as pn
import holoviews as hv
from holoviews import opts

pn.extension('tabulator')
hv.extension('bokeh')

from oggm import cfg, utils, workflow, tasks, graphics

cfg.initialize(logging_level='WARNING')


# workaround to unpickle tensor on cpu which were originally stored on gpu
class CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


options_selects = []
all_experiment_settings = {}
experiment_select = []
glacier_select = []
menu = pn.Column()
button = pn.widgets.Button(name='Select new one', button_type='primary')
open_files = {}


def define_options_for_experiment(event):
    global options_selects
    global menu
    global button
    global experiment_select
    options_selects = []
    all_options = all_experiment_settings[experiment_select.value]
    main_options = []
    option_used = []
    for opt in all_options:
        if ''.join(re.findall("[a-zA-Z]+", opt)) not in option_used:
            main_options.append(opt)
            option_used.append(''.join(re.findall("[a-zA-Z]+", opt)))
    for main_opt in main_options:
        name_tmp = ''.join(re.findall("[a-zA-Z]+", main_opt))
        index = [all([tmp == '' or tmp.isdecimal()
                      for tmp in opt.split(name_tmp)])
                 for opt in all_options]
        options_tmp = list(compress(all_options, index))
        options_selects.append(pn.widgets.Select(name=name_tmp, options=options_tmp))

    new_menu = pn.Column(glacier_select,
                         experiment_select)

    for opt_select in options_selects:
        new_menu.append(opt_select)

    new_menu.append(button)

    menu.objects = [new_menu]


def individual_experiment_dashboard(working_dir, input_folder,
                                    option_description):
    global menu
    global all_experiment_settings
    global experiment_select
    global glacier_select
    global open_files
    cfg.PATHS['working_dir'] = working_dir
    gdirs = workflow.init_glacier_directories()

    # Extract info from directory
    all_files = os.listdir(input_folder)
    all_glaciers = []
    all_experiments = []

    open_files = {}

    for file in all_files:
        # file = all_files[0]
        glacier_experiment, experiment_settings = file.split('-')

        # get glacier and experiment
        glacier = glacier_experiment.split('_')[0]
        if glacier not in all_glaciers:
            all_glaciers.append(glacier)
        experiment = '_'.join(glacier_experiment.split('_')[1:])
        if experiment not in all_experiments:
            all_experiments.append(experiment)
            all_experiment_settings[experiment] = []

        # get different settings
        experiment_settings = os.path.splitext(experiment_settings)[0]
        for exp_set in experiment_settings.split('_'):
            if exp_set != '' and exp_set not in all_experiment_settings[experiment]:
                all_experiment_settings[experiment].append(exp_set)

    all_experiment_settings[experiment].sort()

    # Plot for investigating iterations of single expriment
    # Define Performance measurements
    def BIAS(a1, a2):
        return (a1 - a2).mean().item()

    def RMSE(a1, a2):
        return np.sqrt(((a1 - a2) ** 2).mean()).item()

    def DIFF(a1, a2):
        return np.max(np.abs(a1 - a2)).item()

    def AERR(a1, a2):
        return np.mean(np.abs(a1 - a2)).item()

    # Define menu
    glacier_select = pn.widgets.Select(name='glacier', options=all_glaciers)
    experiment_select = pn.widgets.Select(name='experiment', options=all_experiments)

    experiment_select.param.watch(define_options_for_experiment, 'value')

    define_options_for_experiment(None)

    # Add table with explanation of experiments
    def get_description_accordion():
        accordion = pn.Accordion()
        for key in option_description.keys():
            # key = 'control_vars'
            options = list(option_description[key].keys())
            settings = [str(option_description[key][setting]) for setting in options]
            df = pd.DataFrame({'options': options, 'settings': settings})
            accordion.append((key, pn.widgets.Tabulator(df.set_index('options'))))
        accordion.toogle = True
        return accordion

    # Define individual plots
    def get_individual_plot(current_file):
        ds = open_files[current_file]

        # get reference flowline for true values
        rgi_id = ds.attrs['rgi_id']
        # rgi_id = translate_name_rgi[glacier_select.value]
        for gdir in gdirs:
            if gdir.rgi_id == rgi_id:
                fl_ref = gdir.read_pickle('model_flowlines',
                                          filesuffix='_combine_true_init')[0]

        # now calculate data for delta bed_h and w0_m
        data_bed_h = []
        d_bed_h_lim = 0
        data_w0_m = []
        d_w0_m_lim = 0
        for i, fl in enumerate(ds.flowlines.values):
            x_all = ds.coords['x'][ds.ice_mask].values

            # bed_h
            d_bed_h = (fl.bed_h - fl_ref.bed_h)[ds.ice_mask]
            d_bed_h_lim = np.max([d_bed_h_lim, np.max(np.abs(d_bed_h))])
            for el in [(x, i, v) for x, v in zip(x_all, d_bed_h)]:
                data_bed_h.append(el)

            # w0_m
            d_w0_m = (fl._w0_m - fl_ref._w0_m)[ds.ice_mask]
            d_w0_m_lim = np.max([d_w0_m_lim, np.max(np.abs(d_w0_m))])
            for el in [(x, i, v) for x, v in zip(x_all, d_w0_m)]:
                data_w0_m.append(el)

        def get_heatmap(data, lim, title, kdim='x', vdim='Iteration', height=200):
            return hv.HeatMap(data, kdims=[kdim, vdim]).opts(
                opts.HeatMap(tools=['hover'],
                             colorbar=True,
                             width=350,
                             height=height,
                             invert_yaxis=True,
                             ylabel='Iteration',
                             title=title,
                             clim=(-lim, lim),
                             cmap='RdBu'))

        # plots for delta bed_h and w0_m
        delta_bed_h_plot = get_heatmap(data_bed_h,
                                       d_bed_h_lim,
                                       'Delta bed_h',
                                       kdim='ice_mask_x',
                                       height=200)

        delta_w0_m_plot = get_heatmap(data_w0_m,
                                      d_w0_m_lim,
                                      'Delta w0_m',
                                      kdim='ice_mask_x',
                                      height=200)

        parameter_indices = ds.attrs['parameter_indices']
        # plot for height shift spinup if there
        if 'height_shift_spinup' in parameter_indices.keys():
            height_shift_data = []
            for i, unknown_p in enumerate(ds.unknown_parameters.values):
                height_shift_data.append(
                    (i, unknown_p[parameter_indices['height_shift_spinup']]))
            height_shift_spinup_plot = hv.Curve(height_shift_data,
                                                kdims='Iterations',
                                                vdims='shift (m)',
                                                ).opts(
                opts.Curve(title='spinup height shift',
                           tools=['hover'],
                           height=200)
            )
        else:
            height_shift_spinup_plot = None

        # get gradients
        data_grad_bed_h = None
        data_grad_area_bed_h = None
        data_grad_w0_m = None
        data_grad_surface_h = None
        data_grad_height_shift_spinup = None

        if 'bed_h' in parameter_indices.keys():
            data_grad_bed_h = []
            grad_bed_h_lim = 0
        if 'area_bed_h' in parameter_indices.keys():
            data_grad_area_bed_h = []
            grad_area_bed_h_lim = 0
        if 'w0_m' in parameter_indices.keys():
            data_grad_w0_m = []
            grad_w0_m_lim = 0
        if 'surface_h' in parameter_indices.keys():
            data_grad_surface_h = []
            grad_surface_h_lim = 0
        if 'height_shift_spinup' in parameter_indices.keys():
            data_grad_height_shift_spinup = []

        for i, grad in enumerate(ds.grads.values):
            x_ice_mask = ds.coords['x'][ds.ice_mask].values
            x_all = ds.coords['x'].values

            # bed_h
            if 'bed_h' in parameter_indices.keys():
                grad_bed_h = grad[parameter_indices['bed_h']]
                grad_bed_h_lim = np.max([grad_bed_h_lim, np.max(np.abs(grad_bed_h))])
                for el in [(x, i, v) for x, v in zip(x_ice_mask, grad_bed_h)]:
                    data_grad_bed_h.append(el)
            if 'area_bed_h' in parameter_indices.keys():
                grad_area_bed_h = grad[parameter_indices['area_bed_h']]
                grad_area_bed_h_lim = np.max([grad_area_bed_h_lim, np.max(np.abs(
                    grad_area_bed_h))])
                for el in [(x, i, v) for x, v in zip(x_ice_mask, grad_area_bed_h)]:
                    data_grad_area_bed_h.append(el)
            if 'w0_m' in parameter_indices.keys():
                grad_w0_m = grad[parameter_indices['w0_m']]
                grad_w0_m_lim = np.max([grad_w0_m_lim, np.max(np.abs(grad_w0_m))])
                for el in [(x, i, v) for x, v in zip(x_ice_mask, grad_w0_m)]:
                    data_grad_w0_m.append(el)
            if 'surface_h' in parameter_indices.keys():
                grad_surface_h = grad[parameter_indices['surface_h']]
                grad_surface_h_lim = np.max([grad_surface_h_lim, np.max(np.abs(grad_surface_h))])
                for el in [(x, i, v) for x, v in zip(x_all, grad_surface_h)]:
                    data_grad_surface_h.append(el)
            if 'height_shift_spinup' in parameter_indices.keys():
                data_grad_height_shift_spinup.append(
                    (i, grad[parameter_indices['height_shift_spinup']]))

        grad_plots = None
        if 'bed_h' in parameter_indices.keys():
            grad_plots = pn.Column(get_heatmap(data_grad_bed_h,
                                               grad_bed_h_lim,
                                               'Grad bed_h',
                                               kdim='ice_mask_x',
                                               height=200),
                                   sizing_mode='stretch_width')
        elif 'area_bed_h' in parameter_indices.keys():
            grad_plots = pn.Column(get_heatmap(data_grad_area_bed_h,
                                               grad_area_bed_h_lim,
                                               'Grad area_bed_h',
                                               kdim='ice_mask_x',
                                               height=200),
                                   sizing_mode='stretch_width')

        if 'w0_m' in parameter_indices.keys():
            grad_plots.append(get_heatmap(data_grad_w0_m,
                                          grad_w0_m_lim,
                                          'Grad w0_m',
                                          kdim='ice_mask_x',
                                          height=200))

        if 'surface_h' in parameter_indices.keys():
            grad_plots.append(get_heatmap(data_grad_surface_h,
                                          grad_surface_h_lim,
                                          'Grad surface_h',
                                          kdim='total_distance_x',
                                          height=200))

        if 'height_shift_spinup' in parameter_indices.keys():
            grad_plots.append(hv.Curve(data_grad_height_shift_spinup,
                                       kdims='Iterations',
                                       vdims='gradient',
                                       ).opts(
                opts.Curve(title='spinup height shift gradient',
                           tools=['hover'],
                           height=200)
            ))

        # convert c_terms
        c_terms_conv = {}
        hover_height = 0
        for term in ds.c_terms_description.values:
            # term = ds.c_terms_description.values[0]
            for var in term.keys():
                var_use = var.replace(':', '_')
                var_use = var_use.replace('-', '')
                if type(term[var]) == dict:
                    yr = list(term[var].keys())[0]
                    yr_use = yr.replace('-', '_')
                    if var_use + '_' + yr_use not in c_terms_conv.keys():
                        c_terms_conv[var_use + '_' + yr_use] = []
                    c_terms_conv[var_use + '_' + yr_use].append(term[var][yr])
                    hover_height = np.max([hover_height, np.max(term[var][yr])])
                else:
                    if var_use not in c_terms_conv.keys():
                        c_terms_conv[var_use] = []
                    c_terms_conv[var_use].append(term[var])
                    hover_height = np.max([hover_height, np.max(term[var])])

        c_term_area = []
        for one_c_term in c_terms_conv.keys():
            c_term_area.append(hv.Area((ds.coords['iteration'].values,
                                        c_terms_conv[one_c_term]),
                                       kdims='Iterations',
                                       vdims='c_terms',
                                       label=one_c_term))
        overlay_c_terms = hv.Overlay(c_term_area)
        stack_c_terms = hv.Area.stack(overlay_c_terms)

        df_c_terms = pd.DataFrame(c_terms_conv)
        df_c_terms['Iteration'] = ds.coords['iteration'].values
        df_c_terms['hover_height'] = np.repeat(hover_height / 2,
                                               len(ds.coords['iteration'].values))

        tooltips_c_terms = [('Iteration', '@{Iteration}')]
        tooltips_c_terms += [(key, '@{' + key + '}{%0.4f}') for key in c_terms_conv.keys()]
        hover_c_terms = HoverTool(tooltips=tooltips_c_terms,
                                  formatters=dict([('@{' + key + '}', 'printf') for key in
                                                   c_terms_conv.keys()]),
                                  mode='vline')
        vdims_curve_c_terms = ['hover_height']
        for key in c_terms_conv.keys():
            vdims_curve_c_terms.append(key)
        curve_c_terms = hv.Curve(df_c_terms,
                                 kdims='Iteration',
                                 vdims=vdims_curve_c_terms).opts(tools=[hover_c_terms],
                                                                 line_alpha=0)

        c_terms_plot = (stack_c_terms * curve_c_terms).opts(width=500,
                                                            height=200,
                                                            legend_position='left',
                                                            title='Cost Terms')

        # calculate differences of surface height at start, rgi and end
        for gdir in gdirs:
            if gdir.rgi_id == rgi_id:
                fl_ref_rgi = gdir.read_pickle('model_flowlines',
                                              filesuffix='_combine_true_init')[0]
                fl_ref_start = gdir.read_pickle('model_flowlines',
                                                filesuffix='_spinup')[0]
                fl_ref_end = gdir.read_pickle('model_flowlines',
                                              filesuffix='_combine_true_end')[0]

        def get_performance_sfc_h_array(fct, data, ref_val):
            return [np.around(fct(val, ref_val), decimals=2) for val in data]

        def get_sfc_h_table(data, ref_val, title):
            df = pd.DataFrame({'RMSE': get_performance_sfc_h_array(RMSE, data, ref_val),
                               'BIAS': get_performance_sfc_h_array(BIAS, data, ref_val),
                               'DIFF': get_performance_sfc_h_array(DIFF, data, ref_val),
                               'AERR': get_performance_sfc_h_array(AERR, data, ref_val),
                               })
            return pn.Column(pn.pane.Markdown('Statistics ' + title),
                             pn.widgets.Tabulator(df,
                                                  titles={'index': 'I'},
                                                  height=200),
                             sizing_mode='stretch_width')

        # sfc_h_end
        d_sfc_h_end_lim = 0.
        data_sfc_h_end = []
        table_data_sfc_h_end = []
        for i, fl in enumerate(ds.flowlines.values):
            x_all = ds.coords['x'].values
            d_sfc_h_end = (fl.surface_h -
                           fl_ref_end.surface_h)
            d_sfc_h_end_lim = np.max([d_sfc_h_end_lim, np.max(np.abs(d_sfc_h_end))])
            table_data_sfc_h_end.append(fl.surface_h)
            for el in [(x, i, v) for x, v in zip(x_all, d_sfc_h_end)]:
                data_sfc_h_end.append(el)
        delta_sfc_h_end_plot = get_heatmap(data_sfc_h_end,
                                           d_sfc_h_end_lim,
                                           'Delta sfc_h_end',
                                           kdim='total_distance_x',
                                           height=150)
        delta_sfc_h_end_table = get_sfc_h_table(table_data_sfc_h_end,
                                                fl_ref_end.surface_h,
                                                'sfc_h_end')

        # sfc_h_rgi
        if 'fl_surface_h:m' in ds.observations_mdl.values[0].keys():
            d_sfc_h_rgi_lim = 0.
            data_sfc_h_rgi = []
            table_data_sfc_h_rgi = []
            for i, obs in enumerate(ds.observations_mdl.values):
                x_all = ds.coords['x'].values
                d_sfc_h_rgi = (list(obs['fl_surface_h:m'].values())[0] -
                               fl_ref_rgi.surface_h)
                d_sfc_h_rgi_lim = np.max([d_sfc_h_rgi_lim, np.max(np.abs(d_sfc_h_rgi))])
                table_data_sfc_h_rgi.append(list(obs['fl_surface_h:m'].values()
                                                 )[0])
                for el in [(x, i, v) for x, v in zip(x_all, d_sfc_h_rgi)]:
                    data_sfc_h_rgi.append(el)
            delta_sfc_h_rgi_plot = get_heatmap(data_sfc_h_rgi,
                                               d_sfc_h_rgi_lim,
                                               'Delta sfc_h_rgi',
                                               kdim='total_distance_x',
                                               height=150)
            delta_sfc_h_rgi_table = get_sfc_h_table(table_data_sfc_h_rgi,
                                                    fl_ref_rgi.surface_h,
                                                    'sfc_h_rgi')
        else:
            delta_sfc_h_rgi_plot = None
            delta_sfc_h_rgi_table = None

        # sfc_h_start
        d_sfc_h_start_lim = 0.
        data_sfc_h_start = []
        table_data_sfc_h_start = []
        for i, tmp_sfc_h in enumerate(ds.sfc_h_start.values):
            x_all = ds.coords['x'].values
            d_sfc_h_start = (tmp_sfc_h -
                             fl_ref_start.surface_h)
            d_sfc_h_start_lim = np.max([d_sfc_h_start_lim, np.max(np.abs(d_sfc_h_start))])
            table_data_sfc_h_start.append(tmp_sfc_h)
            for el in [(x, i, v) for x, v in zip(x_all, d_sfc_h_start)]:
                data_sfc_h_start.append(el)
        delta_sfc_h_start_plot = get_heatmap(data_sfc_h_start,
                                             d_sfc_h_start_lim,
                                             'Delta sfc_h_start',
                                             kdim='total_distance_x',
                                             height=150)
        delta_sfc_h_start_table = get_sfc_h_table(table_data_sfc_h_start,
                                                  fl_ref_start.surface_h,
                                                  'sfc_h_start')

        # create Table with performance measures (bed_h, w0_m, sfc_h_start, sfc_h_end, sfc_h_rgi,
        # fct_calls, time, device)
        def get_performance_array(fct, attr):
            return [np.around(fct(val, getattr(fl_ref, attr)[ds.ice_mask]),
                              decimals=2) for val in
                    [getattr(fl.values.item(), attr)[ds.ice_mask]
                     for fl in ds.flowlines]]

        def get_performance_table(attr):
            df = pd.DataFrame({'RMSE': get_performance_array(RMSE, attr),
                               'BIAS': get_performance_array(BIAS, attr),
                               'DIFF': get_performance_array(DIFF, attr),
                               'AERR': get_performance_array(AERR, attr),
                               })
            return pn.Column(pn.pane.Markdown('Statistics ' + attr),
                             pn.widgets.Tabulator(df,
                                                  titles={'index': 'I'},
                                                  height=200),
                             sizing_mode='stretch_width')

        def get_minimise_performance_table():
            df = pd.DataFrame({'forward runs': ds.fct_calls.values,
                               'computing time': ds.time_needed.values,
                               'device': np.repeat(ds.attrs['device'],
                                                   len(ds.time_needed.values))
                               })
            return pn.widgets.Tabulator(df)

        performance_tables = \
            pn.Column(get_performance_table('bed_h'),
                      get_performance_table('_w0_m'),
                      get_minimise_performance_table(),
                      sizing_mode='stretch_width')

        # create plot for exploration of geometry
        # thickness at end time
        data_thick_end = []
        thick_end_lim = 0.
        for i, fl in enumerate(ds.flowlines.values):
            x_all = ds.coords['x'].values
            thick_end = fl.thick
            thick_end_lim = np.max([thick_end_lim, np.max(np.abs(thick_end))])
            for el in [(x, i, v) for x, v in zip(x_all, thick_end)]:
                data_thick_end.append(el)
        thick_end_plot = get_heatmap(data_thick_end,
                                     thick_end_lim,
                                     'Ice thickness at end time',
                                     kdim='total_distance_x',
                                     height=150)
        thick_end_true = fl_ref_end.thick
        thick_end_true_lim = np.max(np.abs(thick_end_true))
        x_all = ds.coords['x'].values
        data_thick_end_true = []
        for el in [(x, 0, v) for x, v in zip(x_all, thick_end_true)]:
            data_thick_end_true.append(el)
        thick_end_true_plot = get_heatmap(data_thick_end_true,
                                          thick_end_true_lim,
                                          'Ice thickness at end time TRUE',
                                          kdim='total_distance_x',
                                          vdim='true',
                                          height=100)

        # surface widths
        def get_width_curve(x, width, label, color, height=150):
            return (hv.Curve((x, width / 2),
                             kdims='total_distance_x',
                             vdims='widths',
                             label=label,
                             ) *
                    hv.Curve((x, - width / 2),
                             kdims='total_distance_x',
                             vdims='widths',
                             label=label,
                             )
                    ).opts(opts.Curve(color=color,
                                      tools=['hover'],
                                      height=height))

        x_all = ds.coords['x'].values
        surface_widths_rgi_true_plot = get_width_curve(x_all,
                                                       fl_ref_rgi.widths_m,
                                                       'RGI',
                                                       'blue')
        surface_widths_start_true_plot = get_width_curve(x_all,
                                                         fl_ref_start.widths_m,
                                                         'Start',
                                                         'red')
        surface_widths_end_true_plot = get_width_curve(x_all,
                                                       fl_ref_end.widths_m,
                                                       'End',
                                                       'gray')
        widths_plot = (surface_widths_start_true_plot *
                       surface_widths_rgi_true_plot *
                       surface_widths_end_true_plot
                       ).opts(title='Surface widths',
                              # legend_position='right',
                              show_legend=False
                              )

        # Surface_h with bed_h
        x_all = ds.coords['x'].values

        def get_curve(x, y, label, color, height=200):
            return hv.Curve((x, y),
                            kdims='total_distance_x',
                            vdims='heights',
                            label=label
                            ).opts(opts.Curve(color=color,
                                              tools=['hover'],
                                              height=height))

        surface_height_start_true_plot = get_curve(x_all,
                                                   fl_ref_start.surface_h,
                                                   'Start',
                                                   'red')
        surface_height_rgi_true_plot = get_curve(x_all,
                                                 fl_ref_rgi.surface_h,
                                                 'RGI',
                                                 'blue')
        surface_height_end_true_plot = get_curve(x_all,
                                                 fl_ref_end.surface_h,
                                                 'End',
                                                 'gray')
        bed_height_true_plot = get_curve(x_all,
                                         fl_ref_rgi.bed_h,
                                         'bed_h',
                                         'black')

        surface_height_plot = (bed_height_true_plot *
                               surface_height_start_true_plot *
                               surface_height_rgi_true_plot *
                               surface_height_end_true_plot
                               ).opts(title='Surface heights',
                                      legend_position='bottom',
                                      legend_cols=2
                                      )

        return pn.Column('## ' + current_file,
                         pn.Row(
                             pn.Column(
                                 pn.Row(
                                     pn.Column(delta_bed_h_plot,
                                               delta_w0_m_plot,
                                               height_shift_spinup_plot,
                                               sizing_mode='stretch_width'
                                               ),
                                     grad_plots
                                 ),
                                 pn.Row(c_terms_plot,
                                        sizing_mode='stretch_width'),
                             ),
                             pn.Column(delta_sfc_h_start_plot,
                                       delta_sfc_h_rgi_plot,
                                       delta_sfc_h_end_plot,
                                       delta_sfc_h_start_table,
                                       delta_sfc_h_rgi_table,
                                       delta_sfc_h_end_table,
                                       sizing_mode='stretch_width'),
                             pn.Column(thick_end_plot,
                                       thick_end_true_plot,
                                       widths_plot,
                                       surface_height_plot,
                                       performance_tables,
                                       sizing_mode='stretch_width'),
                         ),
                         sizing_mode='stretch_width')

    # put together and link menu with plots
    current_file_first = list(compress(all_files,
                                       [glacier_select.value in file and
                                        experiment_select.value in file
                                        for file in all_files]))
    for opt_select in options_selects:
        current_file_first = list(compress(current_file_first,
                                           ['_' + opt_select.value in file
                                            for file in current_file_first]))
    current_file_first = current_file_first[0]
    with open(input_folder + current_file_first, 'rb') as handle:
        try:
            open_files[current_file_first] = pickle.load(handle)
        except:
            open_files[current_file_first] = CpuUnpickler(handle).load()
        # pickle.load(handle)

    figure = get_individual_plot(current_file_first)

    def change_figure(event, open_files):
        # here get the right filename for the current selection
        current_file = list(compress(all_files,
                                     [glacier_select.value + '_' +
                                      experiment_select.value in file
                                      for file in all_files]))

        # select all other options
        for opt_select in options_selects:
            current_file = list(compress(current_file,
                                         ['_' + opt_select.value in file
                                          for file in current_file]))

        if len(current_file) == 0:
            button.name = 'no file found'
            button.button_type = 'danger'
        else:
            button.name = 'Select new one'
            button.button_type = 'primary'
            current_file = current_file[0]

            # if the first time open it
            if current_file not in open_files.keys():
                with open(input_folder + current_file, 'rb') as handle:
                    try:
                        open_files[current_file] = pickle.load(handle)
                    except:
                        open_files[current_file] = CpuUnpickler(handle).load()
                    # pickle.load(handle,)

            figure.objects = [get_individual_plot(current_file)]

    button.on_click(partial(change_figure, open_files=open_files))

    individual_app = pn.Row(pn.Column(menu,
                                      get_description_accordion()),
                            figure)

    return individual_app, open_files
