import pickle
from functools import partial
from itertools import compress
import os
import xarray as xr
import pandas as pd
import numpy as np
from bokeh.models import HoverTool
import panel as pn
import holoviews as hv
from holoviews import opts

pn.extension('tabulator')
hv.extension('bokeh')

import hvplot.xarray

from oggm import cfg, utils, workflow, tasks, graphics
cfg.initialize(logging_level='WARNING')

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
        if opt[:-1] not in option_used:
            main_options.append(opt)
            option_used.append(opt[:-1])
    for main_opt in main_options:
        name_tmp = main_opt[:-1]
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
            d_bed_h = (fl.bed_h.detach().numpy() - fl_ref.bed_h)[ds.ice_mask]
            d_bed_h_lim = np.max([d_bed_h_lim, np.max(np.abs(d_bed_h))])
            for el in [(x, i, v) for x, v in zip(x_all, d_bed_h)]:
                data_bed_h.append(el)

            # w0_m
            d_w0_m = (fl._w0_m.detach().numpy() - fl_ref._w0_m)[ds.ice_mask]
            d_w0_m_lim = np.max([d_w0_m_lim, np.max(np.abs(d_w0_m))])
            for el in [(x, i, v) for x, v in zip(x_all, d_w0_m)]:
                data_w0_m.append(el)

        def get_heatmap(data, lim, title, kdim='x', height=200):
            return hv.HeatMap(data, kdims=[kdim, 'Iteration']).opts(
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

        # get gradients
        parameter_indices = ds.attrs['parameter_indices']

        data_grad_bed_h = None
        data_grad_w0_m = None

        if 'bed_h' in parameter_indices.keys():
            data_grad_bed_h = []
            grad_bed_h_lim = 0
        if 'w0_m' in parameter_indices.keys():
            data_grad_w0_m = []
            grad_w0_m_lim = 0

        for i, grad in enumerate(ds.grads.values):
            x_all = ds.coords['x'][ds.ice_mask].values

            # bed_h
            if 'bed_h' in parameter_indices.keys():
                grad_bed_h = grad[parameter_indices['bed_h']]
                grad_bed_h_lim = np.max([grad_bed_h_lim, np.max(np.abs(grad_bed_h))])
                for el in [(x, i, v) for x, v in zip(x_all, grad_bed_h)]:
                    data_grad_bed_h.append(el)
            if 'w0_m' in parameter_indices.keys():
                grad_w0_m = grad[parameter_indices['w0_m']]
                grad_w0_m_lim = np.max([grad_w0_m_lim, np.max(np.abs(grad_w0_m))])
                for el in [(x, i, v) for x, v in zip(x_all, grad_w0_m)]:
                    data_grad_w0_m.append(el)

        grad_plots = None
        if 'bed_h' in parameter_indices.keys():
            grad_plots = pn.Column(get_heatmap(data_grad_bed_h,
                                               grad_bed_h_lim,
                                               'Grad bed_h',
                                               kdim='ice_mask_x',
                                               height=200),
                                   sizing_mode='stretch_width')

        if 'w0_m' in parameter_indices.keys():
            grad_plots.append(get_heatmap(data_grad_w0_m,
                                          grad_w0_m_lim,
                                          'Grad w0_m',
                                          kdim='ice_mask_x',
                                          height=200))

        # convert c_terms
        c_terms_conv = {}
        hover_height = 0
        for term in ds.c_terms_description.values:
            # term = ds.c_terms_description.values[0]
            for var in term.keys():
                var_use = var.replace(':', '_')
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

        # sfc_h_end
        d_sfc_h_end_lim = 0.
        data_sfc_h_end = []
        for i, fl in enumerate(ds.flowlines.values):
            x_all = ds.coords['x'].values
            d_sfc_h_end = (fl.surface_h.detach().numpy() -
                           fl_ref_end.surface_h)
            d_sfc_h_end_lim = np.max([d_sfc_h_end_lim, np.max(np.abs(d_sfc_h_end))])
            for el in [(x, i, v) for x, v in zip(x_all, d_sfc_h_end)]:
                data_sfc_h_end.append(el)
        delta_sfc_h_end_plot = get_heatmap(data_sfc_h_end,
                                           d_sfc_h_end_lim,
                                           'Delta sfc_h_end',
                                           kdim='total_distance_x',
                                           height=150)

        # sfc_h_rgi
        d_sfc_h_rgi_lim = 0.
        data_sfc_h_rgi = []
        for i, obs in enumerate(ds.observations_mdl.values):
            x_all = ds.coords['x'].values
            d_sfc_h_rgi = (list(obs['fl_surface_h:m'].values())[0].detach().numpy() -
                           fl_ref_rgi.surface_h)
            d_sfc_h_rgi_lim = np.max([d_sfc_h_rgi_lim, np.max(np.abs(d_sfc_h_rgi))])
            for el in [(x, i, v) for x, v in zip(x_all, d_sfc_h_rgi)]:
                data_sfc_h_rgi.append(el)
        delta_sfc_h_rgi_plot = get_heatmap(data_sfc_h_rgi,
                                           d_sfc_h_rgi_lim,
                                           'Delta sfc_h_rgi',
                                           kdim='total_distance_x',
                                           height=150)

        # sfc_h_start
        d_sfc_h_start_lim = 0.
        data_sfc_h_start = []
        for i, tmp_sfc_h in enumerate(ds.sfc_h_start.values):
            x_all = ds.coords['x'].values
            d_sfc_h_start = (tmp_sfc_h -
                             fl_ref_start.surface_h)
            d_sfc_h_start_lim = np.max([d_sfc_h_start_lim, np.max(np.abs(d_sfc_h_start))])
            for el in [(x, i, v) for x, v in zip(x_all, d_sfc_h_start)]:
                data_sfc_h_start.append(el)
        delta_sfc_h_start_plot = get_heatmap(data_sfc_h_start,
                                             d_sfc_h_start_lim,
                                             'Delta sfc_h_start',
                                             kdim='total_distance_x',
                                             height=150)

        # create Table with performance measures (bed_h, w0_m, sfc_h_start, sfc_h_end, sfc_h_rgi,
        # fct_calls, time, device)
        def get_performance_array(fct, attr):
            return [fct(val, getattr(fl_ref, attr)[ds.ice_mask]) for val in
                    [getattr(fl.values.item(), attr).detach().numpy()[ds.ice_mask]
                     for fl in ds.flowlines]]

        def get_performance_table(attr):
            df = pd.DataFrame({'RMSE(' + attr + ')': get_performance_array(RMSE, attr),
                               'BIAS(' + attr + ')': get_performance_array(BIAS, attr),
                               'DIFF(' + attr + ')': get_performance_array(DIFF, attr),
                               'AERR(' + attr + ')': get_performance_array(AERR, attr),
                               })
            return pn.widgets.Tabulator(df)

        def get_minimise_performance_table():
            df = pd.DataFrame({'forward runs': ds.fct_calls.values,
                               'computing time': ds.time_needed.values,
                               'device': np.repeat(ds.attrs['device'],
                                                   len(ds.time_needed.values))
                               })
            return pn.widgets.Tabulator(df)

        performance_accordion = \
            pn.Accordion(('performance bed_h', get_performance_table('bed_h')),
                         ('performance w0_m', get_performance_table('_w0_m')),
                         ('minimise performance', get_minimise_performance_table()),
                         sizing_mode='stretch_width')

        return pn.Column('## ' + current_file,
                         pn.Row(
                             pn.Column(
                                 pn.Row(
                                     pn.Column(delta_bed_h_plot,
                                               delta_w0_m_plot,
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
                                       performance_accordion,
                                       sizing_mode='stretch_width')
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
        open_files[current_file_first] = pickle.load(handle)

    figure = get_individual_plot(current_file_first)

    def change_figure(event, open_files):
        # here get the right filename for the current selection
        current_file = list(compress(all_files,
                                     [glacier_select.value in file and
                                      experiment_select.value in file
                                      for file in all_files]))
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
                open_files[current_file] = pickle.load(handle)

        figure.objects = [get_individual_plot(current_file)]

    button.on_click(partial(change_figure, open_files=open_files))

    individual_app = pn.Row(pn.Column(menu,
                                      get_description_accordion()),
                            figure)

    return individual_app, open_files


