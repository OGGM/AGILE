import numpy as np
from scipy.optimize import minimize
import holoviews as hv
from holoviews import opts
from combine.core.cost_function import creat_cost_fct, creat_spinup_cost_fct
from combine.core.arithmetics import magnitude
from oggm.core.massbalance import LinearMassBalance
from oggm.core.flowline import FluxBasedModel as oggm_FluxModel
from oggm.core.flowline import RectangularBedFlowline
from oggm.core.flowline import ParabolicBedFlowline
from oggm.core.flowline import TrapezoidalBedFlowline
from oggm import GlacierDirectory
from oggm.tasks import define_glacier_region
import geopandas as gpd
from oggm.utils import get_demo_file
from oggm.core import climate, inversion, centerlines
from oggm import utils
import shapely.geometry as shpg
import copy
from oggm import cfg
# hv.extension('bokeh')
# cfg.initialize()


def define_geometry(used_bed_h_geometry='linear',
                    used_along_glacier_geometry='linear',
                    bed_geometry='parabolic'):
    geometry = {}
    # number of steps from bottem to top of glacier
    geometry['nx'] = 100

    # model grid spacing in m
    geometry['map_dx'] = 100

    # distance along glacier (x-axis of glacier profil plot) in km
    geometry['distance_along_glacier'] = (np.linspace(0, geometry['nx'],
                                                      geometry['nx']) *
                                          geometry['map_dx'] * 1e-3)

    if used_bed_h_geometry == 'linear':
        # glacier top height
        geometry['top_height'] = 4000

        # glacier bottom height
        geometry['bottom_height'] = 0
        # define linear glacier bed with zero ice thickness
        geometry['bed_h'] = np.linspace(geometry['top_height'],
                                        geometry['bottom_height'],
                                        geometry['nx'])
    elif used_bed_h_geometry == 'cliff':
        # glacier top height
        geometry['top_height'] = 4000

        # glacier bottom height
        geometry['bottom_height'] = 1000
        # define extend of cliff
        cliff_top = 3200
        cliff_bottom = 3050

        geometry['bed_h'] = np.concatenate(
            (np.linspace(geometry['top_height'],
                         cliff_top,
                         geometry['nx'] / 4),
             np.linspace(cliff_bottom,
                         geometry['bottom_height'],
                         geometry['nx'] * 3 / 4)))
    elif used_bed_h_geometry == 'random':
        # glacier top height
        geometry['top_height'] = 4000

        # glacier bottom height
        geometry['bottom_height'] = 1500
        # set numpy seed to always get the same random bed
        np.random.seed(0)

        # get the differences from linear bed
        offsets = np.random.normal(scale=30., size=geometry['nx'])

        # define random glacier bed
        geometry['bed_h'] = (np.linspace(geometry['top_height'],
                                         geometry['bottom_height'],
                                         geometry['nx']) +
                             offsets)
    else:
        raise ValueError('Unknown bed height geometry!')

    if used_along_glacier_geometry == 'linear':
        if bed_geometry == 'rectangular':
            geometry['widths'] = np.zeros(geometry['nx']) + 1.
        elif bed_geometry == 'parabolic':
            geometry['bed_shapes'] = np.zeros(geometry['nx']) + 1.
        elif bed_geometry == 'trapezoid':
            geometry['w0'] = np.zeros(geometry['nx']) + 1.
            geometry['lambdas'] = np.zeros(geometry['nx']) + 1.
        else:
            raise ValueError('Unkonwn bed shape!')
    elif used_along_glacier_geometry == 'random':
        np.random.seed(0)
        random_shape = np.random.normal(loc=1.,
                                        scale=0.1,
                                        size=geometry['nx'])

        if bed_geometry == 'rectangular':
            geometry['widths'] = random_shape
        elif bed_geometry == 'parabolic':
            geometry['bed_shapes'] = random_shape
        elif bed_geometry == 'trapezoid':
            geometry['w0'] = random_shape
            geometry['lambdas'] = np.zeros(geometry['nx']) + 1.
        else:
            raise ValueError('Unkonwn bed shape!')
    else:
        raise ValueError('Unknown along glacier geometry!')

    return geometry


def define_mb_model(mb_type='linear',
                    mb_opts={'ELA': np.array([3000.]),
                             'grad': np.array([4.])}):
    if mb_type == 'linear':
        # define linear mass balance
        ELA = mb_opts['ELA']
        mb_gradient = mb_opts['grad']
        if len(ELA) == 1:
            mb_model = LinearMassBalance(ELA, grad=mb_gradient)
        else:
            mb_model = []
            for i in np.arange(len(ELA)):
                mb_model.append(LinearMassBalance(ELA[i],
                                                  grad=mb_gradient[i]))
    else:
        raise ValueError('Unknown mass balance type!')

    return mb_model


def create_measurements(geometry,
                        mb_model,
                        bed_geometry='rectangular',
                        glacier_state='equilibrium',
                        add_noise=False):
    measurements = {}

    # Create glacier and let it run
    if bed_geometry == 'rectangular':
        oggm_fl = RectangularBedFlowline(surface_h=geometry['bed_h'],
                                         bed_h=geometry['bed_h'],
                                         widths=geometry['widths'],
                                         map_dx=geometry['map_dx'])
    elif bed_geometry == 'parabolic':
        oggm_fl = ParabolicBedFlowline(surface_h=geometry['bed_h'],
                                       bed_h=geometry['bed_h'],
                                       bed_shape=geometry['bed_shapes'],
                                       map_dx=geometry['map_dx'])
    elif bed_geometry == 'trapezoid':
        oggm_fl = TrapezoidalBedFlowline(surface_h=geometry['bed_h'],
                                         bed_h=geometry['bed_h'],
                                         widths=geometry['w0'],
                                         lambdas=geometry['lambdas'],
                                         map_dx=geometry['map_dx'],
                                         )
    else:
        raise ValueError('Unkown bed shape!')

    if glacier_state == 'equilibrium':
        ref_model = oggm_FluxModel(oggm_fl, mb_model=mb_model, y0=0.)
        ref_model.run_until_equilibrium()

    elif glacier_state == 'advancing':
        ref_model = oggm_FluxModel(oggm_fl, mb_model=mb_model, y0=0.)
        ref_model.run_until_equilibrium()
        eq_years = ref_model.yr
        ref_model = oggm_FluxModel(oggm_fl, mb_model=mb_model, y0=0.)
        ref_model.run_until(int(eq_years/2))

    elif (glacier_state == 'retreating' or
          glacier_state == 'retreating with unknow spinup'):
        assert len(mb_model) >= 2,\
            'Need at least two mass balance models for retreating!'
        years_to_retreat = None
        for i in range(2):
            start_model = oggm_FluxModel(oggm_fl,
                                         mb_model=mb_model[0],
                                         y0=0.)
            start_model.run_until_equilibrium()
            retreat_model = oggm_FluxModel(start_model.fls[-1],
                                           mb_model=mb_model[1],
                                           y0=0.)
            if years_to_retreat is None:
                retreat_model.run_until_equilibrium()
                years_to_retreat = int(retreat_model.yr / 2)
            else:
                retreat_model.run_until(years_to_retreat)
                break
        ref_model = retreat_model

    else:
        raise ValueError('Unknown glacier state!')

    # to add some noise to the measurements
    sfc_h_noise = np.zeros(geometry['nx'])
    widths_noise = np.zeros(geometry['nx'])

    if add_noise:
        raise NotImplementedError('measurement noise not ready yet!')

    # safe model
    measurements['model'] = ref_model

    # get 'measurements' of glacier
    measurements['sfc_h'] = ref_model.fls[0].surface_h + sfc_h_noise
    measurements['widths'] = ref_model.fls[0].widths_m + widths_noise
    measurements['yrs_to_run'] = ref_model.yr

    # find ice and ice free points
    ice_mask = np.where(ref_model.fls[0].thick > 10e-2, True, False)
    measurements['ice_mask'] = ice_mask
    measurements['bed_known'] = ref_model.fls[0].bed_h[~ice_mask]
    measurements['bed_unknown'] = ref_model.fls[0].bed_h[ice_mask]
    measurements['bed_all'] = ref_model.fls[0].bed_h
    if bed_geometry == 'parabolic':
        measurements['shape_known'] = ref_model.fls[0].bed_shape[~ice_mask]
        measurements['shape_unknown'] = ref_model.fls[0].bed_shape[ice_mask]
        measurements['shape_all'] = ref_model.fls[0].bed_shape
    else:
        measurements['shape_known'] = None
        measurements['shape_unknown'] = None
        measurements['shape_all'] = None

    if (glacier_state == 'equilibrium') or (glacier_state == 'advancing'):
        measurements['spinup_sfc'] = np.zeros(len(measurements['bed_all']))
    elif glacier_state == 'retreating':
        measurements['spinup_sfc'] = start_model.fls[0].surface_h
    elif glacier_state == 'retreating with unknow spinup':
        measurements['spinup_volume'] = start_model.fls[0].volume_m3
        measurements['spinup_bin_area'] = start_model.fls[0].bin_area_m2
        measurements['ref_spinup_sfc'] = start_model.fls[0].surface_h
    else:
        raise ValueError('Unknown glacier state!')

    return measurements


def get_first_guess(measurements,
                    method='oggm',
                    bed_geometry='rectangular',
                    const={'bed_h': None, 'shape': None},
                    opti_parameter='bed_h',
                    lambdas=None):
    first_guess = {'bed_h': None, 'shape': None}
    get_bed_h = get_shape = False
    if opti_parameter == 'bed_h':
        get_bed_h = True
    elif opti_parameter == 'shape':
        get_shape = True
    elif opti_parameter == 'bed_h and shape':
        get_bed_h = get_shape = True
    else:
        raise ValueError('Unknown regularisation parameter!')

    if method == 'plus const':
        if get_bed_h:
            bed_unknown = measurements['bed_unknown']
            if const['bed_h'] is None:
                const['bed_h'] = np.repeat(np.mean(bed_unknown) / 10,
                                           len(bed_unknown))
            elif len(const['bed_h']) == 1:
                const['bed_h'] = np.repeat(const['bed_h'],
                                           len(bed_unknown))

            first_guess['bed_h'] = bed_unknown + const['bed_h']
        else:
            first_guess['bed_h'] = bed_unknown

        if get_shape:
            shape_unknown = measurements['shape_unknown']
            if const['shape'] is None:
                const['shape'] = np.repeat(np.mean(shape_unknown) / 10,
                                           len(shape_unknown))
            elif len(const['shape']) == 1:
                const['shape'] = np.repeat(const['shape'],
                                           len(shape_unknown))

            first_guess['shape'] = shape_unknown + const['shape']
        else:
            first_guess['shape'] = shape_unknown

        return first_guess

    elif method == 'oggm':
        model = measurements['model']
        # Init
        cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')

        hef_file = get_demo_file('Hintereisferner_RGI5.shp')
        entity = gpd.read_file(hef_file).iloc[0]

        tmpdir = utils.gettempdir(dirname='OGGM_Inversion', reset=True)
        gdir = GlacierDirectory(entity, base_dir=tmpdir, reset=True)
        define_glacier_region(gdir)

        fl = model.fls[0]
        pg = np.where(fl.thick > 10e-2)
        ice_index = pg[0]
        line = shpg.LineString([fl.line.coords[int(p)] for p in ice_index])
        sh = fl.surface_h[ice_index]
        # *2 because of gdir.grid.dx
        flo = centerlines.Centerline(line, dx=fl.dx * 2,
                                     surface_h=sh)
        flo.widths = fl.widths[ice_index] * 2  # * 2 because of gdir.grid.dx
        if bed_geometry == 'rectangular':
            flo.is_rectangular = np.ones(flo.nx).astype(np.bool)

        elif bed_geometry == 'parabolic' or bed_geometry == 'trapezoid':
            flo.is_rectangular = np.zeros(flo.nx).astype(np.bool)

        else:
            raise ValueError('unknown bed shape!')

        gdir.write_pickle(copy.deepcopy([flo]), 'inversion_flowlines')

        climate.apparent_mb_from_linear_mb(gdir,
                                           mb_gradient=model.mb_model.grad)
        mb_parameters = gdir.read_pickle('linear_mb_params')
        print('estimated ELA = ' + str(mb_parameters['ela_h']) +
              ' , grad = ' + str(mb_parameters['grad']))

        inversion.prepare_for_inversion(gdir)
        inversion.mass_conservation_inversion(gdir)
        inversion.filter_inversion_output(gdir)

        ocls = gdir.read_pickle('inversion_output')

        if get_bed_h:
            first_guess['bed_h'] = ocls[-1]['hgt'] - ocls[-1]['thick']
        else:
            first_guess['bed_h'] = measurements['bed_unknown']

        if get_shape:
            first_guess['shape'] = np.where(ocls[-1]['width'] >= 10,
                                            4 * ocls[-1]['thick'] /
                                            ocls[-1]['width']**2,
                                            4 * ocls[-1]['thick'] / 10**2)
        else:
            if bed_geometry == 'parabolic':
                first_guess['shape'] = measurements['shape_unknown']

        if bed_geometry == 'trapezoid':
            first_guess['w0'] = np.clip(ocls[-1]['width'] -
                                        lambdas[:len(ocls[-1]['thick'])] *
                                        ocls[-1]['thick'],
                                        0,
                                        np.Inf)

        return first_guess
    elif method == 'perfect':
        first_guess['bed_h'] = measurements['bed_unknown']
        first_guess['shape'] = measurements['shape_unknown']
        return first_guess
    else:
        raise ValueError('Unknown first guess method!')


def get_reg_parameters(opti_var,
                       measurements,
                       geometry,
                       mb_model,
                       torch_type,
                       bed_geometry,
                       first_guess,
                       glacier_state,
                       wanted_c_terms=None,
                       lambdas=None):
    if opti_var == 'bed_h':
        bed_h = measurements['bed_known']
        shape = measurements['shape_all']
        first_guess_cost_fct = first_guess['bed_h']
    elif opti_var == 'shape':
        bed_h = measurements['bed_all']
        shape = measurements['shape_known']
        first_guess_cost_fct = first_guess['shape']
    elif opti_var == 'bed_h and shape':
        bed_h = measurements['bed_known']
        shape = measurements['shape_known']
        first_guess_cost_fct = np.append(first_guess['bed_h'],
                                         first_guess['shape'])

    if wanted_c_terms is None:
        reg_parameters = np.array([1., 1., 1., 1., 1., 1.])
    else:
        reg_parameters = wanted_c_terms

    cost_fct = creat_cost_fct(
        bed_h=bed_h,
        shape=shape,
        spinup_surf=measurements['spinup_sfc'],
        reg_parameter=reg_parameters,
        ref_surf=measurements['sfc_h'],
        ref_width=measurements['widths'],
        ice_mask=measurements['ice_mask'],
        yrs_to_run=measurements['yrs_to_run'],
        dx=geometry['map_dx'],
        mb_model=mb_model,
        opti_var=opti_var,
        torch_type=torch_type,
        used_geometry=bed_geometry,
        get_c_terms=True,
        lambdas=lambdas)

    c_terms = cost_fct(first_guess_cost_fct)

    desired_mag = magnitude(c_terms[0])
    for i, c_term in enumerate(c_terms):
        if c_term != 0.:
            term_mag = magnitude(c_term)
            if term_mag == -100:
                reg_parameters[i] = 10.**(desired_mag)
            else:
                reg_parameters[i] = 10.**(desired_mag - term_mag)

    print('\nreg_parameters = ' + str(reg_parameters))
    return reg_parameters


def plot_result(res,
                measurements,
                geometry,
                first_guess,
                opti_parameter,
                bed_geometry):
    plot_height = 450
    plot_width = 800

    if opti_parameter == 'bed_h':
        bed_h_res = res.x
        if bed_geometry == 'parabolic':
            shape_res = measurements['shape_unknown']
        single_plot_height = plot_height
    elif opti_parameter == 'shape':
        bed_h_res = measurements['bed_unknown']
        shape_res = res.x
        single_plot_height = plot_height
    elif opti_parameter == 'bed_h and shape at once':
        split_point = int(len(res.x) / 2)
        bed_h_res = res.x[:split_point]
        shape_res = res.x[split_point:]
        single_plot_height = int(plot_height / 2)
    elif opti_parameter == 'bed_h and shape separeted':
        bed_h_res = res[0].x
        shape_res = res[1].x
        single_plot_height = int(plot_height / 2)
    else:
        raise ValueError('Unknown optimisation parameter!')

    # plot for bed height difference
    diff_estimated_bed = hv.Curve((geometry['distance_along_glacier'],
                                   np.append(bed_h_res,
                                             measurements['bed_known']) -
                                   measurements['bed_all']),
                                  'distance',
                                  'diff bed_h',
                                  label='diff estimated bed')

    diff_first_guess_bed = hv.Curve((geometry['distance_along_glacier'],
                                     np.append(first_guess['bed_h'],
                                               measurements['bed_known']) -
                                     measurements['bed_all']),
                                    'distance',
                                    'diff bed_h',
                                    label='first guess bed')

    zero_line_bed_h = hv.Curve((geometry['distance_along_glacier'],
                                np.zeros(len(
                                    geometry['distance_along_glacier']))),
                               'distance',
                               'diff bed_h').opts(line_color='black')

    bed_h_plot = (zero_line_bed_h *
                  diff_first_guess_bed *
                  diff_estimated_bed
                  ).opts(width=plot_width,
                         height=single_plot_height,
                         xaxis='top')

    # plot for shape difference
    diff_estimated_shape = hv.Curve((geometry['distance_along_glacier'],
                                     np.append(shape_res,
                                               measurements['shape_known']) -
                                     measurements['shape_all']),
                                    'distance',
                                    'diff shape',
                                    label='diff estimated shape')

    diff_first_guess_shape = hv.Curve((geometry['distance_along_glacier'],
                                       np.append(first_guess['shape'],
                                                 measurements['shape_known']) -
                                       measurements['shape_all']),
                                      'distance',
                                      'diff shape',
                                      label='first guess shape')

    zero_line_shape = hv.Curve((geometry['distance_along_glacier'],
                                np.zeros(len(
                                    geometry['distance_along_glacier']))),
                               'distance',
                               'diff shape').opts(line_color='black')

    shape_plot = (zero_line_shape *
                  diff_first_guess_shape *
                  diff_estimated_shape
                  ).opts(width=plot_width,
                         height=single_plot_height)

    if opti_parameter == 'bed_h':
        return bed_h_plot.opts(opts.Curve(line_width=3))
    elif opti_parameter == 'shape':
        return shape_plot.opts(opts.Curve(line_width=3))
    elif ((opti_parameter == 'bed_h and shape at once') or
          (opti_parameter == 'bed_h and shape separeted')):
        return_plot = (bed_h_plot + shape_plot).cols(1)
        return return_plot.opts(opts.Curve(line_width=3))
    else:
        raise ValueError('Unknown optimisation parameter!')


def get_spinup_sfc(measurements,
                   mb_model,
                   first_guess,
                   minimize_options,
                   bed_geometry,
                   geometry,
                   lambdas,
                   torch_type='double'):
    # cost function to find tbias for spinup
    cost_fct = creat_spinup_cost_fct(measurements,
                                     mb_model,
                                     first_guess,
                                     bed_geometry,
                                     geometry,
                                     lambdas,
                                     torch_type)

    # first guess for tbias is 0
    tbias = 0

    res = minimize(fun=cost_fct,
                   x0=tbias,
                   method='L-BFGS-B',
                   jac=True,
                   # bounds=bounds,
                   options=minimize_options,
                   # callback=dl.callback_fct
                   )

    # calculate final spinup surface
    tbias = res.x
    mb_model.temp_bias = tbias
    if bed_geometry == 'rectangular':
        oggm_fl = RectangularBedFlowline(surface_h=first_guess['bed_h'],
                                         bed_h=first_guess['bed_h'],
                                         widths=measurements['widths'],
                                         map_dx=geometry['map_dx'])
    elif bed_geometry == 'parabolic':
        oggm_fl = ParabolicBedFlowline(surface_h=first_guess['bed_h'],
                                       bed_h=first_guess['bed_h'],
                                       bed_shape=first_guess['shape'],
                                       map_dx=geometry['map_dx'])
    elif bed_geometry == 'trapezoid':
        oggm_fl = TrapezoidalBedFlowline(surface_h=first_guess['bed_h'],
                                         bed_h=first_guess['bed_h'],
                                         widths=first_guess['w0'],
                                         lambdas=geometry['lambdas'],
                                         map_dx=geometry['map_dx'],
                                         )

    final_model = oggm_FluxModel(oggm_fl, mb_model=mb_model, y0=0.)
    final_model.run_until_equilibrium()

    # TODO: print message with difference of actual and calculated sfc, t_bias

    return final_model.fls[0].surface_h
