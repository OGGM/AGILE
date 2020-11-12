import numpy as np
from scipy.optimize import minimize
import holoviews as hv
from holoviews import opts
from combine.core.cost_function import create_cost_fct, creat_spinup_cost_fct
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
hv.extension('bokeh')
# cfg.initialize()


def define_geometry(used_bed_h_geometry='linear',
                    used_along_glacier_geometry='constant',
                    bed_geometry='rectangular'):
    '''
    Defines the glacier domain for idealized experiments.

    Parameters
    ----------
    used_bed_h_geometry : string, optional
        Defines the geometry of the glacier bed height.
        Options: 'linear', 'cliff' or 'random'
        The default is 'linear'.
    used_along_glacier_geometry : string, optional
        Defines the geometry along the glacier.
        Options: 'constant' or 'random'
        The default is 'constant'.
    bed_geometry : string, optional
        Defines the shape of the glacier.
        Options: 'rectangular', 'parabolic' or 'trapezoidal'.
        The default is 'rectangular'.

    Returns
    -------
    geometry : dict
        Contains geometry information for the glacier domain:
            'nx': Number of grid points along the glacier
            'map_dx': grid point spacing in m
            'distance_along_glacier': coordinates along the glacier
            'top_height': highest point of glacier
            'bottom_height': lowest point of glacier
            'bed_h': glacier bed height at each grid point
            depending on bed_geometry one variable describing the shape:
            for 'rectangular': 'widths'
            for 'parabolic': 'bed_shape'
            for 'trapezoidal': 'w0'

    '''
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
                         int(geometry['nx'] / 4)),
             np.linspace(cliff_bottom,
                         geometry['bottom_height'],
                         int(geometry['nx'] * 3 / 4))))
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

    if used_along_glacier_geometry == 'constant':
        if bed_geometry == 'rectangular':
            geometry['widths'] = np.zeros(geometry['nx']) + 1.
        elif bed_geometry == 'parabolic':
            geometry['bed_shape'] = np.zeros(geometry['nx']) + 1.
        elif bed_geometry == 'trapezoidal':
            geometry['w0'] = np.zeros(geometry['nx']) + 1.
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
            geometry['bed_shape'] = random_shape
        elif bed_geometry == 'trapezoidal':
            geometry['w0'] = random_shape
        else:
            raise ValueError('Unkonwn bed shape!')
    else:
        raise ValueError('Unknown along glacier geometry!')

    return geometry


def define_mb_model(mb_opts={'ELA': np.array([3000.]),
                             'grad': np.array([4.])}):
    '''
    generate one or more OGGM LinearMassBalance models

    Parameters
    ----------
    mb_opts : dict of ELA and grad, optional
        ELA defines the equilibrium line altitude and grad defines the
        gradient of the mass-balance-profile. The length of 'ELA' and 'grad'
        must be the same.
        The default is {'ELA': np.array([3000.]), 'grad': np.array([4.])}.

    Returns
    -------
    mb_model : LinearMassBalance or list of LinearMassBalances
        If one ELA and grad is given only one LinearMassBalance is returned,
        otherwise a list with the same length as ELA and grad.

    '''
    # define linear mass balance
    ELA = mb_opts['ELA']
    mb_gradient = mb_opts['grad']
    if len(ELA) == 1:
        mb_model = LinearMassBalance(ELA, grad=mb_gradient)
    else:
        if len(ELA) != len(mb_gradient):
            raise ValueError('The length of ELA and grad must be the same!!!')
        mb_model = []
        for i in np.arange(len(ELA)):
            mb_model.append(LinearMassBalance(ELA[i],
                                              grad=mb_gradient[i]))

    return mb_model


def create_measurements(geometry,
                        mb_model,
                        bed_geometry='rectangular',
                        glacier_state='equilibrium'):
    '''
    Creating artifical measurements for idealized experiments using the OGGM
    FluxModel.

    Parameters
    ----------
    geometry : dict
        Describing the geometry of the glacier. Depending on the different
        bed geometrys it must contain different
        variables describing it (see Docstring of function define_geometry).
    mb_model : OGGM MassBalanceModel or list of OGGM MassBalanceModels
        For the glacier states 'equilibrium' or 'advancing' defines the
        driving mass balance. For retreating two mass balance models are
        needed.
    bed_geometry : string, optional
        Defines the bed geometry, the options are: 'rectangular', 'parabolic'
        and 'trapezoidal'. The default is 'rectangular'.
    glacier_state : string, optional
        Defines the glacier state at the end, the options are: 'equilibrium',
        'advancing' or 'retreating'. For 'equilibrium' the glacier is runs
        until it reaches equilibrium, for 'advancing' the glacier runs half the
        time before it reaches equilibrium and in the case of 'retreating'
        the glacier runs until equilibrium with the first MassBalanceModel and
        then runs half the time until it reaches the equilibrium with the
        second MassBalanceModelmb_model. The default is 'equilibrium'.

    Returns
    -------
    measurements : dict
        Dictionary containing the artifical measurements from the OGGM model
        run output. Contains a ice mask indicating ice with a thickness larger
        then 0.01 m.

    '''
    measurements = {}

    # Create a flowline
    if bed_geometry == 'rectangular':
        oggm_fl = RectangularBedFlowline(surface_h=geometry['bed_h'],
                                         bed_h=geometry['bed_h'],
                                         widths=geometry['widths'],
                                         map_dx=geometry['map_dx'])
    elif bed_geometry == 'parabolic':
        oggm_fl = ParabolicBedFlowline(surface_h=geometry['bed_h'],
                                       bed_h=geometry['bed_h'],
                                       bed_shape=geometry['bed_shape'],
                                       map_dx=geometry['map_dx'])
    elif bed_geometry == 'trapezoidal':
        # for trapezoidal bed lambda is always set to 1
        # (see https://docs.oggm.org/en/latest/ice-dynamics.html#trapezoidal)
        oggm_fl = TrapezoidalBedFlowline(surface_h=geometry['bed_h'],
                                         bed_h=geometry['bed_h'],
                                         widths=geometry['w0'],
                                         lambdas=np.zeros(geometry['nx']) + 1.,
                                         map_dx=geometry['map_dx'],
                                         )
    else:
        raise ValueError('Unkown bed shape!')

    # let the model run according to the desired glacier state
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

    # save model
    measurements['model'] = ref_model

    # get 'measurements' of glacier
    measurements['sfc_h'] = ref_model.fls[0].surface_h
    measurements['widths'] = ref_model.fls[0].widths_m
    measurements['yrs_to_run'] = ref_model.yr

    # find ice and ice free points, with a thickness larger than 0.01 m
    ice_mask = np.where(ref_model.fls[0].thick > 10e-2, True, False)
    measurements['ice_mask'] = ice_mask

    # define spinup_sfc from which the experiments start
    if (glacier_state == 'equilibrium') or (glacier_state == 'advancing'):
        measurements['spinup_sfc'] = np.zeros(len(measurements['sfc_h']))
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
                    bed_geometry='rectangular',
                    opti_parameter='bed_h'):
    '''
    Creates a first guess using the inversion of OGGM.

    Parameters
    ----------
    measurements : dict
        Containing measurements of the glacier surface.
    bed_geometry : string, optional
        Defines the bed geometry, the options are: 'rectangular', 'parabolic'
        and 'trapezoidal'. The default is 'rectangular'.
    opti_parameter : string, optional
        Defines the optimisation parameter. Depending on the bed geometry this
        could be one ore two.
        Options for 'rectangular': 'bed_h'.
        Options for 'parabolic': 'bed_h', 'bed_shape' or 'bed_h and bed_shape'
        Options for 'trapezoidal': 'bed_h', 'w0' or 'bed_h and w0'
        The default is 'bed_h'.

    Returns
    -------
    first_guess : dict
        Containing first guess parameters.

    '''
    # use the FluxModel of the measurements to get the 'coordinates' of the
    # flowline to create a new flowline for the inversion (see below)
    model = measurements['model']

    # use demo files to initialise the inversion
    cfg.set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
    cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
    cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')

    hef_file = get_demo_file('Hintereisferner_RGI5.shp')
    entity = gpd.read_file(hef_file).iloc[0]

    # create a temporary directory for inversion
    tmpdir = utils.gettempdir(dirname='OGGM_Inversion', reset=True)
    gdir = GlacierDirectory(entity, base_dir=tmpdir, reset=True)
    define_glacier_region(gdir)

    # use the flowline of the 'measurement' run to create a new flowline
    # only where there is ice along the flowline
    fl = model.fls[0]
    pg = np.where(fl.thick > 10e-2)
    ice_index = pg[0]
    line = shpg.LineString([fl.line.coords[int(p)] for p in ice_index])
    sh = fl.surface_h[ice_index]
    # *2 because of gdir.grid.dx, default dx for hef demo is double of the used
    # flowline dx
    flo = centerlines.Centerline(line, dx=fl.dx * 2,
                                 surface_h=sh)
    flo.widths = fl.widths[ice_index] * 2  # * 2 because of gdir.grid.dx

    # set the geometry along the new flowline
    if bed_geometry == 'rectangular':
        flo.is_rectangular = np.ones(flo.nx).astype(np.bool)
    elif bed_geometry in ['parabolic', 'trapezoidal']:
        flo.is_rectangular = np.zeros(flo.nx).astype(np.bool)
    else:
        raise ValueError('unknown bed shape!')

    # save the new flowline in the temporary directory
    gdir.write_pickle(copy.deepcopy([flo]), 'inversion_flowlines')

    # get the apparent mass balance parameters (assuming glacier is in
    # equilibrium, even this is not true)
    climate.apparent_mb_from_linear_mb(gdir,
                                       mb_gradient=model.mb_model.grad)
    mb_parameters = gdir.read_pickle('linear_mb_params')
    print('estimated ELA = ' + str(mb_parameters['ela_h']) +
          ' , grad = ' + str(mb_parameters['grad']))

    # do the OGGM inversion
    inversion.prepare_for_inversion(gdir)
    inversion.mass_conservation_inversion(gdir)
    # TODO: check what this is doing and if I should use it
    # inversion.filter_inversion_output(gdir)

    # read the result of OGGM inversion
    ocls = gdir.read_pickle('inversion_output')

    # help functions to read out first guess
    def read_bed_h(ocls):
        return ocls[-1]['hgt'] - ocls[-1]['thick']

    def read_bed_shape(ocls):
        return np.where(ocls[-1]['width'] >= 10,
                        4 * ocls[-1]['thick'] /
                        ocls[-1]['width']**2,
                        4 * ocls[-1]['thick'] / 10**2)

    def read_w0(ocls):
        # TODO: check that w0 is positive
        # for trapazeoidal bed shape lambda is set to 1, to still see it in the
        # equation there is the multiplication with 1.
        # (see https://docs.oggm.org/en/latest/ice-dynamics.html#trapezoidal)
        return np.clip(ocls[-1]['width'] - 1. * ocls[-1]['thick'],
                       0,
                       np.Inf)

    # depending on bed_geometry and opti_parameter get first guess
    first_guess = {}

    if bed_geometry == 'rectangular':
        if opti_parameter == 'bed_h':
            first_guess['bed_h'] = read_bed_h(ocls)
        else:
            raise ValueError('Unknown optimisation parameter for rectangular!')

    elif bed_geometry == 'parabolic':
        if opti_parameter == 'bed_h':
            first_guess['bed_h'] = read_bed_h(ocls)
        elif opti_parameter == 'bed_shape':
            first_guess['bed_shape'] = read_bed_shape(ocls)
        elif opti_parameter == 'bed_h and bed_shape':
            first_guess['bed_h'] = read_bed_h(ocls)
            first_guess['bed_shape'] = read_bed_shape(ocls)
        else:
            raise ValueError('Unknown optimisation parameter for parabolic!')

    elif bed_geometry == 'trapezoidal':
        if opti_parameter == 'bed_h':
            first_guess['bed_h'] = read_bed_h(ocls)
        elif opti_parameter == 'w0':
            first_guess['w0'] = read_w0(ocls)
        elif opti_parameter == 'bed_h and w0':
            first_guess['bed_h'] = read_bed_h(ocls)
            first_guess['w0'] = read_w0(ocls)
        else:
            raise ValueError('Unknown optimisation parameter for trapezoidal!')

    else:
        raise ValueError('Unknown bed geometry!')

    return first_guess


def get_reg_parameters(opti_var,
                       measurements,
                       geometry,
                       mb_model,
                       bed_geometry,
                       first_guess,
                       torch_type='double',
                       wanted_c_terms=None):
    '''
    Calculates the regularisation parameters by scaling the magnitude of each
    term of the cost function. How the scaling should look like can be defined
    with the wanted_c_terms variable (e.g. wanted_c_terms=[1., 10., 1., 1.]
    equals that the magnitude of the second cost term is 1/10 of the other
    terms)

    Parameters
    ----------
    opti_var : str
        Defines the optimisation parameter. Depending on the bed geometry this
        could be one ore two.
        Options for 'rectangular': 'bed_h'.
        Options for 'parabolic': 'bed_h', 'bed_shape' or 'bed_h and bed_shape'
        Options for 'trapezoidal': 'bed_h', 'w0' or 'bed_h and w0'
    measurements : dict
        Dictionary containing the measurements from:
            'spinup_sfc_h' : the spinup surface height (start ice height)
            'sfc_h': the desired ice surface height at the end
            'widths': the desired widths at the end
            'yrs_to_run': the number of years the model should run
            'ice_mask': indicates where ice is located at the end (TRUE = ice)
    geometry : dict
        Describing the geometry of the glacier. Depending on the different
        bed geometrys it must contain different
        variables describing it (see Docstring of function define_geometry).
    mb_model : :py:class:`oggm.core.massbalance.MassBalanceModel`
        The mass balance model to use.
    bed_geometry : str
        Defines the bed shape.
        Options: 'rectangular', 'parabolic' or 'trapezoidal'
    first_guess : dict
        Containing first guess parameters.
    torch_type : str, optional
        Defines type for torch.Tensor. If 'double' use torch.double, otherwise
        use torch.float. The default is 'double'.
    wanted_c_terms : :py:class:`numpy.ndarray`, optional
        Can scale the individual cost terms,
        e.g. wanted_c_terms=[1., 10., 1., 1.] equals that the magnitude of the
        second cost term is 1/10 of the other terms. The default is None.

    Returns
    -------
    reg_parameters : :py:class:`numpy.ndarray`
        Regularisation parameters for the individual terms of the cost
        function.

    '''

    if opti_var in ['bed_h', 'bed_shape', 'w0']:
        guess_parameter = first_guess[opti_var]
        known_parameter = geometry[opti_var][~measurements['ice_mask']]
        # here the second geomety variable is selected
        if opti_var == 'bed_h':
            if bed_geometry == 'rectangular':
                geometry_var = geometry['widths']
            elif bed_geometry == 'parabolic':
                geometry_var = geometry['bed_shape']
            elif bed_geometry == 'trapezoidal':
                geometry_var = geometry['w0']
            else:
                raise ValueError('Unknown bed geometry!')
        else:
            geometry_var = geometry['bed_h']
    elif opti_var in ['bed_h and bed_shape', 'bed_h and w0']:
        if opti_var == 'bed_h and bed_shape':
            opti_var_2 = 'bed_shape'
        else:
            opti_var_2 = 'w0'

        guess_parameter = np.append(first_guess['bed_h'],
                                    first_guess[opti_var_2])
        known_parameter = np.append(
            geometry['bed_h'][~measurements['ice_mask']],
            geometry[opti_var_2][~measurements['ice_mask']])
        geometry_var = None
    else:
        raise ValueError('Unknown optimisation variable!')

    if wanted_c_terms is None:
        reg_parameters = np.array([1., 1., 1., 1.])
    else:
        reg_parameters = wanted_c_terms

    cost_fct = create_cost_fct(
                known_parameter=known_parameter,
                geometry_var=geometry_var,
                bed_geometry=bed_geometry,
                measurements=measurements,
                reg_parameters=reg_parameters,
                dx=geometry['map_dx'],
                mb_model=mb_model,
                opti_var=opti_var,
                datalogger=None,
                only_get_c_terms=True)

    c_terms = cost_fct(guess_parameter)

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


def plot_result(dl, plot_height=450, plot_width=800):
    x = dl.geometry['distance_along_glacier']

    # start with first optimisation variable
    y1_final_guess = np.zeros(len(x))
    y1_first_guess = np.zeros(len(x))

    y1_final_guess[dl.ice_mask] = dl.guessed_opti_var_1[-1]
    y1_final_guess[~dl.ice_mask] = dl.known_opti_var_1

    y1_first_guess[dl.ice_mask] = dl.first_guessed_opti_var_1
    y1_first_guess[~dl.ice_mask] = dl.known_opti_var_1

    # if only one
    if dl.opti_parameter in ['bed_h', 'bed_shape', 'w0']:
        y2_final_guess = None
        y2_first_guess = None

        single_plot_height = plot_height

    elif dl.opti_parameter in ['bed_h and bed_shape', 'bed_h and w0']:
        y2_final_guess = np.zeros(len(x))
        y2_first_guess = np.zeros(len(x))

        y2_final_guess[dl.ice_mask] = dl.guessed_opti_var_2[-1]
        y2_final_guess[~dl.ice_mask] = dl.known_opti_var_2

        y2_first_guess[dl.ice_mask] = dl.first_guessed_opti_var_2
        y2_first_guess[~dl.ice_mask] = dl.known_opti_var_2

        single_plot_height = plot_height / 2

    else:
        raise ValueError('Unknown optimisation parameter!')

    # plot for first optimisation variable
    diff_estimated_opti_1 = hv.Curve((x,
                                      y1_final_guess -
                                      dl.geometry[dl.opti_var_1]),
                                     'distance',
                                     'diff ' + dl.opti_var_1,
                                     label='diff estimated ' + dl.opti_var_1)

    diff_first_guess_opti_1 = hv.Curve((x,
                                        y1_first_guess -
                                        dl.geometry[dl.opti_var_1]),
                                       'distance',
                                       'diff ' + dl.opti_var_1,
                                       label='diff estimated ' + dl.opti_var_1)

    zero_line_opti_1 = hv.Curve((x,
                                 np.zeros(len(x))),
                                'distance',
                                'diff ' + dl.opti_var_1
                                ).opts(line_color='black')

    opti_1_plot = (zero_line_opti_1 *
                   diff_first_guess_opti_1 *
                   diff_estimated_opti_1
                   ).opts(width=plot_width,
                          height=single_plot_height,
                          xaxis='top')

    if y2_final_guess is not None:
        # plot for first optimisation variable
        diff_estimated_opti_2 = hv.Curve((x,
                                          y2_final_guess -
                                          dl.geometry[dl.opti_var_2]),
                                         'distance',
                                         'diff ' + dl.opti_var_2,
                                         label='diff estimated ' +
                                         dl.opti_var_2)

        diff_first_guess_opti_2 = hv.Curve((x,
                                            y2_first_guess -
                                            dl.geometry[dl.opti_var_2]),
                                           'distance',
                                           'diff ' + dl.opti_var_2,
                                           label='diff estimated ' +
                                           dl.opti_var_2)

        zero_line_opti_2 = hv.Curve((x,
                                     np.zeros(len(x))),
                                    'distance',
                                    'diff ' + dl.opti_var_2
                                    ).opts(line_color='black')

        opti_2_plot = (zero_line_opti_2 *
                       diff_first_guess_opti_2 *
                       diff_estimated_opti_2
                       ).opts(width=plot_width,
                              height=single_plot_height,
                              xaxis='top')

        final_plot = (opti_1_plot + opti_2_plot).cols(1)

    else:
        final_plot = opti_1_plot

    return final_plot.opts(opts.Curve(line_width=3))


def get_spinup_sfc(measurements,
                   mb_model,
                   first_guess,
                   minimize_options,
                   bed_geometry,
                   geometry,
                   torch_type='double'):
    # cost function to find tbias for spinup
    cost_fct = creat_spinup_cost_fct(measurements,
                                     mb_model,
                                     first_guess,
                                     bed_geometry,
                                     geometry,
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
    elif bed_geometry == 'trapezoidal':
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
