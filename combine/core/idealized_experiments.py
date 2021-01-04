import numpy as np
from scipy.optimize import minimize
import holoviews as hv
from holoviews import opts
from combine.core.cost_function import create_cost_fct, creat_spinup_cost_fct
from combine.core.arithmetics import magnitude, to_numpy_array
from combine.core.massbalance_adapted import LinearMassBalance
from oggm.core.massbalance import LinearMassBalance as oggm_MassBalance
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
# hv.extension('matplotlib')
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
        geometry['bottom_height'] = 1500
        # define linear glacier bed with zero ice thickness
        geometry['bed_h'] = np.linspace(geometry['top_height'],
                                        geometry['bottom_height'],
                                        geometry['nx'])
    elif used_bed_h_geometry == 'cliff':
        # glacier top height
        geometry['top_height'] = 4000

        # glacier bottom height
        geometry['bottom_height'] = 1500
        # define extend of cliff
        cliff_top = 3410
        cliff_bottom = 3260

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
        offsets = np.random.normal(scale=15., size=geometry['nx'])

        # define random glacier bed
        geometry['bed_h'] = (np.linspace(geometry['top_height'],
                                         geometry['bottom_height'],
                                         geometry['nx']) +
                             offsets)
    elif used_bed_h_geometry == 'HEF':
        # values from OGGM with one flowline and border 160
        geometry['bed_h'] = \
          np.array([3562.70677504, 3527.4858046, 3472.52571313, 3411.57542718,
                    3358.0548272, 3313.48067006, 3275.55890779, 3238.0165849,
                    3198.62734657, 3160.09663662, 3124.83568412, 3091.90543697,
                    3059.74280012, 3025.98518381, 2990.94221003, 2961.71164828,
                    2947.79394967, 2940.16275285, 2930.52655444, 2917.76454803,
                    2897.43989702, 2868.87850995, 2839.53608826, 2813.09166515,
                    2784.75649346, 2754.33987814, 2726.41651275, 2702.75810288,
                    2671.37765628, 2660.67750326, 2653.78034147, 2640.92767901,
                    2621.94233493, 2606.31346286, 2596.23694635, 2596.96094506,
                    2618.26342448, 2639.39338067, 2639.48164866, 2622.04887056,
                    2597.01589119, 2570.19586357, 2543.28723268, 2519.60963731,
                    2501.2282631, 2486.37078546, 2479.82474295, 2486.0137465,
                    2498.45581989, 2506.57882406, 2505.26120971, 2497.25609396,
                    2488.54444807, 2486.01781571, 2488.07319594, 2484.93328333,
                    2473.01395369, 2455.87735422, 2439.25837156, 2425.08090702,
                    2414.43352677, 2407.97426318, 2403.93386355, 2402.85882024,
                    2405.51277492, 2408.23587675, 2406.27364113, 2396.0666943,
                    2377.62981613, 2443.88810655, 2466.16253877, 2462.32123851,
                    2457.6986927, 2455.99618625, 2435.87195744, 2429.09519908,
                    2421.68448148, 2415.98726658, 2410.75386663, 2405.50922222,
                    2400.56915322, 2395.8144795, 2391.50425869, 2388.47531217,
                    2385.90372679, 2382.23971773, 2377.02803048, 2370.81036541,
                    2364.38249107, 2357.990078, 2351.24701864, 2344.1122576,
                    2337.34364444, 2331.37823445, 2325.58828556, 2319.01701197,
                    2311.31823603, 2303.77261744, 2298.59398467, 2295.75923327,
                    2292.82364901, 2288.36897106, 2282.84753551, 2276.92350556,
                    2270.60930098, 2263.78438354, 2257.58653999, 2253.24143509,
                    2250.23216794, 2247.04847434, 2241.84954595, 2233.90123619,
                    2223.97760971, 2213.50639582, 2203.80855183, 2195.32142893,
                    2188.58185249, 2183.51776317, 2178.62763966, 2173.29104114,
                    2169.08990312, 2167.72838051, 2168.3868889, 2167.50819405,
                    2162.17193519, 2153.65719285, 2146.46297229, 2142.43968526,
                    2140.16844941, 2137.81983618, 2133.76756334, 2128.32928208,
                    2123.16760346, 2119.40727123, 2116.57457293, 2112.97211766,
                    2108.25128187, 2103.4734259, 2098.78014202, 2093.60237987,
                    2088.35186543, 2083.30900384, 2078.64879454, 2074.11818997,
                    2069.02779927, 2062.99739291, 2056.19305849, 2049.08804816,
                    2041.91698684, 2035.28561303, 2030.27399175, 2026.98694028,
                    2024.37521116, 2021.12090646, 2017.45103987, 2014.04693093,
                    2010.51105812, 2006.12727727, 2001.2439116, 1997.15979794,
                    1994.64072372, 1992.5539009, 1989.41614212, 1984.8695196,
                    1979.08294564, 1972.2212291, 1964.43079394, 1955.66734295,
                    1946.61881943, 1940.27537555])
        geometry['nx'] = 170

        # model grid spacing in m
        geometry['map_dx'] = 100

        # distance along glacier (x-axis of glacier profil plot) in km
        geometry['distance_along_glacier'] = (np.linspace(0, geometry['nx'],
                                                          geometry['nx']) *
                                              geometry['map_dx'] * 1e-3)
    else:
        raise ValueError('Unknown bed height geometry!')

    if used_along_glacier_geometry == 'constant':
        if bed_geometry == 'rectangular':
            geometry['widths'] = np.zeros(geometry['nx']) + 100.
        elif bed_geometry == 'parabolic':
            geometry['bed_shape'] = np.zeros(geometry['nx']) + 1.
        elif bed_geometry == 'trapezoidal':
            geometry['w0'] = np.zeros(geometry['nx']) + 100.
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

    elif used_along_glacier_geometry == 'wide_top':
        top_len = np.int(np.round(geometry['nx']/5))
        bottom_len = np.int(geometry['nx'] - top_len)

        if bed_geometry == 'rectangular':
            geometry['widths'] = np.append(np.ones(top_len) * 100.,
                                           np.ones(bottom_len) * 50.)
        elif bed_geometry == 'parabolic':
            geometry['bed_shape'] = np.append(np.ones(top_len) * 1.,
                                              np.ones(bottom_len) * 2.)
        elif bed_geometry == 'trapezoidal':
            geometry['w0'] = np.append(np.ones(top_len) * 100.,
                                       np.ones(bottom_len) * 50.)

    elif used_along_glacier_geometry == 'HEF':
        if bed_geometry == 'rectangular':
            raise ValueError('HEF with rectangular bed shape not supported'
                             ' right now!')
        elif bed_geometry == 'parabolic':
            # shapes from OGGM with one flowline and border 160, for small
            # values (below 10e-3) just the minimum value is used (in OGGM this
            # are turned into trapezoidal bed shapes), for calculation see
            # first guess read_bed_shape
            geometry['bed_shape'] = np.array(
               [0.00107054, 0.00107054, 0.00107054, 0.00107054, 0.00107054,
                0.00107054, 0.00107054, 0.00107054, 0.00107054, 0.00107054,
                0.00107054, 0.00107054, 0.00107054, 0.00107054, 0.00107054,
                0.00107054, 0.00107054, 0.00107054, 0.00107054, 0.00107054,
                0.00107054, 0.00107054, 0.00107054, 0.00107054, 0.00107054,
                0.00107054, 0.00107054, 0.00107054, 0.00107054, 0.00120932,
                0.00133276, 0.00163194, 0.00215384, 0.00252789, 0.00269482,
                0.00250768, 0.00165607, 0.00113463, 0.0011865, 0.00144704,
                0.00162967, 0.00186586, 0.00224803, 0.00248859, 0.00253416,
                0.00262055, 0.00263808, 0.00248462, 0.00223089, 0.0019766,
                0.0018938, 0.00192726, 0.00198884, 0.00205006, 0.00202163,
                0.00201426, 0.00223581, 0.00275373, 0.00325139, 0.00356528,
                0.0037691, 0.00390319, 0.00403195, 0.00410994, 0.00406024,
                0.00412114, 0.00453837, 0.00528921, 0.00667731, 0.00339772,
                0.00168472, 0.00172862, 0.00179631, 0.00128631, 0.00165788,
                0.00169041, 0.00171644, 0.00178054, 0.00179791, 0.00179014,
                0.00176765, 0.00170378, 0.00162566, 0.00154092, 0.00146853,
                0.0014306, 0.00145756, 0.00151241, 0.00158214, 0.00170855,
                0.0018003, 0.00184653, 0.00187683, 0.00188681, 0.0018165,
                0.00173589, 0.00167999, 0.00165497, 0.00164312, 0.00164765,
                0.00167924, 0.00170735, 0.00170471, 0.00168203, 0.00169445,
                0.0017083, 0.00172693, 0.00178663, 0.00186519, 0.00190758,
                0.00192303, 0.00193766, 0.00190639, 0.00187091, 0.00190057,
                0.00194452, 0.00192548, 0.00186401, 0.00185822, 0.00183096,
                0.00184521, 0.00194849, 0.00209709, 0.00227083, 0.00242281,
                0.00253738, 0.0026477, 0.00275435, 0.0028146, 0.00287328,
                0.00294054, 0.00298865, 0.0030354, 0.00301016, 0.00295979,
                0.00283414, 0.00271797, 0.00259706, 0.00250961, 0.00234948,
                0.00227174, 0.00212203, 0.00199175, 0.00193261, 0.00192727,
                0.00192839, 0.00193488, 0.00192495, 0.00186157, 0.001889,
                0.00187572, 0.00185337, 0.00182207, 0.00174877, 0.00159437,
                0.00143465, 0.00132158, 0.00122413, 0.00117649, 0.0011509,
                0.00116002, 0.00118688, 0.0012093, 0.00122351, 0.00123201,
                0.00124645, 0.00125037, 0.0012631, 0.00126884, 0.00128768])
        else:
            raise ValueError('Unknown bed shape!')
    else:
        raise ValueError('Unknown along glacier geometry!')

    return geometry


def define_mb_model(mb_opts={'ELA': np.array([3100., 3300.]),
                             'grad': np.array([4., 4.])}):
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
                                         widths=(geometry['widths'] /
                                                 geometry['map_dx']),
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
                                         widths=(geometry['w0'] /
                                                 geometry['map_dx']),
                                         lambdas=np.zeros(geometry['nx']) + 1.,
                                         map_dx=geometry['map_dx'],
                                         )
    else:
        raise ValueError('Unkown bed shape!')

    # convert COMBINE MassBalanceModel to OGGM MassBalanceModel
    assert len(mb_model) >= 2,\
        'Need at least two mass balance models!'
    oggm_mb_model = [oggm_MassBalance(
                        to_numpy_array(mb_model[0].ela_h),
                        grad=to_numpy_array(mb_model[0].grad)),
                     oggm_MassBalance(
                         to_numpy_array(mb_model[1].ela_h),
                         grad=to_numpy_array(mb_model[1].grad))]

    # let the model run according to the desired glacier state
    if glacier_state == 'equilibrium':
        ref_model = oggm_FluxModel(oggm_fl, mb_model=oggm_mb_model[1], y0=0.)
        ref_model.run_until_equilibrium()

    elif glacier_state == 'advancing':
        years_to_advance = None
        for i in range(2):
            start_model = oggm_FluxModel(oggm_fl,
                                         mb_model=oggm_mb_model[1],
                                         y0=0.)
            start_model.run_until_equilibrium()
            advance_model = oggm_FluxModel(start_model.fls[-1],
                                           mb_model=oggm_mb_model[0],
                                           y0=0.)
            if years_to_advance is None:
                advance_model.run_until_equilibrium()
                years_to_advance = int(advance_model.yr / 2)
            else:
                advance_model.run_until(years_to_advance)
                break
        ref_model = advance_model

    elif glacier_state in ['retreating', 'retreating with unknow spinup']:
        years_to_retreat = None
        for i in range(2):
            start_model = oggm_FluxModel(oggm_fl,
                                         mb_model=oggm_mb_model[0],
                                         y0=0.)
            start_model.run_until_equilibrium()
            retreat_model = oggm_FluxModel(start_model.fls[-1],
                                           mb_model=oggm_mb_model[1],
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

    # find ice and ice free points, with a thickness larger than 0.01 m
    ice_mask = np.where(ref_model.fls[0].thick > 10e-2, True, False)
    measurements['ice_mask'] = ice_mask

    # define spinup_sfc from which the experiments start and years to run
    if glacier_state == 'equilibrium':
        measurements['spinup_sfc'] = ref_model.fls[0].surface_h
        measurements['yrs_to_run'] = ref_model.yr
    elif glacier_state in ['retreating', 'advancing']:
        measurements['spinup_sfc'] = start_model.fls[0].surface_h
        measurements['yrs_to_run'] = ref_model.yr
    elif glacier_state == 'retreating with unknow spinup':
        measurements['spinup_volume'] = start_model.fls[0].volume_m3
        measurements['spinup_bin_area'] = start_model.fls[0].bin_area_m2
        measurements['ref_spinup_sfc'] = start_model.fls[0].surface_h
        measurements['yrs_to_run'] = ref_model.yr
    else:
        raise ValueError('Unknown glacier state!')

    return measurements


def get_first_guess(measurements,
                    bed_geometry='rectangular',
                    opti_parameter='bed_h',
                    job_id=0,
                    task_id=0):  # only needed for computation on the cluster
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
    tmpdir = utils.gettempdir(dirname=('OGGM_Inversion_' + str(job_id) + '_' +
                                       str(task_id)),
                              reset=True,
                              home=True)  # home=True needed for cluster
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
    print('\n    estimated ELA = ' + str(mb_parameters['ela_h']) +
          ' , grad = ' + str(mb_parameters['grad']))

    # do the OGGM inversion
    inversion.prepare_for_inversion(gdir)
    inversion.mass_conservation_inversion(gdir)

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
        # for trapazeoidal bed shape lambda is set to 1, to still see it in the
        # equation there is the multiplication with 1.
        # (see https://docs.oggm.org/en/latest/ice-dynamics.html#trapezoidal)
        return np.clip(ocls[-1]['width'] - 1. * ocls[-1]['thick'],
                       1.,
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
            # filter first guess bed_h so that w0 > 0.5,
            # division through lambda = 1.
            max_thick = (flo.widths - 0.5) / 1.
            first_guess['bed_h'] = np.where(sh - first_guess['bed_h'] >
                                            max_thick,
                                            sh - max_thick,
                                            first_guess['bed_h'])
        elif opti_parameter == 'w0':
            first_guess['w0'] = read_w0(ocls)
        elif opti_parameter == 'bed_h and w0':
            first_guess['bed_h'] = read_bed_h(ocls)
            first_guess['w0'] = read_w0(ocls)
        else:
            raise ValueError('Unknown optimisation parameter for trapezoidal!')

    else:
        raise ValueError('Unknown bed geometry!')

    # empty the tmp directory on the cluster
    tmpdir = utils.gettempdir(dirname=('OGGM_Inversion_' + str(job_id) + '_' +
                                       str(task_id)),
                              reset=True,
                              home=True)  # home=True needed for cluster
    return first_guess


def first_guess_run(first_guess,
                    bed_geometry,
                    measurements,
                    mb_model,
                    geometry,
                    opti_parameter='bed_h',):
    ice_mask = measurements['ice_mask']

    if opti_parameter in ['bed_h', 'bed_h and bed_shape', 'bed_h and w0']:
        bed_h = np.empty(len(first_guess['bed_h']) +
                         len(geometry['bed_h'][~ice_mask]))
        bed_h[ice_mask] = first_guess['bed_h']
        bed_h[~ice_mask] = geometry['bed_h'][~ice_mask]
    else:
        bed_h = geometry['bed_h']

    # Create a flowline
    if bed_geometry == 'rectangular':
        oggm_fl = RectangularBedFlowline(surface_h=measurements['spinup_sfc'],
                                         bed_h=bed_h,
                                         widths=geometry['widths'],
                                         map_dx=geometry['map_dx'])
    elif bed_geometry == 'parabolic':
        if opti_parameter in ['bed_shape', 'bed_h and bed_shape']:
            bed_shape = np.empty(len(first_guess['bed_shape']) +
                                 len(geometry['bed_shape'][~ice_mask]))
            bed_shape[ice_mask] = first_guess['bed_shape']
            bed_shape[~ice_mask] = geometry['bed_shape'][~ice_mask]
        else:
            bed_shape = geometry['bed_shape']
        oggm_fl = ParabolicBedFlowline(surface_h=measurements['spinup_sfc'],
                                       bed_h=bed_h,
                                       bed_shape=bed_shape,
                                       map_dx=geometry['map_dx'])
    elif bed_geometry == 'trapezoidal':
        if opti_parameter in ['w0', 'bed_h and w0']:
            w0 = np.empty(len(first_guess['w0']) +
                          len(geometry['w0'][~ice_mask]))
            w0[ice_mask] = first_guess['w0']
            w0[~ice_mask] = geometry['w0'][~ice_mask]
        else:
            # calculate widths so that w0 is genarated with spinup_sfc
            w0 = np.where((measurements['spinup_sfc'] - bed_h) > 0,
                          geometry['w0'] + 1. * (measurements['spinup_sfc']
                                                 - bed_h),
                          geometry['w0'])
        # for trapezoidal bed lambda is always set to 1
        # (see https://docs.oggm.org/en/latest/ice-dynamics.html#trapezoidal)
        oggm_fl = TrapezoidalBedFlowline(surface_h=measurements['spinup_sfc'],
                                         bed_h=bed_h,
                                         widths=(w0 + 1. *
                                                 (measurements['spinup_sfc'] -
                                                  bed_h)
                                                 ) / geometry['map_dx'],
                                         lambdas=np.zeros(geometry['nx']) + 1.,
                                         map_dx=geometry['map_dx'],
                                         )

    # convert COMBINE MassBalanceModel to OGGM MassBalanceModel
    oggm_mb_model = oggm_MassBalance(to_numpy_array(mb_model.ela_h),
                                     grad=to_numpy_array(mb_model.grad))

    # create a model and let it run
    model = oggm_FluxModel(oggm_fl, mb_model=oggm_mb_model, y0=0.)
    model.run_until(measurements['yrs_to_run'])

    # get values of flowline
    fg_sfc_h = model.fls[0].surface_h
    fg_widths = model.fls[0].widths_m

    return fg_sfc_h, fg_widths


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
                two_parameter_option='None',
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

    print('\n    reg_parameters = ' + str(reg_parameters))
    return reg_parameters


def get_bounds(control_var,
               measurements=None,
               min_value=None,
               max_value=None):
    if control_var == 'bed_h':
        if measurements is None:
            raise ValueError('Need surface height measurements for bed_h '
                             'bounds!')
        if (min_value is None) or (max_value is None):
            raise ValueError('Need max and min ice thickness for bed_h '
                             'bounds!')
        return [(sfc_h - max_value, sfc_h - min_value)
                for sfc_h in measurements]

    elif control_var == 'bed_shape':
        if measurements is None:
            raise ValueError('Need surface height/width measurements for '
                             'bed_shape to have length of needed bounds!')
        return [(min_value, None) for m in measurements]

    elif control_var == 'w0':
        if measurements is None:
            raise ValueError('Need width measurements for w0 bounds!')
        if min_value is None:
            raise ValueError('Need a minimum value for w0!')
        return [(min_value, width) for width in measurements]
    else:
        raise ValueError('Unknown control variable for bounds calculation!')


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
