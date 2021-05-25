import numpy as np

from combine1d.core.type_conversions import to_numpy_array

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


def get_first_guess(measurements,
                    bed_geometry='rectangular',
                    opti_parameter='bed_h',
                    glacier_state='equilibrium',
                    job_id=0,    # only needed for computation on the cluster
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
    first_guess = {}

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

    if glacier_state == 'retreating with unknow spinup':
        first_guess['max_spinup_ELA'] = mb_parameters['ela_h']
        first_guess['spinup_ELA'] = mb_parameters['ela_h'] - 50
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
                    glacier_state='equilibrium',
                    spinup_yrs=200,
                    opti_parameter='bed_h',):
    '''
    Runs the experiment with the first guess to quantify misfit between
    modelled and observed glacier surface.

    Parameters
    ----------
    first_guess : TYPE
        DESCRIPTION.
    bed_geometry : TYPE
        DESCRIPTION.
    measurements : TYPE
        DESCRIPTION.
    mb_model : TYPE
        DESCRIPTION.
    geometry : TYPE
        DESCRIPTION.
    glacier_state : TYPE, optional
        DESCRIPTION. The default is 'equilibrium'.
    spinup_yrs : TYPE, optional
        DESCRIPTION. The default is 200.
    opti_parameter : TYPE, optional
        DESCRIPTION. The default is 'bed_h'.

    Returns
    -------
    fg_run : TYPE
        DESCRIPTION.

    '''
    fg_run = {}
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

        # calculate widths, want to set w0, but w0 is calculated internaly,
        # so here reverse this calculation
        if measurements['spinup_sfc'].sum() == 0.:
            widths_use = w0
        else:
            widths_use = (w0 + 1. * (measurements['spinup_sfc'] - bed_h))
        # for trapezoidal bed lambda is always set to 1
        # (see https://docs.oggm.org/en/latest/ice-dynamics.html#trapezoidal)
        oggm_fl = TrapezoidalBedFlowline(surface_h=measurements['spinup_sfc'],
                                         bed_h=bed_h,
                                         widths=(widths_use /
                                                 geometry['map_dx']),
                                         lambdas=np.zeros(geometry['nx']) + 1.,
                                         map_dx=geometry['map_dx'],
                                         )

    if glacier_state in ['equilibrium', 'advancing', 'retreating']:
        # convert COMBINE MassBalanceModel to OGGM MassBalanceModel
        oggm_mb_model = oggm_MassBalance(to_numpy_array(mb_model.ela_h),
                                         grad=to_numpy_array(mb_model.grad))

        # create a model and let it run
        model = oggm_FluxModel(oggm_fl, mb_model=oggm_mb_model, y0=0.)
        model.run_until(measurements['yrs_to_run'])

        # get values of flowline
        fg_run['sfc_h'] = model.fls[0].surface_h
        fg_run['widths'] = model.fls[0].widths_m
    elif glacier_state == 'retreating with unknow spinup':
        # create mb model for speenup
        spinup_mb_model = oggm_MassBalance(first_guess['spinup_ELA'],
                                           grad=mb_model['grad_spinup'])

        # run the model for the spinup
        spinup_model = oggm_FluxModel(oggm_fl,
                                      mb_model=spinup_mb_model,
                                      y0=0.)
        spinup_model.run_until(spinup_yrs)

        # safe the first guess spinup sfc and widths
        fg_run['spinup_sfc_h'] = spinup_model.fls[0].surface_h
        fg_run['spinup_widths'] = spinup_model.fls[0].widths_m

        # continue model run with known mass balance
        known_mb_model = oggm_MassBalance(
            to_numpy_array(mb_model['model_known'].ela_h),
            grad=to_numpy_array(mb_model['model_known'].grad))

        final_model = oggm_FluxModel(spinup_model.fls[0],
                                     mb_model=known_mb_model,
                                     y0=0.)
        final_model.run_until(measurements['yrs_to_run'])

        # safe final outcome of first guess run
        fg_run['sfc_h'] = final_model.fls[0].surface_h
        fg_run['widths'] = final_model.fls[0].widths_m

    return fg_run
