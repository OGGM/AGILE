import collections
import torch
import numpy as np
from combine1d.core.flowline_adapted import ParabolicBedFlowline
from combine1d.core.flowline_adapted import RectangularBedFlowline
from combine1d.core.flowline_adapted import TrapezoidalBedFlowline
from combine1d.core.flowline_adapted import FluxBasedModel


def run_flowline_forward_core(bed_h,
                              shape_var,
                              bed_geometry,
                              mb_model,
                              spinup_sfc_h,
                              spinup_sfc_known,
                              yrs_to_run,
                              spinup_yrs,
                              map_dx,
                              torch_type):
    '''
    Performs the forward run of the Flowlinemodel

    Parameters
    ----------
    bed_h : :py:class:`torch.Tensor`
        Height of the glacier bed.
    shape_var : :py:class:`torch.Tensor`
        Second variable describing the glacier bed. The meaning depends on the
        'bed_geometry'. ('rectangular' = width, 'parabolic' = shape_factor,
                         'trapezoidal' = bottom width)
    bed_geometry : str
        Defines the bed shape.
        Options: 'rectangular', 'parabolic' or 'trapezoidal'
    mb_model : :py:class:`oggm.core.massbalance.MassBalanceModel`
        The mass balance model to use.
    spinup_sfc_h : :py:class:`numpy.ndarray`
        Surface height of the glacier at the start of the model run.
    yrs_to_run : float
        Years for how long the glacier should be modeled with the given mass
        balance model.
    map_dx : float
        Model grid spacing in meters.
    torch_type : :py:class:`torch.dtype`
        Defines type for torch.Tensor.

    Returns
    -------
    :py:class:`oggm.Flowline`
        The flowline at the end of the model run.

    '''
    # initialise flowline according to bed_geometry
    if bed_geometry == 'rectangular':
        flowline = RectangularBedFlowline(surface_h=spinup_sfc_h,
                                          bed_h=bed_h,
                                          widths_m=shape_var,
                                          map_dx=map_dx,
                                          torch_type=torch_type)
    elif bed_geometry == 'parabolic':
        flowline = ParabolicBedFlowline(surface_h=spinup_sfc_h,
                                        bed_h=bed_h,
                                        bed_shape=shape_var,
                                        map_dx=map_dx,
                                        torch_type=torch_type)
    elif bed_geometry == 'trapezoidal':
        # lambda is set constant 1
        # (see https://docs.oggm.org/en/latest/ice-dynamics.html#trapezoidal)
        lambdas = torch.tensor([1.],
                               dtype=torch_type,
                               requires_grad=False)

        # w0 = torch.clamp(shape_var, min=1.) should be done by bounds
        flowline = TrapezoidalBedFlowline(surface_h=spinup_sfc_h,
                                          bed_h=bed_h,
                                          w0_m=shape_var,
                                          lambdas=lambdas,
                                          map_dx=map_dx,
                                          torch_type=torch_type)
    else:
        raise ValueError('Unknown bed geometry!')

    if spinup_sfc_known:
        final_model = FluxBasedModel(flowline,
                                     mb_model=mb_model,
                                     y0=0.,
                                     fs=0.,
                                     mb_elev_feedback='annual')

        final_model.run_until(yrs_to_run)
    else:
        spinup_model = FluxBasedModel(flowline,
                                      mb_model=mb_model['spinup_mb_model'],
                                      y0=0.,
                                      fs=0.,
                                      mb_elev_feedback='annual')
        spinup_model.run_until(spinup_yrs)

        final_model = FluxBasedModel(spinup_model.fls[0],
                                     mb_model=mb_model['known_mb_model'],
                                     y0=0.)
        final_model.run_until(yrs_to_run)

    return final_model.fls[0]


def run_model_and_get_modeled_obs(flowline, mb_models, Obs):
    '''

    Parameters
    ----------
    flowline: py:class:`oggm.Flowline`
    mb_models: dict
        {'MB1': {'mb_model': lala, 'years': np.array([1950, 2010])}}
    Obs: dict
        {'Area', {'2010': np.array([23])}, 'dh': {'2010-2015': np.array([45])}}

    Returns
    -------
    : dict
        dict with the same structure as Obs
    '''
    # creates a ordered dict with all years and Obs we must save during the forward run
    needed_Obs = {}
    for obs_key in Obs.keys():
        if obs_key == 'dh':
            needed_years = np.array([], dtype=int)
            for years in Obs[obs_key].keys():
                for year in years.split('-'):
                    if int(year) not in needed_years:
                        needed_years = np.append(needed_years, int(year))
            obs_key = 'h'
        else:
            needed_years = np.array([int(k) for k in Obs[obs_key].keys()])

        for year in needed_years:
            if year not in needed_Obs.keys():
                needed_Obs[year] = [obs_key]
            else:
                needed_Obs[year].append(obs_key)

    needed_mdl_Obs = collections.OrderedDict(sorted(needed_Obs.items()))

    # start the actual forward run and get observations from model
    raw_Obs_mdl = {}
    for mb_key in mb_models.keys():
        flux_model = FluxBasedModel(flowline,
                                    mb_model=mb_models[mb_key]['mb_model'],
                                    y0=mb_models[mb_key]['years'][0],
                                    fs=0.,
                                    mb_elev_feedback='annual')

        # years with observations using the same mass balance model
        obs_yrs = [k for k in needed_mdl_Obs.keys()
                   if mb_models[mb_key]['years'][0] < k <= mb_models[mb_key]['years'][1]]
        for obs_yr in obs_yrs:
            # let the model run to the year of observation
            flux_model.run_until(obs_yr)

            # save model counterparts of observation
            for var in needed_Obs[obs_yr]:
                if var not in raw_Obs_mdl.keys():
                    raw_Obs_mdl[var] = {}
                if var == 'Area':
                    raw_Obs_mdl[var][obs_yr] = flux_model.fls[0].area_km2
                if var == 'h':
                    raw_Obs_mdl[var][obs_yr] = flux_model.fls[0].area_km2

        # after getting all obs. run the model to the end of the current mb_model use time
        flux_model.run_until(mb_models[mb_key]['years'][1])

        # save flowline for switching to the next mb_model
        flowline = flux_model.fls[0]

    # postprocessing mdl observations to get to the given observations (calculate delta values)
    out = Obs.copy()
    for var_key in out.keys():
        if var_key == 'Area':
            for year in out[var_key].keys():
                out[var_key][year] = raw_Obs_mdl[var_key][int(year)]
        if var_key == 'dh':
            mdl_key = var_key[1]
            for years in out[var_key].keys():
                y1, y2 = years.split('-')
                out[var_key][years] = (raw_Obs_mdl[mdl_key][int(y2)] -
                                       raw_Obs_mdl[mdl_key][int(y1)])

    return out
