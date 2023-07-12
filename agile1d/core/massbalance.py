"""Mass-balance models"""
# Locals
import oggm.cfg as cfg
from oggm.cfg import SEC_IN_YEAR
from oggm.core.massbalance import ConstantMassBalance
from oggm.utils import SuperclassMeta, ncDataset
from oggm.utils._funcs import date_to_floatyear, floatyear_to_date

# packages for agile
import torch

# help functions
from agile1d.core.torch_interp1d import Interp1d

# other stuff
from functools import partial
import numpy as np
from itertools import tee


class MassBalanceModel(object, metaclass=SuperclassMeta):
    """Common logic for the mass balance models.
    All mass-balance models should implement this interface.
    Attributes
    ----------
    valid_bounds : [float, float]
        The altitudinal bounds where the MassBalanceModel is valid. This is
        necessary for automated ELA search.
    """

    def __init__(self, torch_type=torch.double, device='cpu'):
        """ Initialize."""
        self.valid_bounds = None
        self.hemisphere = None
        self.torch_type = torch_type
        self.device = device
        self.rho = torch.tensor(cfg.PARAMS['ice_density'],
                                dtype=self.torch_type,
                                device=self.device,
                                requires_grad=False)

    def get_monthly_mb(self, heights, year=None, fl_id=None, fls=None):
        """Monthly mass-balance at given altitude(s) for a moment in time.
        Units: [m s-1], or meters of ice per second
        Note: `year` is optional because some simpler models have no time
        component.
        Parameters
        ----------
        heights: ndarray
            the atitudes at which the mass-balance will be computed
        year: float, optional
            the time (in the "hydrological floating year" convention)
        fl_id: float, optional
            the index of the flowline in the fls array (might be ignored
            by some MB models)
        fls: list of flowline instances, optional
            the flowlines array, in case the MB model implementation needs
            to know details about the glacier geometry at the moment the
            MB model is called
        Returns
        -------
        the mass-balance (same dim as `heights`) (units: [m s-1])
        """
        raise NotImplementedError()

    def get_annual_mb(self, heights, year=None, fl_id=None, fls=None):
        """Like `self.get_monthly_mb()`, but for annual MB.
        For some simpler mass-balance models ``get_monthly_mb()` and
        `get_annual_mb()`` can be equivalent.
        Units: [m s-1], or meters of ice per second
        Note: `year` is optional because some simpler models have no time
        component.
        Parameters
        ----------
        heights: ndarray
            the atitudes at which the mass-balance will be computed
        year: float, optional
            the time (in the "floating year" convention)
        fl_id: float, optional
            the index of the flowline in the fls array (might be ignored
            by some MB models)
        fls: list of flowline instances, optional
            the flowlines array, in case the MB model implementation needs
            to know details about the glacier geometry at the moment the
            MB model is called
        Returns
        -------
        the mass-balance (same dim as `heights`) (units: [m s-1])
        """
        raise NotImplementedError()


class LinearMassBalance(MassBalanceModel):
    """Constant mass-balance as a linear function of altitude.
    """

    def __init__(self, ela_h, grad=3., max_mb=None, torch_type=torch.double, device='cpu'):
        """ Initialize.
        Parameters
        ----------
        ela_h: float
            Equilibrium line altitude (units: [m])
        grad: float
            Mass-balance gradient (unit: [mm w.e. yr-1 m-1])
        max_mb: float
            Cap the mass balance to a certain value (unit: [mm w.e. yr-1])
        Attributes
        ----------
        temp_bias : float, default 0
            A "temperature bias" doesn't makes much sense in the linear MB
            context, but we implemented a simple empirical rule:
            + 1K -> ELA + 150 m
        """
        super(LinearMassBalance, self).__init__(torch_type=torch_type, device=device)
        self.hemisphere = 'nh'
        self.valid_bounds = [-1e4, 2e4]  # in m
        self.orig_ela_h = torch.tensor(ela_h,
                                       dtype=self.torch_type,
                                       device=self.device,
                                       requires_grad=False)
        self.ela_h = torch.tensor(ela_h,
                                  dtype=self.torch_type,
                                  device=self.device,
                                  requires_grad=False)
        self.grad = torch.tensor(grad,
                                 dtype=self.torch_type,
                                 device=self.device,
                                 requires_grad=False)
        self.max_mb = torch.tensor(max_mb,
                                   dtype=self.torch_type,
                                   device=self.device,
                                   requires_grad=False)
        self._temp_bias = torch.tensor(0,
                                       dtype=self.torch_type,
                                       device=self.device,
                                       requires_grad=False)
        self.SEC_IN_YEAR = torch.tensor(SEC_IN_YEAR,
                                        dtype=self.torch_type,
                                        device=self.device,
                                        requires_grad=False)

    @property
    def temp_bias(self):
        """Temperature bias to add to the original series."""
        return self._temp_bias

    @temp_bias.setter
    def temp_bias(self, value):
        """Temperature bias to change the ELA."""
        value = torch.tensor(value,
                             dtype=self.torch_type,
                             device=self.device,
                             requires_grad=False)
        self.ela_h = self.orig_ela_h + value * torch.tensor(150,
                                                            dtype=self.torch_type,
                                                            device=self.device,
                                                            requires_grad=False)
        self._temp_bias = value

    def get_monthly_mb(self, heights, **kwargs):
        heights = torch.tensor(heights,
                               dtype=self.torch_type,
                               device=self.device,
                               requires_grad=False)
        mb = (heights - self.ela_h) * self.grad
        if self.max_mb is not None:
            torch.clamp(mb, max=self.max_mb)
        return mb / self.SEC_IN_YEAR / self.rho

    def get_annual_mb(self, heights, **kwargs):
        return self.get_monthly_mb(heights, **kwargs)


class ConstantMassBalanceTorch(ConstantMassBalance):
    """Constant mass-balance during a chosen period.
    This is useful for equilibrium experiments.
    This class is adapted from OGGMs ConstantMassBalance for the use of PyTorch
    """

    def __init__(self, gdir, melt_f=None, temp_bias=None, prcp_fac=None,
                 bias=0, y0=None, halfsize=15, height_shift=0.,
                 filename='climate_historical', input_filesuffix='',
                 torch_type=torch.double, device='cpu', **kwargs):
        """Initialize
        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        melt_f : float, optional
            set to the value of the melt factor you want to use,
            here the unit is kg m-2 day-1 K-1
            (the default is to use the calibrated value).
        temp_bias : float, optional
            set to the value of the temperature bias you want to use
            (the default is to use the calibrated value).
        prcp_fac : float, optional
            set to the value of the precipitation factor you want to use
            (the default is to use the calibrated value).
        bias : float, optional
            set to the alternative value of the annual bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
        y0 : int, optional, default: tstar
            the year at the center of the period of interest. The default
            is to use tstar as center.
        halfsize : int, optional
            the half-size of the time window (window size = 2 * halfsize + 1)
        height_shift : float or :py:class:`torch.Tensor`, optional
            for vertically shifting the MassBalance profile, positive values
            shift the profile upwards, negative values downwards
        filename : str, optional
            set to a different BASENAME if you want to use alternative climate
            data.
        input_filesuffix : str
            the file suffix of the input climate file
        """

        super(ConstantMassBalanceTorch, self).__init__(gdir, melt_f=melt_f,
                                                       temp_bias=temp_bias,
                                                       prcp_fac=prcp_fac,
                                                       bias=bias, y0=y0,
                                                       halfsize=halfsize,
                                                       filename=filename,
                                                       input_filesuffix=input_filesuffix,
                                                       **kwargs)

        self.torch_type = torch_type
        self.device = device

        if type(height_shift) != torch.Tensor:
            self.height_shift = torch.tensor(height_shift,
                                             dtype=self.torch_type,
                                             device=self.device,
                                             requires_grad=False)
        else:
            self.height_shift = height_shift

        self._get_annual_mb = None
        self._get_monthly_mb = None
        self.initialize_get_annual_mb()
        self.initialize_get_monthly_mb()

    def initialize_get_annual_mb(self):
        mb_on_h = self.hbins * 0.
        for yr in self.years:
            mb_on_h += self.mbmod.get_annual_mb(self.hbins, year=yr)

        def interp1d_wrapper(x, y):
            def out(xnew):
                return Interp1d()(x, y, xnew)

            return out

        self._get_annual_mb = interp1d_wrapper(x=torch.tensor(self.hbins,
                                                              dtype=self.torch_type,
                                                              device=self.device,
                                                              requires_grad=False),
                                               y=torch.tensor(mb_on_h / len(self.years),
                                                              dtype=self.torch_type,
                                                              device=self.device,
                                                              requires_grad=False))

    def get_annual_mb(self, heights, year=None, add_climate=False, **kwargs):
        return torch.squeeze(self._get_annual_mb(heights - self.height_shift))

    def initialize_get_monthly_mb(self):
        # monthly MB
        months = np.arange(12) + 1
        _get_monthly_mb = []

        def interp1d_wrapper(x, y):
            def out(xnew):
                return Interp1d()(x, y, xnew)

            return out

        for m in months:
            mb_on_h = self.hbins * 0.
            for yr in self.years:
                yr = date_to_floatyear(yr, m)
                mb_on_h += self.mbmod.get_monthly_mb(self.hbins, year=yr)
            _get_monthly_mb.append(interp1d_wrapper(
                x=torch.tensor(self.hbins,
                               dtype=self.torch_type,
                               device=self.device,
                               requires_grad=False),
                y=torch.tensor(mb_on_h / len(self.years),
                               dtype=self.torch_type,
                               device=self.device,
                               requires_grad=False)))
        self._get_monthly_mb = _get_monthly_mb

    def get_monthly_mb(self, heights, year=None, add_climate=False, **kwargs):
        yr, m = floatyear_to_date(year)
        return torch.squeeze(self._get_monthly_mb[m - 1](heights -
                                                         self.height_shift))


class MBModelTorchWrapper:

    def __init__(self, gdir, mb_model,
                 torch_type=torch.double, device='cpu'):

        self.mbmod = mb_model
        self.torch_type = torch_type
        self.device = device

        self.hemisphere = self.mbmod.hemisphere

        # This is a quick'n dirty optimisation
        try:
            fls = gdir.read_pickle('model_flowlines')
            h = []
            for fl in fls:
                # We use bed because of overdeepenings
                h = np.append(h, fl.bed_h)
                h = np.append(h, fl.surface_h)
            zminmax = np.round([np.min(h) - 50, np.max(h) + 2000])
        except FileNotFoundError:
            # in case we don't have them
            with ncDataset(gdir.get_filepath('gridded_data')) as nc:
                if np.isfinite(nc.min_h_dem):
                    # a bug sometimes led to non-finite
                    zminmax = [nc.min_h_dem - 250, nc.max_h_dem + 1500]
                else:
                    zminmax = [nc.min_h_glacier - 1250, nc.max_h_glacier + 1500]

        self.hbins = np.arange(*zminmax, step=20)
        self.valid_bounds = self.hbins[[0, -1]]

    def get_annual_mb(self, heights, year=None, fl_id=None, fls=None):
        if isinstance(year, torch.Tensor):
            year = year.detach().to('cpu').numpy().astype(np.float64)

        annual_mb = self.mbmod.get_annual_mb(heights=self.hbins, year=year,
                                             fl_id=fl_id, fls=fls)

        return torch.squeeze(Interp1d()(torch.tensor(self.hbins,
                                                     dtype=self.torch_type,
                                                     device=self.device,
                                                     requires_grad=False),
                                        torch.tensor(annual_mb,
                                                     dtype=self.torch_type,
                                                     device=self.device,
                                                     requires_grad=False),
                                        heights))

    def get_monthly_mb(self, heights, year=None, fl_id=None, fls=None):
        if isinstance(year, torch.Tensor):
            year = year.detach().to('cpu').numpy().astype(np.float64)

        monthly_mb = self.mbmod.get_monthly_mb(heights=self.hbins, year=year,
                                               fl_id=fl_id, fls=fls)

        return torch.squeeze(Interp1d()(torch.tensor(self.hbins,
                                                     dtype=self.torch_type,
                                                     device=self.device,
                                                     requires_grad=False),
                                        torch.tensor(monthly_mb,
                                                     dtype=self.torch_type,
                                                     device=self.device,
                                                     requires_grad=False),
                                        heights))


def pairwise(iterable):
    """ In Python 3.10 this is available in itertools.pairwise"""
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class StackedMassBalance(MassBalanceModel):
    """Define different MassBalanceModels for different time periods.
    """

    def __init__(self, gdir, mb_model_settings, filename='climate_historical',
                 input_filesuffix=''):

        super(StackedMassBalance, self).__init__()

        self.hemisphere = gdir.hemisphere

        # set periods of different mb_models
        all_periods = []
        for key_mb in mb_model_settings:
            all_periods.append(mb_model_settings[key_mb]['years'])
        self._periods = np.sort(np.unique(np.concatenate(all_periods)))

        # set mb_model for all specific sorted periods
        mb_models = {}
        for p_nr, (p_start, p_end) in enumerate(pairwise(self._periods)):
            # check if their is a mb_model defined for the current period
            found_mb_model = False
            for key_mb in mb_model_settings:
                if np.all([p_start, p_end] ==
                          mb_model_settings[key_mb]['years']):
                    if mb_model_settings[key_mb]['type'] == 'constant':
                        halfsize_run = (p_end - p_start) / 2
                        mb_models[p_nr] = ConstantMassBalance(
                            gdir, y0=p_start + halfsize_run,
                            halfsize=halfsize_run, filename=filename,
                            input_filesuffix=input_filesuffix)
                        found_mb_model = True
                        break
                    else:
                        raise NotImplementedError('')

            if not found_mb_model:
                raise ValueError(f'No mb model defined for period {p_start} '
                                 f'to {p_end}!')

        self._mb_models = mb_models

    def get_period_nr(self, year):
        current_period = np.searchsorted(self._periods, year, side='right')

        # the given year is smaller than minimum defined in periods
        if current_period == 0:
            raise ValueError(f'No mb model defined for year {year}')

        if current_period >= len(self._periods):
            if year == self._periods[-1]:
                # ok at the upper limit we use again the last mb_model
                current_period -= 1
            else:
                # the given year is larger than the maximum defined in periods
                raise ValueError(f'No mb model defined for year {year}')

        return current_period - 1  # because mb_model index starts with 0

    def get_monthly_mb(self, heights, year=None, fl_id=None, fls=None):
        return self._mb_models[self.get_period_nr(year)].get_monthly_mb(
            heights=heights, year=year, fl_id=fl_id, fls=fls)

    def get_annual_mb(self, heights, year=None, fl_id=None, fls=None):
        return self._mb_models[self.get_period_nr(year)].get_annual_mb(
            heights=heights, year=year, fl_id=fl_id, fls=fls)
