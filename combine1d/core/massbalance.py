"""Mass-balance models"""
# Locals
import oggm.cfg as cfg
from oggm.cfg import SEC_IN_YEAR
from oggm.core.massbalance import ConstantMassBalance
from oggm.utils import SuperclassMeta
from oggm.utils._funcs import date_to_floatyear, floatyear_to_date

# packages for COMBINE
import torch

# help functions
from combine1d.core.torch_interp1d import Interp1d

# other stuff
from functools import partial
import numpy as np


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

    def __init__(self, gdir, mu_star=None, bias=None,
                 y0=None, halfsize=15, filename='climate_historical',
                 input_filesuffix='', torch_type=torch.double, device='cpu', **kwargs):
        """Initialize
        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        mu_star : float, optional
            set to the alternative value of mu* you want to use
            (the default is to use the calibrated value)
        bias : float, optional
            set to the alternative value of the annual bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
        y0 : int, optional, default: tstar
            the year at the center of the period of interest. The default
            is to use tstar as center.
        halfsize : int, optional
            the half-size of the time window (window size = 2 * halfsize + 1)
        filename : str, optional
            set to a different BASENAME if you want to use alternative climate
            data.
        input_filesuffix : str
            the file suffix of the input climate file
        """

        super(ConstantMassBalanceTorch, self).__init__(gdir, mu_star=mu_star, bias=bias,
                                                       y0=y0, halfsize=halfsize, filename=filename,
                                                       input_filesuffix=input_filesuffix, **kwargs)

        self.torch_type = torch_type
        self.device = device
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
                return Interp1d.apply(x, y, xnew)

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
        return torch.squeeze(self._get_annual_mb(heights))

    def initialize_get_monthly_mb(self):
        # monthly MB
        months = np.arange(12) + 1
        _get_monthly_mb = []

        def interp1d_wrapper(x, y):
            def out(xnew):
                return Interp1d.apply(x, y, xnew)
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
        return torch.squeeze(self._get_monthly_mb[m - 1](heights))
