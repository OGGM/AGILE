"""Mass-balance models"""
# Locals
import oggm.cfg as cfg
from oggm.cfg import SEC_IN_YEAR
from oggm.core.massbalance import ConstantMassBalance
from oggm.utils import SuperclassMeta

# packages for COMBINE
import torch

# help functions
from combine1d.core.type_conversions import to_torch_tensor
from combine1d.core.torch_interp1d import Interp1d

# other stuff
from functools import partial

class MassBalanceModel(object, metaclass=SuperclassMeta):
    """Common logic for the mass balance models.
    All mass-balance models should implement this interface.
    Attributes
    ----------
    valid_bounds : [float, float]
        The altitudinal bounds where the MassBalanceModel is valid. This is
        necessary for automated ELA search.
    """

    def __init__(self, torch_type=torch.double):
        """ Initialize."""
        self.valid_bounds = None
        self.hemisphere = None
        self.torch_type = torch_type
        self.rho = to_torch_tensor(cfg.PARAMS['ice_density'], self.torch_type)

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

    def __init__(self, ela_h, grad=3., max_mb=None, torch_type=torch.double):
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
        super(LinearMassBalance, self).__init__(torch_type=torch_type)
        self.hemisphere = 'nh'
        self.valid_bounds = [-1e4, 2e4]  # in m
        self.orig_ela_h = to_torch_tensor(ela_h,
                                          torch_type=self.torch_type)
        self.ela_h = to_torch_tensor(ela_h,
                                     torch_type=self.torch_type)
        self.grad = to_torch_tensor(grad,
                                    torch_type=self.torch_type)
        self.max_mb = to_torch_tensor(max_mb,
                                      torch_type=self.torch_type)
        self._temp_bias = to_torch_tensor(0,
                                          torch_type=self.torch_type)
        self.SEC_IN_YEAR = to_torch_tensor(SEC_IN_YEAR,
                                           torch_type=self.torch_type)

    @property
    def temp_bias(self):
        """Temperature bias to add to the original series."""
        return self._temp_bias

    @temp_bias.setter
    def temp_bias(self, value):
        """Temperature bias to change the ELA."""
        value = to_torch_tensor(value,
                                torch_type=self.torch_type)
        self.ela_h = self.orig_ela_h + value * 150
        self._temp_bias = value

    def get_monthly_mb(self, heights, **kwargs):
        heights = to_torch_tensor(heights,
                                  torch_type=self.torch_type)
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
                 input_filesuffix='', torch_type=torch.double, **kwargs):
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
        self._get_annual_mb = None
        self.initialize_get_annual_mb()

    def initialize_get_annual_mb(self):
        mb_on_h = self.hbins * 0.
        for yr in self.years:
            mb_on_h += self.mbmod.get_annual_mb(self.hbins, year=yr)
        self._get_annual_mb = partial(Interp1d(),
                                      to_torch_tensor(self.hbins,
                                                      torch_type=self.torch_type),
                                      to_torch_tensor(mb_on_h / len(self.years),
                                                      torch_type=self.torch_type))

    def get_annual_mb(self, heights, year=None, add_climate=False, **kwargs):
        return self._get_annual_mb(heights)
