from oggm.core.massbalance import MassBalanceModel
from oggm import cfg
from oggm.cfg import SEC_IN_YEAR


class ClippedLinearMassBalance(MassBalanceModel):
    """Constant mass-balance as a linear function of altitude. Mass balance
    values above and below a given altitude can be clipped
    """

    def __init__(self, ela_h, grad=3., max_mb_alt=None, min_mb_alt=None):
        """ Initialize.

        Parameters
        ----------
        ela_h: float
            Equilibrium line altitude (units: [m])
        grad: float
            Mass-balance gradient (unit: [mm w.e. yr-1 m-1])
        max_mb_alt: float
            Mass-balance above this altitude is capped to the mass-balance
            value corresponding to this altitude (unit: [m])
        min_mb_alt: float
            Mass-balance below this altitude is capped to the mass-balance
            value corresponding to this altitude (unit: [m])
        """
        super(ClippedLinearMassBalance, self).__init__()
        self.valid_bounds = [-1e4, 2e4]  # in m
        self.orig_ela_h = ela_h
        self.ela_h = ela_h
        self.grad = grad
        self.max_mb_alt = max_mb_alt
        self.min_mb_alt = min_mb_alt
        self.rho = cfg.PARAMS['ice_density']

    def get_monthly_mb(self, heights, year=None):
        """
        Computes the monthly mass-balance

        Parameters
        ----------
        heights: FloatTensor
            Heights for which the mass-balance should be computed and
            returned (unit: [m])
        year:
            Year for which the mass-balance should be computed

        Returns
        -------
        the mass-balance (same dim as `heights`) (units: [m s-1])
        """
        mb = (heights - self.ela_h) * self.grad
        if self.max_mb_alt is not None:
            mb = mb.clip(None, (self.max_mb_alt - self.ela_h) * self.grad)
        if self.min_mb_alt is not None:
            mb = mb.clip((self.min_mb_alt - self.ela_h) * self.grad, None)
        return mb / SEC_IN_YEAR / self.rho

    def get_annual_mb(self, heights, year=None):
        """
        Computes the annual mass-balance

        Parameters
        ----------
        heights: FloatTensor
           Heights for which the mass-balance should be computed and
           returned (unit: [m])
        year:
           Year for which the mass-balance should be computed

        Returns
        -------
        the mass-balance (same dim as `heights`) (units: [m s-1])
        """
        return self.get_monthly_mb(heights, year=year)