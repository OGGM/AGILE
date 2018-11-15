import numpy as np
from numpy import ix_
import xarray as xr
import torch
torch.utils.backcompat.broadcast_warning.enabled = True
import os

from oggm import cfg, utils
from oggm.cfg import G, SEC_IN_YEAR, SEC_IN_DAY, SEC_IN_MONTH

# BIG TODO: re-check entire model

def filter_ice_border(ice_thick):
    """Sets the ice thickness at the border of the domain to zero."""
    ice_thick[0, :] = 0
    ice_thick[-1, :] = 0
    ice_thick[:, 0] = 0
    ice_thick[:, -1] = 0
    return ice_thick


class Model2D(object):
    """Interface to a distributed model"""

    def __init__(self, bed_topo, init_ice_thick=None, dx=None, dy=None,
                 mb_model=None, y0=0., glen_a=None, mb_elev_feedback='annual',
                 ice_thick_filter=filter_ice_border):
        """Create a new 2D model from gridded data.

        Parameters
        ----------
        bed_topo : 2d array
            the topography
        init_ice_thick : 2d array (optional)
            the initial ice thickness (default is zero everywhere)
        dx : float
            map resolution (m)
        dy : float
            map resolution (m)
        mb_model : oggm.core.massbalance model
            the mass-balance model to use for the simulation
        y0 : int
            the starting year
        glen_a : float
            Glen's flow law parameter A
        mb_elev_feedback : str (default: 'annual')
            when to update the mass-balance model ('annual', 'monthly', or
            'always')
        ice_thick_filter : func
            function to apply to the ice thickness *after* each time step.
            See filter_ice_border for an example. Set to None for doing nothing
        """

        # Mass balance
        self.mb_model = mb_model
        self.mb_elev_feedback = mb_elev_feedback
        _mb_call = None
        if mb_model:
            if mb_elev_feedback == 'always':
                _mb_call = mb_model.get_monthly_mb
            elif mb_elev_feedback == 'monthly':
                _mb_call = mb_model.get_monthly_mb
            elif mb_elev_feedback == 'annual':
                _mb_call = mb_model.get_annual_mb
            else:
                raise ValueError('mb_elev_feedback not understood')
        self._mb_call = _mb_call
        self._mb_current_date = None
        self._mb_current_out = None

        # Defaults
        if glen_a is None:
            glen_a = cfg.PARAMS['glen_a']
        self.glen_a = glen_a

        if dy is None:
            dy = dx

        self.dx = torch.tensor(dx, dtype=torch.float, requires_grad=False)
        self.dy = torch.tensor(dy, dtype=torch.float, requires_grad=False)
        self.dxdy = dx * dy

        self.y0 = None
        self.t = None
        self.reset_y0(y0)

        self.ice_thick_filter = ice_thick_filter

        # Data
        self.bed_topo = bed_topo
        self.ice_thick = None
        self.reset_ice_thick(init_ice_thick)
        self.ny, self.nx = bed_topo.shape
        self.RHO = cfg.PARAMS['ice_density']
        self.N = cfg.PARAMS['glen_n']

    def reset_y0(self, y0):
        """Reset the initial model time"""
        self.y0 = y0
        self.t = torch.tensor(0., dtype=torch.float, requires_grad=False)

    def reset_ice_thick(self, ice_thick=None):
        """Reset the ice thickness"""
        if ice_thick is None:
            ice_thick = torch.zeros(self.bed_topo.shape)
        self.ice_thick = ice_thick.clone()
        #self.ice_thick.requires_grad = False

    @property
    def yr(self):
        return self.y0 + self.t / SEC_IN_YEAR

    @property
    def area_m2(self):
        return torch.sum(self.ice_thick > 0) * self.dxdy

    @property
    def volume_m3(self):
        return torch.sum(self.ice_thick * self.dxdy)

    @property
    def volume_km3(self):
        return self.volume_m3 * 1e-9

    @property
    def area_km2(self):
        return self.area_m2 * 1e-6

    @property
    def surface_h(self):
        return self.bed_topo + self.ice_thick

    def get_mb(self, year=None):
        """Get the mass balance at the requested height and time.

        Optimized so that no mb model call is necessary at each step.
        """

        with torch.no_grad():
            if year is None:
                year = self.yr

            # Do we have to optimise?
            if self.mb_elev_feedback == 'always':
                return self._mb_call(self.bed_topo + self.ice_thick, year)

            date = floatyear_to_date(year)
            if self.mb_elev_feedback == 'annual':
                # ignore month changes
                date = (date[0], date[0])

            if self._mb_current_date != date or (self._mb_current_out is None):
                # We need to reset all
                self._mb_current_date = date
                _mb = self._mb_call(self.surface_h.view(self.surface_h.numel()),
                                                        year)
                self._mb_current_out = _mb.view((self.ny, self.nx))

        return self._mb_current_out

    def step(self, dt):
        """Advance one step."""
        raise NotImplementedError

    def run_until(self, y1, stop_if_border=False):
        """Run until a selected year."""

        t = (y1 - self.y0) * SEC_IN_YEAR
        while self.t < t:
            self.step(t - self.t)
            #if stop_if_border:
            #    if (torch.any(self.ice_thick[0, :] > 10) or
            #            torch.any(self.ice_thick[-1, :] > 10) or
            #            torch.any(self.ice_thick[:, 0] > 10) or
            #            torch.any(self.ice_thick[:, -1] > 10)):
            #        raise RuntimeError('Glacier exceeds boundaries')
            #if self.ice_thick_filter is not None:
            #    self.ice_thick = self.ice_thick_filter(self.ice_thick)

        #if torch.any(~torch.isfinite(self.ice_thick)):
        #    raise FloatingPointError('NaN in numerical solution.')

    def run_until_equilibrium(self, rate=0.001, ystep=5, max_ite=200):
        """Run until an equilibrium is reached (can take a while)."""

        ite = 0
        was_close_zero = 0
        t_rate = 1
        while (t_rate > rate) and (ite <= max_ite) and (was_close_zero < 5):
            ite += 1
            v_bef = self.volume_m3
            self.run_until(self.yr + ystep)
            v_af = self.volume_m3
            if torch.isclose(v_bef, 0., atol=1):
                t_rate = 1
                was_close_zero += 1
            else:
                t_rate = torch.abs(v_af - v_bef) / v_bef
        if ite > max_ite:
            raise RuntimeError('Did not find equilibrium.')

    def run_until_and_store(self, ye, step=2, run_path=None, grid=None,
                            print_stdout=False, stop_if_border=False):
        """Run until a selected year and store the output in a NetCDF file."""

        yrs = torch.arange(torch.floor(self.yr), torch.floor(ye) + 1, step)
        out_thick = torch.zeros((len(yrs), self.ny, self.nx))
        for i, yr in enumerate(yrs):
            if print_stdout and (yr / 10) == int(yr / 10):
                print('{}: year {} of {}, '
                      'max thick {:.1f}m\r'.format(print_stdout,
                                                   int(yr),
                                                   int(ye),
                                                   self.ice_thick.max()))
            self.run_until(yr, stop_if_border=stop_if_border)
            out_thick[i, :, :] = self.ice_thick

        run_ds = grid.to_dataset() if grid else xr.Dataset()
        run_ds['ice_thickness'] = xr.DataArray(out_thick,
                                               dims=['time', 'y', 'x'],
                                               coords={'time': yrs})

        run_ds['bed_topo'] = xr.DataArray(self.bed_topo,
                                          dims=['y', 'x'])

        # write output?
        if run_path is not None:
            if os.path.exists(run_path):
                os.remove(run_path)
            run_ds.to_netcdf(run_path)

        return run_ds


class Upstream2D(Model2D):
    """Actual model"""

    def __init__(self, bed_topo, init_ice_thick=None, dx=None,
                 mb_model=None, y0=0., glen_a=None, mb_elev_feedback='annual',
                 cfl=0.124, max_dt=31*SEC_IN_DAY,
                 ice_thick_filter=filter_ice_border):
        """Create a new 2D model from gridded data.

        Parameters
        ----------
        bed_topo : 2d array
            the topography
        init_ice_thick : 2d array (optional)
            the initial ice thickness (default is zero everywhere)
        dx : float
            map resolution (m)
        dy : float
            map resolution (m)
        mb_model : oggm.core.massbalance model
            the mass-balance model to use for the simulation
        y0 : int
            the starting year
        glen_a : float
            Glen's flow law parameter A
        mb_elev_feedback : str (default: 'annual')
            when to update the mass-balance model ('annual', 'monthly', or
            'always')
        cfl : float (default:0.124)
            forward time stepping stability criteria. Default is just beyond
            R. Hindmarsh's idea of 1/2(n+1).
        max_dt : int (default: 31 days)
            maximum allow time step (in seconds). Useful because otherwise the
            automatic time step can be quite ambitious.
        ice_thick_filter : func
            function to apply to the ice thickness *after* each time step.
            See filter_ice_border for an example. Set to None for doing nothing
        """
        super(Upstream2D, self).__init__(bed_topo,
                                         init_ice_thick=init_ice_thick,
                                         dx=dx, mb_model=mb_model, y0=y0,
                                         glen_a=glen_a,
                                         mb_elev_feedback=mb_elev_feedback,
                                         ice_thick_filter=ice_thick_filter)

        # We introduce Gamma to shorten the equations
        self.gamma = self.glen_a * (self.RHO * G) ** self.N / (self.N + 2) * \
                     torch.tensor(2., dtype=torch.float)

        # forward time stepping stability criteria
        # default is just beyond R. Hindmarsh's idea of 1/2(n+1)

        self.cfl = torch.tensor(cfl, dtype=torch.float)
        self.max_dt = torch.tensor(max_dt, dtype=torch.float)

        # extend into 2D
        self.Lx = 0.5 * (self.nx - 1) * self.dx
        self.Ly = 0.5 * (self.ny - 1) * self.dy

    def diffusion_upstream_2d(self):
        # Builded upon the Eq. (62) with the term in y in the diffusivity.
        # It differs from diffusion_Upstream_2D_V1 only for the definition of
        # "s_grad", l282-283 & l305-306 in V1 and l355-356 & l379-380 in V2)
        #with torch.no_grad():
        H = self.ice_thick
        S = self.surface_h

        # Optim
        S_ixklp = S[1:self.ny-1, 2:self.nx]
        S_ixkl = S[1:self.ny-1, 1:self.nx-1]
        S_ixklm = S[1:self.ny-1, 0:self.nx-2]
        S_ixkml = S[0:self.ny-2, 1:self.nx-1]
        S_ixkpl = S[2:self.ny, 1:self.nx-1]
        S_ixkplp = S[2:self.ny, 2:self.nx]
        S_ixkplm = S[2:self.ny, 0:self.nx-2]
        S_ixkmlm = S[0:self.ny-2, 0:self.nx-2]
        S_ixkmlp = S[0:self.ny-2, 2:self.nx]

        Hl = H[1:self.ny-1, 1:self.nx-1]
        Hlp = H[1:self.ny-1, 2:self.nx]
        Hlm = H[1:self.ny-1, 0:self.nx-2]
        Hk = Hl
        Hkp = H[2:self.ny, 1:self.nx-1]
        Hkm = H[0:self.ny-2, 1:self.nx-1]

        # --- all the l components

        # applying Eq. (61) to the scheme
        H_l_up = 0.5 * (Hlp + Hl)
        H_l_dn = 0.5 * (Hl + Hlm)

        # applying Eq. (62) to the scheme
        S_diff = S_ixkpl - S_ixkml
        S_lpdiff = S_ixklp - S_ixkl
        S_lmdiff = S_ixkl - S_ixklm
        H_l_upstream_up = torch.where(S_lpdiff > 0, Hlp, Hl)
        H_l_upstream_dn = torch.where(S_lmdiff > 0, Hl, Hlm)

        s_l_grad_up = (((S_diff + S_ixkplp - S_ixkmlp)
                        ** 2. / (4 * self.dx) ** 2.) +
                       (S_lpdiff ** 2. / self.dy ** 2.)) ** ((self.N - 1.) / 2.)
        s_l_grad_dn = (((S_diff + S_ixkplm - S_ixkmlm)
                        ** 2. / (4 * self.dx) ** 2.) +
                       (S_lmdiff ** 2. / self.dy ** 2.)) ** ((self.N - 1.) / 2.)

        D_l_up = self.gamma * H_l_up ** (self.N + 1) * H_l_upstream_up * s_l_grad_up
        D_l_dn = self.gamma * H_l_dn ** (self.N + 1) * H_l_upstream_dn * s_l_grad_dn

        # --- all the k components

        # applying Eq. (61) to the scheme
        H_k_up = 0.5 * (Hkp + Hl)
        H_k_dn = 0.5 * (Hl + Hkm)

        # applying Eq. (62) to the scheme
        S_diff = S_ixklp - S_ixklm
        S_kpdiff = S_ixkpl - S_ixkl
        S_kmdiff = S_ixkl - S_ixkml

        H_k_upstream_up = torch.where(S_kpdiff > 0, Hkp, Hk)
        H_k_upstream_dn = torch.where(S_kmdiff > 0, Hk, Hkm)

        s_k_grad_up = (((S_diff + S_ixkplp - S_ixkplm)
                        ** 2. / (4 * self.dy) ** 2.) +
                       (S_kpdiff ** 2. / self.dx ** 2.)) ** ((self.N - 1.) / 2.)
        s_k_grad_dn = (((S_diff + S_ixkmlp - S_ixkmlm)
                        ** 2. / (4 * self.dy) ** 2.) +
                       (S_kmdiff ** 2. / self.dx ** 2.)) ** ((self.N - 1.) / 2.)

        D_k_up = self.gamma * H_k_up ** (self.N + 1) * H_k_upstream_up * s_k_grad_up
        D_k_dn = self.gamma * H_k_dn ** (self.N + 1) * H_k_upstream_dn * s_k_grad_dn

        # --- Check the cfl condition
        divisor = torch.max(torch.max(torch.max(torch.abs(D_k_up)),
                                      torch.max(torch.abs(D_k_dn))),
                            torch.max(torch.max(torch.abs(D_l_up)),
                                      torch.max(torch.abs(D_l_dn))))
        if divisor == 0:
            dt_cfl = self.max_dt
         # TODO: check when this happens -> raise exception?
        else:
            dt_cfl = (self.cfl * torch.min(self.dx ** 2., self.dy ** 2.) /
                      divisor)

        # --- Calculate Final diffusion term
        div_k = (D_k_up * S_kpdiff / self.dy -
                 D_k_dn * S_kmdiff / self.dy) / self.dy
        div_l = (D_l_up * S_lpdiff / self.dx -
                 D_l_dn * S_lmdiff / self.dx) / self.dx

        return div_l + div_k, dt_cfl

    def step(self, dt):
        """Advance one step."""

        div_q, dt_cfl = self.diffusion_upstream_2d()

        dt_use = torch.clamp(torch.min(torch.tensor([dt_cfl, dt],
                                                    dtype=torch.float)),
                             0., self.max_dt)
        # TODO: track for memory leak
        # do not allow for less than a tenth of 'usual' time stepping to avoid
        # memory overflow (restrict it to twenty times minimal ...
        if dt_cfl != dt and dt_cfl / self.max_dt < 0.001:
            print('problem ahead')
        if dt_cfl != dt and dt_cfl / self.max_dt < 0.001:
            raise MemoryError('Stopping dynamics run to avoid memory overflow')

        self.ice_thick[1:-1, 1:-1] = torch.clamp(
            self.surface_h[1:-1, 1:-1] +
            (self.get_mb()[1:-1, 1:-1] +
             div_q) * dt_use -
            self.bed_topo[1:-1, 1:-1],
            min=0)

        # Next step
        self.t = self.t + dt_use
        return dt

# @torch.no_grad()
def floatyear_to_date(yr):
    """Converts a float year to an actual (year, month) pair.

    Note that this doesn't account for leap years (365-day no leap calendar),
    and that the months all have the same length.

    Parameters
    ----------
    yr : float
        The floating year
    """

    #try:
    sec = torch.fmod(yr, 1.)
    out_y = (yr - sec).type(torch.IntTensor)
    sec = torch.floor(sec * SEC_IN_YEAR)
    if sec == SEC_IN_YEAR:
        # Floating errors
        out_y += 1
        sec = torch.tensor(0.)
    out_m = (sec / SEC_IN_MONTH).type(torch.IntTensor) + 1
    return out_y, out_m