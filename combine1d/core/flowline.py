"""Flowline modelling: bed shapes and model numerics.


"""
# Builtins
from collections import OrderedDict
from time import gmtime, strftime
import warnings

# External libs
import torch
import numpy as np
import shapely.geometry as shpg
import xarray as xr

# Locals
from oggm import __version__
import oggm.cfg as cfg
from oggm import utils
from oggm.exceptions import InvalidParamsError
from oggm.core.centerlines import Centerline

# help function for type checking
from combine1d.core.type_conversions import to_torch_tensor

# help function for gradient calculation
from combine1d.core.special_gradient_functions import para_width_from_thick
from combine1d.core.special_gradient_functions import para_thick_from_section

# Constants
from oggm.cfg import SEC_IN_DAY, SEC_IN_YEAR
from oggm.cfg import G, GAUSSIAN_KERNEL


class Flowline(Centerline):
    """Common logic for different types of flowlines used as input to the model


    """

    def __init__(self, line=None, dx=1, map_dx=None,
                 surface_h=None, bed_h=None, rgi_id=None,
                 water_level=None, torch_type=torch.double, device='cpu'):
        """ Initialize a Flowline

        Parameters
        ----------
        line : :py:class:`shapely.geometry.LineString`
            the geometrical line of a :py:class:`oggm.Centerline`
        dx : float
            Grid spacing in pixel coordinates
        map_dx : float
            DEM grid spacing in meters
        surface_h: :py:class:`numpy.ndarray`
            elevation [m] of the flowline grid points
        bed_h: :py:class:`torch.Tensor`
            elevation[m] of the bedrock at the flowline grid points
        rgi_id : str
            The glacier's RGI identifier
        water_level : float
            The water level (to compute volume below sea-level)
        torch_type : torch Data type
            defines the precision of torch tensors
        """

        # This is do add flexibility for testing. I have no time for fancier
        # stuff right now, but I will do this better one day:
        if dx is None:
            dx = 1
        if line is None:
            coords = np.arange(0, len(surface_h) - 0.5, dx)
            line = shpg.LineString(np.vstack([coords, coords * 0.]).T)

        super(Flowline, self).__init__(line, dx, surface_h)

        self.torch_type = torch_type
        self.device = device

        if type(bed_h) != torch.Tensor:
            self.bed_h = torch.tensor(bed_h,
                                      dtype=self.torch_type,
                                      device=device,
                                      requires_grad=False)
        else:
            self.bed_h = bed_h

        if type(surface_h) != torch.Tensor:
            surface_h = torch.tensor(surface_h,
                                     dtype=self.torch_type,
                                     device=device,
                                     requires_grad=False)
        thick = surface_h - self.bed_h
        self.thick = torch.clamp(thick,
                                 min=0)

        self.map_dx = torch.tensor(map_dx,
                                   dtype=self.torch_type,
                                   device=self.device,
                                   requires_grad=False)
        self.dx = torch.tensor(dx,
                               dtype=self.torch_type,
                               device=self.device,
                               requires_grad=False)
        self.dx_meter = self.map_dx * self.dx
        self.rgi_id = rgi_id
        self.water_level = water_level

        # volume not yet removed from the flowline
        self.calving_bucket_m3 = 0

    def has_ice(self):
        return torch.any(self.thick > 0)

    # @Centerline.widths.getter
    # def widths(self):
    #     """Compute the widths out of H and shape"""
    #     return self.widths_m / self.map_dx

    @Centerline.surface_h.getter
    def surface_h(self):
        return self.thick + self.bed_h

    @surface_h.setter
    def surface_h(self, value):
        if type(value) != torch.Tensor:
            raise TypeError('surface_h must be torch.Tensor!!!')
        self.thick = value - self.bed_h

    @property
    def bin_area_m2(self):
        # area of the grid point
        # this takes the ice thickness into account
        return torch.where(self.thick > torch.tensor(0,
                                                     dtype=self.torch_type,
                                                     device=self.device,
                                                     requires_grad=False),
                           self.widths_m, torch.tensor(0,
                                                       dtype=self.torch_type,
                                                       device=self.device,
                                                       requires_grad=False)
                           ) * self.dx_meter

    @property
    def length_m(self):
        # We define the length a bit differently: but more robust
        # TODO: take calving bucket into account
        # pok = torch.where(self.thick > 0.)[0]
        non_zero_points = torch.sum(torch.where(self.thick > 0,
                                                torch.tensor(1),
                                                torch.tensor(0)))
        return non_zero_points * self.dx_meter

    @property
    def volume_m3(self):
        return torch.sum(self.section * self.dx_meter, dim=0, keepdim=True)

    @property
    def volume_km3(self):
        return self.volume_m3 * torch.tensor(1e-9,
                                             dtype=self.torch_type,
                                             device=self.device,
                                             requires_grad=False)

    def _vol_below_level(self, water_level=0):

        thick = np.copy(self.thick)
        n_thick = np.copy(thick)
        bwl = (self.bed_h < water_level) & (thick > 0)
        n_thick[~bwl] = 0
        self.thick = n_thick
        vol_tot = np.sum(self.section * self.dx_meter)
        n_thick[bwl] = utils.clip_max(self.surface_h[bwl],
                                      water_level) - self.bed_h[bwl]
        self.thick = n_thick
        vol_bwl = np.sum(self.section * self.dx_meter)
        self.thick = thick
        fac = vol_bwl / vol_tot if vol_tot > 0 else 0
        return utils.clip_min(vol_bwl -
                              getattr(self, 'calving_bucket_m3', 0) * fac, 0)

    @property
    def volume_bsl_m3(self):
        return self._vol_below_level(water_level=0)

    @property
    def volume_bsl_km3(self):
        return self.volume_bsl_m3 * torch.tensor(1e-9,
                                                 dtype=self.torch_type,
                                                 device=self.device,
                                                 requires_grad=False)

    @property
    def volume_bwl_m3(self):
        return self._vol_below_level(water_level=self.water_level)

    @property
    def volume_bwl_km3(self):
        return self.volume_bwl_m3 * torch.tensor(1e-9,
                                                 dtype=self.torch_type,
                                                 device=self.device,
                                                 requires_grad=False)

    @property
    def area_m2(self):
        # TODO: take calving bucket into account
        return torch.sum(self.bin_area_m2, dim=0, keepdim=True)

    @property
    def area_km2(self):
        return self.area_m2 * torch.tensor(1e-6,
                                           dtype=self.torch_type,
                                           device=self.device,
                                           requires_grad=False)

    def _add_attrs_to_dataset(self, ds):
        """Add bed specific parameters."""
        raise NotImplementedError()

    def to_dataset(self):
        """Makes an xarray Dataset out of the flowline."""

        h = self.surface_h
        nx = len(h)
        ds = xr.Dataset()
        ds.coords['x'] = np.arange(nx)
        ds.coords['c'] = [0, 1]
        ds['linecoords'] = (['x', 'c'], np.asarray(self.line.coords))
        ds['surface_h'] = (['x'], h)
        ds['bed_h'] = (['x'], self.bed_h)
        ds.attrs['class'] = type(self).__name__
        ds.attrs['map_dx'] = self.map_dx
        ds.attrs['dx'] = self.dx
        self._add_attrs_to_dataset(ds)
        return ds


class ParabolicBedFlowline(Flowline):
    """A parabolic shaped Flowline with one degree of freedom
    """

    def __init__(self, line=None, dx=None, map_dx=None,
                 surface_h=None, bed_h=None, bed_shape=None, rgi_id=None,
                 water_level=None, torch_type=torch.float):
        """ Instanciate.

        Parameters
        ----------
        line : :py:class:`shapely.geometry.LineString`
            the geometrical line of a :py:class:`oggm.Centerline`

        Properties
        ----------
        #TODO: document properties
        """
        super(ParabolicBedFlowline, self).__init__(line, dx, map_dx,
                                                   surface_h, bed_h,
                                                   rgi_id=rgi_id,
                                                   water_level=water_level,
                                                   torch_type=torch_type)

        # check type of bed_shape
        if type(bed_shape) != torch.Tensor:
            raise TypeError('bed_shape must be torch.Tensor!!!')
        self.bed_shape = bed_shape

        self.fl_type = 'ParabolicFlowline'

        # checks that the shape is not set infinity during optimisation
        if not torch.all(torch.isfinite(self.bed_shape)):
            print(self.bed_shape)
        assert torch.all(torch.isfinite(self.bed_shape))

        self.spinup_widths = para_width_from_thick.apply(self.bed_shape,
                                                         self.spinup_sfc_h -
                                                         self.bed_h)

    @property
    def widths_m(self):
        """Compute the widths out of H and shape"""
        return para_width_from_thick.apply(self.bed_shape,
                                           self.thick)

    @property
    def section(self):
        return 2. / 3. * self.widths_m * self.thick

    @section.setter
    def section(self, val):
        if type(val) != torch.Tensor:
            val = to_torch_tensor(val, self.torch_type)
        new_thick = para_thick_from_section.apply(self.bed_shape, val)
        self.thick = new_thick
        # self.thick = (0.75 * val * torch.sqrt(self.bed_shape))**(2./3.)

    @utils.lazy_property
    def shape_str(self):
        """The bed shape in text (for debug and other things)"""
        return np.repeat('parabolic', self.nx)

    def _add_attrs_to_dataset(self, ds):
        """Add bed specific parameters."""
        ds['bed_shape'] = (['x'], self.bed_shape)


class RectangularBedFlowline(Flowline):
    """Simple shaped Flowline, glacier width does not change with ice thickness

    """

    def __init__(self, line=None, dx=None, map_dx=None,
                 surface_h=None, bed_h=None, widths_m=None, rgi_id=None,
                 water_level=None, torch_type=torch.float):
        """ Instanciate.

        Parameters
        ----------
        line : :py:class:`shapely.geometry.LineString`
            the geometrical line of a :py:class:`oggm.Centerline`

        Properties
        ----------
        #TODO: document properties
        """
        super(RectangularBedFlowline, self).__init__(line, dx, map_dx,
                                                     surface_h, bed_h,
                                                     rgi_id=rgi_id,
                                                     water_level=water_level,
                                                     torch_type=torch_type)

        self.fl_type = 'RectangularFlowline'

        self.widths = widths_m

        self.spinup_widths = widths_m

    @property
    def widths_m(self):
        return self.widths

    @property
    def section(self):
        return self.widths_m * self.thick

    @section.setter
    def section(self, val):
        # val = to_torch_tensor(val, self.torch_type)
        self.thick = val / self.widths_m

    @utils.lazy_property
    def shape_str(self):
        """The bed shape in text (for debug and other things)"""
        return np.repeat('rectangular', self.nx)

    def _add_attrs_to_dataset(self, ds):
        """Add bed specific parameters."""
        ds['widths'] = (['x'], self._widths)


class TrapezoidalBedFlowline(Flowline):
    """A Flowline with trapezoidal shape and two degrees of freedom
    """

    def __init__(self, line=None, dx=None, map_dx=None, surface_h=None,
                 bed_h=None, w0_m=None, lambdas=None, rgi_id=None,
                 water_level=None, torch_type=torch.float):
        """ Instanciate.

        Parameters
        ----------
        line : :py:class:`shapely.geometry.LineString`
            the geometrical line of a :py:class:`oggm.Centerline`

        Properties
        ----------
        #TODO: document properties
        """
        super(TrapezoidalBedFlowline, self).__init__(line, dx, map_dx,
                                                     surface_h, bed_h,
                                                     rgi_id=rgi_id,
                                                     water_level=water_level,
                                                     torch_type=torch_type)

        if len(lambdas) == 1:
            lambdas = lambdas.repeat(len(bed_h))
        self._lambdas = to_torch_tensor(lambdas,
                                        self.torch_type)

        self._w0_m = w0_m

        if torch.any(self._w0_m <= 0):
            raise ValueError('Trapezoid beds need to have origin widths > 0.')

        self.fl_type = 'TrapezoidalFlowline'

        self._prec = torch.where(self._lambdas == 0)[0]

        self.spinup_widths = self._w0_m + self._lambdas * (self.spinup_sfc_h -
                                                           self.bed_h)

    @property
    def widths_m(self):
        """Compute the widths out of H and shape"""
        return self._w0_m + self._lambdas * self.thick

    @property
    def section(self):
        return (self.widths_m + self._w0_m) / 2 * self.thick

    @section.setter
    def section(self, val):
        thick = (torch.sqrt(self._w0_m ** 2 + 2 * self._lambdas * val) -
                 self._w0_m) / self._lambdas
        self.thick = thick
        # val = to_torch_tensor(val, self.torch_type)

        # b = 2 * self._w0_m
        # a = 2 * self._lambdas
        # with np.errstate(divide='ignore', invalid='ignore'):
        #    thick = (torch.sqrt(b**2 + 4 * a * val) - b) / a
        # thick[self._prec] = val[self._prec] / self._w0_m[self._prec]
        # self.thick = thick

    @utils.lazy_property
    def shape_str(self):
        """The bed shape in text (for debug and other things)"""
        return np.repeat('trapezoid', self.nx)

    def _add_attrs_to_dataset(self, ds):
        """Add bed specific parameters."""
        ds['widths'] = (['x'], self.widths)
        ds['lambdas'] = (['x'], self._lambdas)


class MixedBedFlowline(Flowline):
    """A Flowline which can take a combination of different shapes (default)
    The default shape is parabolic. At ice divides a rectangular shape is used.
    And if the parabola gets too flat a trapezoidal shape is used.
    """

    def __init__(self, *, line=None, dx=None, map_dx=None, surface_h=None,
                 bed_h=None, section=None, bed_shape=None,
                 is_trapezoid=None, lambdas=None, w0_m=None,
                 rgi_id=None, water_level=None, torch_type=torch.double, device='cpu'):
        """ Instanciate.
        Parameters
        ----------
        line : :py:class:`shapely.geometry.LineString`
            the geometrical line of a :py:class:`oggm.Centerline`
        """

        super(MixedBedFlowline, self).__init__(line=line, dx=dx, map_dx=map_dx,
                                               surface_h=surface_h,
                                               bed_h=bed_h,
                                               rgi_id=rgi_id,
                                               water_level=water_level,
                                               torch_type=torch_type, device=device)
        assert len(lambdas) == self.nx
        if type(lambdas) != torch.Tensor:
            self._lambdas = torch.tensor(lambdas,
                                         dtype=self.torch_type,
                                         device=self.device,
                                         requires_grad=False)
        else:
            self._lambdas = lambdas

        assert len(w0_m) == self.nx
        if type(w0_m) != torch.Tensor:
            self._w0_m = torch.tensor(w0_m,
                                      dtype=self.torch_type,
                                      device=self.device,
                                      requires_grad=False)
        else:
            self._w0_m = w0_m

        assert len(bed_shape) == self.nx
        if type(bed_shape) == torch.Tensor:
            warnings.warn('Gradient calculation not possible for bed_shape!')
        self.bed_shape = torch.tensor(bed_shape,
                                      dtype=self.torch_type,
                                      device=self.device,
                                      requires_grad=False)

        assert len(is_trapezoid) == self.nx
        is_trapezoid_torch = torch.tensor(is_trapezoid,
                                          dtype=torch.bool,
                                          device=self.device,
                                          requires_grad=False)
        self.is_trapezoid = is_trapezoid_torch & (lambdas != 0)
        self.is_rectangular = is_trapezoid_torch & (lambdas == 0)
        self.is_parabolic = ~is_trapezoid_torch

        # Sanity
        self.bed_shape[is_trapezoid] = np.NaN
        self._lambdas[~is_trapezoid] = np.NaN
        self._w0_m[~is_trapezoid] = np.NaN

        # Indices for different calculations
        self._ptrap = torch.where(self.is_trapezoid)[0]
        self._prec = torch.where(self.is_rectangular)[0]
        self._ppar = torch.where(self.is_parabolic)[0]

        # To speedup calculations if bed is not present
        self._do_trapeze = torch.any(self.is_trapezoid)
        self._do_rectangular = torch.any(self.is_rectangular)
        self._do_parabolic = torch.any(self.is_parabolic)

        # some number tensors for calculation
        self.number_two = torch.tensor(2.,
                                       dtype=self.torch_type,
                                       device=self.device,
                                       requires_grad=False)
        self.number_three = torch.tensor(3.,
                                         dtype=self.torch_type,
                                         device=self.device,
                                         requires_grad=False)

        if (torch.any(self._w0_m[is_trapezoid] <= 0) or
                torch.any(~torch.isfinite(self._w0_m[is_trapezoid]))):
            raise ValueError('Trapezoid beds need to have origin widths > 0.')

        assert torch.all(self.bed_shape[~is_trapezoid] > 0)

        if section is not None:
            assert np.all(section >= 0)
            assert torch.allclose(torch.tensor(section, dtype=self.torch_type,
                                               device=self.device),
                                  self.section)

    @property
    def widths_m(self):
        """Compute the widths out of H and shape"""
        out = torch.empty(self.nx,
                          dtype=self.torch_type,
                          device=self.device,
                          requires_grad=False)
        # initialise with NaN to check if there was a calculation at each gridpoint
        out[:] = np.NaN

        # calculate widths depending on the shape
        if self._do_trapeze:
            out[self._ptrap] = self._w0_m[self._ptrap] + self._lambdas[self._ptrap] * self.thick[
                self._ptrap]
        if self._do_rectangular:
            out[self._prec] = self._w0_m[self._prec]
        if self._do_parabolic:
            out[self._ppar] = para_width_from_thick.apply(self.bed_shape[self._ppar],
                                                          self.thick[self._ppar])

        # test that every grid point has a calculated value
        assert torch.any(~torch.isnan(out))

        return out

    @property
    def section(self):
        out = torch.empty(self.nx,
                          dtype=self.torch_type,
                          device=self.device,
                          requires_grad=False)
        # initialise with NaN to check if there was a calculation at each gridpoint
        out[:] = np.NaN

        # calculate section depending on bed shape
        if self._do_trapeze:
            out[self._ptrap] = (self.widths_m[self._ptrap] + self._w0_m[self._ptrap]) / \
                               self.number_two * self.thick[self._ptrap]
        if self._do_rectangular:
            out[self._prec] = self.widths_m[self._prec] * self.thick[self._prec]
        if self._do_parabolic:
            out[self._ppar] = self.number_two / self.number_three * self.widths_m[self._ppar] * \
                              self.thick[self._ppar]

        # test that every grid point has a calculated value
        assert torch.any(~torch.isnan(out))

        return out

    @section.setter
    def section(self, val):
        out = torch.empty(self.nx,
                          dtype=self.torch_type,
                          device=self.device,
                          requires_grad=False)
        # initialise with NaN to check if there was a calculation at each gridpoint
        out[:] = np.NaN

        # set new thick depending on bed shape
        if self._do_trapeze:
            out[self._ptrap] = (torch.sqrt(self._w0_m[self._ptrap].pow(2) +
                                           self.number_two * self._lambdas[self._ptrap] *
                                           val[self._ptrap]) -
                                self._w0_m[self._ptrap]) / self._lambdas[self._ptrap]
        if self._do_rectangular:
            out[self._prec] = val[self._prec] / self.widths_m[self._prec]
        if self._do_parabolic:
            out[self._ppar] = para_thick_from_section.apply(self.bed_shape[self._ppar],
                                                            val[self._ppar])

        # test that every grid point has a calculated value
        assert torch.any(~torch.isnan(out))

        self.thick = out

    @utils.lazy_property
    def shape_str(self):
        """The bed shape in text (for debug and other things)"""
        out = np.repeat('rectangular', self.nx)
        out[self.is_parabolic] = 'parabolic'
        out[self.is_trapezoid] = 'trapezoid'
        return out

    def _add_attrs_to_dataset(self, ds):
        """Add bed specific parameters."""

        ds['section'] = (['x'], self.section)
        ds['bed_shape'] = (['x'], self.bed_shape)
        ds['is_trapezoid'] = (['x'], self.is_trapezoid)
        ds['widths_m'] = (['x'], self._w0_m)
        ds['lambdas'] = (['x'], self._lambdas)


class FlowlineModel(object):
    """Interface to the actual model"""

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None,
                 fs=None, inplace=False, is_tidewater=False,
                 is_lake_terminating=False, mb_elev_feedback='annual',
                 check_for_boundaries=False, water_level=None):
        """Create a new flowline model from the flowlines and a MB model.

        Parameters
        ----------
        flowlines : list
            a list of :py:class:`oggm.Flowline` instances, sorted by order
        mb_model : :py:class:`oggm.core.massbalance.MassBalanceModel`
            the MB model to use
        y0 : int
            the starting year of the simulation
        glen_a : float
            glen's parameter A
        fs: float
            sliding parameter
        inplace : bool
            whether or not to make a copy of the flowline objects for the run
            setting to True implies that your objects will be modified at run
            time by the model (can help to spare memory)
        is_tidewater: bool, default: False
            is this a tidewater glacier?
        is_lake_terminating: bool, default: False
            is this a lake terminating glacier?
        mb_elev_feedback : str, default: 'annual'
            'never', 'always', 'annual', or 'monthly': how often the
            mass-balance should be recomputed from the mass balance model.
            'Never' is equivalent to 'annual' but without elevation feedback
            at all (the heights are taken from the first call).
        check_for_boundaries : bool
            whether the model should raise an error when the glacier exceeds
            the domain boundaries. The default is to follow
            PARAMS['error_when_glacier_reaches_boundaries']
        """

        self.is_tidewater = is_tidewater
        self.is_lake_terminating = is_lake_terminating
        self.is_marine_terminating = is_tidewater and not is_lake_terminating

        if water_level is None:
            self.water_level = 0
            if self.is_lake_terminating:
                if not flowlines[-1].has_ice():
                    raise InvalidParamsError('Set `water_level` for lake '
                                             'terminating glaciers in '
                                             'idealized runs')
                # Arbitrary water level 1m below last grid points elevation
                min_h = flowlines[-1].surface_h[flowlines[-1].thick > 0][-1]
                self.water_level = (min_h -
                                    cfg.PARAMS['free_board_lake_terminating'])
        else:
            self.water_level = water_level

        # Mass balance
        self.mb_elev_feedback = mb_elev_feedback.lower()
        if self.mb_elev_feedback in ['never', 'annual']:
            self.mb_step = 'annual'
        elif self.mb_elev_feedback in ['always', 'monthly']:
            self.mb_step = 'monthly'
        self.mb_model = mb_model

        # define flowlines
        self.fls = None
        self._tributary_indices = None
        self.reset_flowlines(flowlines, inplace=inplace)

        self.torch_type = self.fls[0].torch_type
        self.device = self.fls[0].device
        # update flowline spinup states at initialisation
        self.fls[0].spinup_sfc_h = self.fls[0].surface_h
        self.fls[0].spinup_widths = self.fls[0].widths_m

        # Defaults
        if glen_a is None:
            glen_a = cfg.PARAMS['glen_a']
        if fs is None:
            fs = cfg.PARAMS['fs']
        self.glen_a = torch.tensor(glen_a,
                                   dtype=self.torch_type,
                                   device=self.device,
                                   requires_grad=False)
        self.fs = torch.tensor(fs,
                               dtype=self.torch_type,
                               device=self.device,
                               requires_grad=False)
        self.glen_n = torch.tensor(cfg.PARAMS['glen_n'],
                                   dtype=self.torch_type,
                                   device=self.device,
                                   requires_grad=False)
        self.rho = torch.tensor(cfg.PARAMS['ice_density'],
                                dtype=self.torch_type,
                                device=self.device,
                                requires_grad=False)
        if check_for_boundaries is None:
            check_for_boundaries = cfg.PARAMS[('error_when_glacier_reaches_'
                                               'boundaries')]
        self.check_for_boundaries = check_for_boundaries

        # we keep glen_a as input, but for optimisation we stick to "fd"
        self._fd = torch.tensor(2.,
                                dtype=self.torch_type,
                                device=self.device,
                                requires_grad=False) / \
                   (self.glen_n + torch.tensor(2.,
                                               dtype=self.torch_type,
                                               device=self.device,
                                               requires_grad=False)) * self.glen_a

        # Calving shenanigans
        self.calving_m3_since_y0 = 0.  # total calving since time y0
        self.calving_rate_myr = 0.

        self.y0 = None
        self.t = None
        self.reset_y0(y0)

        # to count how often the model is running
        self.iterations = 0

    @property
    def mb_model(self):
        return self._mb_model

    @mb_model.setter
    def mb_model(self, value):
        # We need a setter because the MB func is stored as an attr too
        _mb_call = None
        if value:
            if self.mb_elev_feedback in ['always', 'monthly']:
                _mb_call = value.get_monthly_mb
            elif self.mb_elev_feedback in ['annual', 'never']:
                _mb_call = value.get_annual_mb
            else:
                raise ValueError('mb_elev_feedback not understood')
        self._mb_model = value
        self._mb_call = _mb_call
        self._mb_current_date = None
        self._mb_current_out = dict()
        self._mb_current_heights = dict()

    def reset_y0(self, y0):
        """Reset the initial model time"""
        self.y0 = y0
        self.t = 0

    def reset_flowlines(self, flowlines, inplace=True):
        """Reset the initial model flowlines"""

        # if not inplace:
        #    flowlines = copy.deepcopy(flowlines)

        try:
            len(flowlines)
        except TypeError:
            flowlines = [flowlines]

        self.fls = flowlines

        # list of tributary coordinates and stuff
        trib_ind = []
        for fl in self.fls:
            # Important also
            fl.water_level = self.water_level
            if fl.flows_to is None:
                trib_ind.append((None, None, None, None))
                continue
            idl = self.fls.index(fl.flows_to)
            ide = fl.flows_to_indice
            if fl.flows_to.nx >= 9:
                gk = GAUSSIAN_KERNEL[9]
                id0 = ide - 4
                id1 = ide + 5
            elif fl.flows_to.nx >= 7:
                gk = GAUSSIAN_KERNEL[7]
                id0 = ide - 3
                id1 = ide + 4
            elif fl.flows_to.nx >= 5:
                gk = GAUSSIAN_KERNEL[5]
                id0 = ide - 2
                id1 = ide + 3
            trib_ind.append((idl, id0, id1, gk))

        self._tributary_indices = trib_ind

    @property
    def yr(self):
        return self.y0 + self.t / SEC_IN_YEAR

    @property
    def area_m2(self):
        return torch.sum(torch.cat([f.area_m2 for f in self.fls], 0))

    @property
    def volume_m3(self):
        return torch.sum(torch.cat([f.volume_m3 for f in self.fls], 0))

    @property
    def volume_km3(self):
        return self.volume_m3 * torch.tensor(1e-9,
                                             dtype=self.torch_type,
                                             device=self.device,
                                             requires_grad=False)

    @property
    def volume_bsl_m3(self):
        return torch.sum(torch.cat([f.volume_bsl_m3 for f in self.fls], 0))

    @property
    def volume_bsl_km3(self):
        return self.volume_bsl_m3 * torch.tensor(1e-9,
                                                 dtype=self.torch_type,
                                                 device=self.device,
                                                 requires_grad=False)

    @property
    def volume_bwl_m3(self):
        return torch.sum(torch.cat([f.volume_bwl_m3 for f in self.fls], 0))

    @property
    def volume_bwl_km3(self):
        return self.volume_bwl_m3 * torch.tensor(1e-9,
                                                 dtype=self.torch_type,
                                                 device=self.device,
                                                 requires_grad=False)

    @property
    def area_km2(self):
        return self.area_m2 * torch.tensor(1e-6,
                                           dtype=self.torch_type,
                                           device=self.device,
                                           requires_grad=False)

    @property
    def length_m(self):
        return self.fls[-1].length_m

    def get_mb(self, heights, year=None, fl_id=None, fls=None):
        """Get the mass balance at the requested height and time.

        Optimized so that no mb model call is necessary at each step.
        """
        # Do we even have to optimise?
        if self.mb_elev_feedback == 'always':
            return self._mb_call(heights, year=year, fl_id=fl_id, fls=fls)

        # Ok, user asked for it
        if fl_id is None:
            raise ValueError('Need fls_id')

        if self.mb_elev_feedback == 'never':
            # The very first call we take the heights
            if fl_id not in self._mb_current_heights:
                # We need to reset just this tributary
                self._mb_current_heights[fl_id] = heights
            # All calls we replace
            heights = self._mb_current_heights[fl_id]

        date = utils.floatyear_to_date(year)
        if self.mb_elev_feedback in ['annual', 'never']:
            # ignore month changes
            date = (date[0], date[0])

        if self._mb_current_date == date:
            if fl_id not in self._mb_current_out:
                # We need to reset just this tributary
                self._mb_current_out[fl_id] = self._mb_call(heights,
                                                            year=year,
                                                            fl_id=fl_id,
                                                            fls=fls)
        else:
            # We need to reset all
            self._mb_current_date = date
            self._mb_current_out = dict()
            self._mb_current_out[fl_id] = self._mb_call(heights,
                                                        year=year,
                                                        fl_id=fl_id,
                                                        fls=fls)

        return self._mb_current_out[fl_id]

    def to_netcdf(self, path):
        """Creates a netcdf group file storing the state of the model."""

        flows_to_id = []
        for trib in self._tributary_indices:
            flows_to_id.append(trib[0] if trib[0] is not None else -1)

        ds = xr.Dataset()
        try:
            ds.attrs['description'] = 'OGGM model output'
            ds.attrs['oggm_version'] = __version__
            ds.attrs['calendar'] = '365-day no leap'
            ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
            ds['flowlines'] = ('flowlines', np.arange(len(flows_to_id)))
            ds['flows_to_id'] = ('flowlines', flows_to_id)
            ds.to_netcdf(path)
            for i, fl in enumerate(self.fls):
                ds = fl.to_dataset()
                ds.to_netcdf(path, 'a', group='fl_{}'.format(i))
        finally:
            ds.close()

    def check_domain_end(self):
        """Returns False if the glacier reaches the domains bound."""
        return np.isclose(self.fls[-1].thick[-1], 0)

    def step(self, dt):
        """Advance the numerical simulation of one single step.

        Important: the step dt is a maximum boundary that is *not* guaranteed
        to be met if dt is too large for the underlying numerical
        implementation. However, ``step(dt)`` should never cross the desired
        time step, i.e. if dt is small enough to ensure stability, step
        should match it.

        The caller will know how much has been actually advanced by looking
        at the output of ``step()`` or by monitoring ``self.t`` or `self.yr``

        Parameters
        ----------
        dt : float
             the step length in seconds

        Returns
        -------
        the actual dt chosen by the numerical implementation. Guaranteed to
        be dt or lower.
        """
        raise NotImplementedError

    def run_until(self, y1):
        """Runs the model from the current year up to a given year date y1.

        This function runs the model for the time difference y1-self.y0
        If self.y0 has not been specified at some point, it is 0 and y1 will
        be the time span in years to run the model for.

        Parameters
        ----------
        y1 : float
            Upper time span for how long the model should run
        """

        # We force timesteps to monthly frequencies for consistent results
        # among use cases (monthly or yearly output) and also to prevent
        # "too large" steps in the adaptive scheme.
        ts = utils.monthly_timeseries(self.yr.detach().numpy()
                                      if type(self.yr) == torch.Tensor
                                      else self.yr, y1)

        # Add the last date to be sure we end on it
        ts = np.append(ts, y1)
        ts = torch.tensor(ts,
                          dtype=self.torch_type,
                          device=self.device,
                          requires_grad=False)

        self.iterations = 0
        # Loop over the steps we want to meet
        for y in ts:
            t = (y - self.y0) * torch.tensor(SEC_IN_YEAR,
                                             dtype=self.torch_type,
                                             device=self.device,
                                             requires_grad=False)
            # because of CFL, step() doesn't ensure that the end date is met
            # lets run the steps until we reach our desired date
            while self.t < t:
                self.step(t - self.t)
                self.iterations = self.iterations + 1

            # Check for domain bounds
            if self.check_for_boundaries:
                if self.fls[-1].thick[-1] > 10:
                    raise RuntimeError('Glacier exceeds domain boundaries, '
                                       'at year: {}'.format(self.yr))

            # Check for NaNs
            for fl in self.fls:
                if torch.any(~torch.isfinite(fl.thick)):
                    raise FloatingPointError('NaN in numerical solution, '
                                             'at year: {}'.format(self.yr))

    def run_until_and_store(self, y1, run_path=None, diag_path=None,
                            store_monthly_step=None):
        """Runs the model and returns intermediate steps in xarray datasets.

        This function repeatedly calls FlowlineModel.run_until for either
        monthly or yearly time steps up till the upper time boundary y1.

        Parameters
        ----------
        y1 : int
            Upper time span for how long the model should run (needs to be
            a full year)
        run_path : str
            Path and filename where to store the model run dataset
        diag_path : str
            Path and filename where to store the model diagnostics dataset
        store_monthly_step : Bool
            If True (False)  model diagnostics will be stored monthly (yearly).
            If unspecified, we follow the update of the MB model, which
            defaults to yearly (see __init__).

        Returns
        -------
        run_ds : xarray.Dataset
            stores the entire glacier geometry. It is useful to visualize the
            glacier geometry or to restart a new run from a modelled geometry.
            The glacier state is stored at the begining of each hydrological
            year (not in between in order to spare disk space).
        diag_ds : xarray.Dataset
            stores a few diagnostic variables such as the volume, area, length
            and ELA of the glacier.
        """

        if int(y1) != y1:
            raise InvalidParamsError('run_until_and_store only accepts '
                                     'integer year dates.')

        if not self.mb_model.hemisphere:
            raise InvalidParamsError('run_until_and_store needs a '
                                     'mass-balance model with an unambiguous '
                                     'hemisphere.')
        # time
        yearly_time = np.arange(np.floor(self.yr), np.floor(y1) + 1)

        if store_monthly_step is None:
            store_monthly_step = self.mb_step == 'monthly'

        if store_monthly_step:
            monthly_time = utils.monthly_timeseries(self.yr, y1)
        else:
            monthly_time = np.arange(np.floor(self.yr), np.floor(y1) + 1)

        sm = cfg.PARAMS['hydro_month_' + self.mb_model.hemisphere]

        yrs, months = utils.floatyear_to_date(monthly_time)
        cyrs, cmonths = utils.hydrodate_to_calendardate(yrs, months,
                                                        start_month=sm)

        # init output
        if run_path is not None:
            self.to_netcdf(run_path)
        ny = len(yearly_time)
        if ny == 1:
            yrs = [yrs]
            cyrs = [cyrs]
            months = [months]
            cmonths = [cmonths]
        nm = len(monthly_time)
        sects = [(np.zeros((ny, fl.nx)) * np.NaN) for fl in self.fls]
        widths = [(np.zeros((ny, fl.nx)) * np.NaN) for fl in self.fls]
        bucket = [(np.zeros(ny) * np.NaN) for _ in self.fls]
        diag_ds = xr.Dataset()

        # Global attributes
        diag_ds.attrs['description'] = 'OGGM model output'
        diag_ds.attrs['oggm_version'] = __version__
        diag_ds.attrs['calendar'] = '365-day no leap'
        diag_ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S",
                                                  gmtime())
        diag_ds.attrs['hemisphere'] = self.mb_model.hemisphere
        diag_ds.attrs['water_level'] = self.water_level

        # Coordinates
        diag_ds.coords['time'] = ('time', monthly_time)
        diag_ds.coords['hydro_year'] = ('time', yrs)
        diag_ds.coords['hydro_month'] = ('time', months)
        diag_ds.coords['calendar_year'] = ('time', cyrs)
        diag_ds.coords['calendar_month'] = ('time', cmonths)

        diag_ds['time'].attrs['description'] = 'Floating hydrological year'
        diag_ds['hydro_year'].attrs['description'] = 'Hydrological year'
        diag_ds['hydro_month'].attrs['description'] = 'Hydrological month'
        diag_ds['calendar_year'].attrs['description'] = 'Calendar year'
        diag_ds['calendar_month'].attrs['description'] = 'Calendar month'

        # Variables and attributes
        diag_ds['volume_m3'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['volume_m3'].attrs['description'] = 'Total glacier volume'
        diag_ds['volume_m3'].attrs['unit'] = 'm 3'
        if self.is_marine_terminating:
            diag_ds['volume_bsl_m3'] = ('time', np.zeros(nm) * np.NaN)
            diag_ds['volume_bsl_m3'].attrs['description'] = ('Glacier volume '
                                                             'below '
                                                             'sea-level')
            diag_ds['volume_bsl_m3'].attrs['unit'] = 'm 3'
            diag_ds['volume_bwl_m3'] = ('time', np.zeros(nm) * np.NaN)
            diag_ds['volume_bwl_m3'].attrs['description'] = ('Glacier volume '
                                                             'below '
                                                             'water-level')
            diag_ds['volume_bwl_m3'].attrs['unit'] = 'm 3'

        diag_ds['area_m2'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['area_m2'].attrs['description'] = 'Total glacier area'
        diag_ds['area_m2'].attrs['unit'] = 'm 2'
        diag_ds['length_m'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['length_m'].attrs['description'] = 'Glacier length'
        diag_ds['length_m'].attrs['unit'] = 'm 3'
        diag_ds['ela_m'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['ela_m'].attrs['description'] = ('Annual Equilibrium Line '
                                                 'Altitude  (ELA)')
        diag_ds['ela_m'].attrs['unit'] = 'm a.s.l'
        if self.is_tidewater:
            diag_ds['calving_m3'] = ('time', np.zeros(nm) * np.NaN)
            diag_ds['calving_m3'].attrs['description'] = ('Total accumulated '
                                                          'calving flux')
            diag_ds['calving_m3'].attrs['unit'] = 'm 3'
            diag_ds['calving_rate_myr'] = ('time', np.zeros(nm) * np.NaN)
            diag_ds['calving_rate_myr'].attrs['description'] = 'Calving rate'
            diag_ds['calving_rate_myr'].attrs['unit'] = 'm yr-1'

        # Run
        j = 0
        for i, (yr, mo) in enumerate(zip(monthly_time, months)):
            self.run_until(yr)
            # Model run
            if mo == 1:
                for s, w, b, fl in zip(sects, widths, bucket, self.fls):
                    s[j, :] = fl.section
                    w[j, :] = fl.widths_m
                    if self.is_tidewater:
                        try:
                            b[j] = fl.calving_bucket_m3
                        except AttributeError:
                            pass
                j += 1
            # Diagnostics
            diag_ds['volume_m3'].data[i] = self.volume_m3
            diag_ds['area_m2'].data[i] = self.area_m2
            diag_ds['length_m'].data[i] = self.length_m
            try:
                ela_m = self.mb_model.get_ela(year=yr, fls=self.fls,
                                              fl_id=len(self.fls) - 1)
                diag_ds['ela_m'].data[i] = ela_m
            except BaseException:
                # We really don't want to stop the model for some ELA issues
                diag_ds['ela_m'].data[i] = np.NaN

            if self.is_tidewater:
                diag_ds['calving_m3'].data[i] = self.calving_m3_since_y0
                diag_ds['calving_rate_myr'].data[i] = self.calving_rate_myr
                if self.is_marine_terminating:
                    diag_ds['volume_bsl_m3'].data[i] = self.volume_bsl_m3
                    diag_ds['volume_bwl_m3'].data[i] = self.volume_bwl_m3

        # to datasets
        run_ds = []
        for (s, w, b) in zip(sects, widths, bucket):
            ds = xr.Dataset()
            ds.attrs['description'] = 'OGGM model output'
            ds.attrs['oggm_version'] = __version__
            ds.attrs['calendar'] = '365-day no leap'
            ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S",
                                                 gmtime())
            ds.coords['time'] = yearly_time
            ds['time'].attrs['description'] = 'Floating hydrological year'
            varcoords = OrderedDict(time=('time', yearly_time),
                                    year=('time', yearly_time))
            ds['ts_section'] = xr.DataArray(s, dims=('time', 'x'),
                                            coords=varcoords)
            ds['ts_width_m'] = xr.DataArray(w, dims=('time', 'x'),
                                            coords=varcoords)
            if self.is_tidewater:
                ds['ts_calving_bucket_m3'] = xr.DataArray(b, dims=('time',),
                                                          coords=varcoords)
            run_ds.append(ds)

        # write output?
        if run_path is not None:
            encode = {'ts_section': {'zlib': True, 'complevel': 5},
                      'ts_width_m': {'zlib': True, 'complevel': 5},
                      }
            for i, ds in enumerate(run_ds):
                ds.to_netcdf(run_path, 'a', group='fl_{}'.format(i),
                             encoding=encode)
            # Add other diagnostics
            diag_ds.to_netcdf(run_path, 'a')

        if diag_path is not None:
            diag_ds.to_netcdf(diag_path)

        return run_ds, diag_ds

    def run_until_equilibrium(self, rate=0.001, ystep=5, max_ite=200):
        """ Runs the model until an equilibrium state is reached.

        Be careful: This only works for CONSTANT (not time-dependant)
        mass-balance models.
        Otherwise the returned state will not be in equilibrium! Don't try to
        calculate an equilibrium state with a RandomMassBalance model!
        """

        ite = 0
        was_close_zero = 0
        t_rate = 1
        while (t_rate > rate) and (ite <= max_ite) and (was_close_zero < 5):
            ite += 1
            v_bef = self.volume_m3
            self.run_until(self.yr + ystep)
            v_af = self.volume_m3
            if np.isclose(v_bef, 0., atol=1):
                t_rate = 1
                was_close_zero += 1
            else:
                t_rate = np.abs(v_af - v_bef) / v_bef
        if ite > max_ite:
            raise RuntimeError('Did not find equilibrium.')


class RectangularBedDiffusiveFlowlineModel(FlowlineModel):
    """A rectangular bed diffusive model, not used in the Thesis of COMBINE1D.
    The actual model, only valid for RectangularBedFlowline"""

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None, fs=0.,
                 fixed_dt=None, min_dt=SEC_IN_DAY, max_dt=31 * SEC_IN_DAY,
                 inplace=False, cfl_nr=0.165, **kwargs):
        """ Instanciate.

        Parameters
        ----------

        Properties
        ----------
        #TODO: document properties
        #TODO: Changed from assumed N=3 to N
        """

        super(RectangularBedDiffusiveFlowlineModel, self).__init__(flowlines,
                                                                   mb_model=mb_model,
                                                                   y0=y0, glen_a=glen_a,
                                                                   fs=fs,
                                                                   inplace=inplace,
                                                                   **kwargs)

        if len(self.fls) > 1:
            raise ValueError('Model does not work with tributaries.')

        # TODO: check if RectangularBedFlowline
        # if type(self.fls[0]) == combine.core.flowline_adapted.RectangularBedFlowline:
        #    raise TypeError('Only can use RectangularBedFlowline')

        self.dt_warning = False,
        if fixed_dt is not None:
            min_dt = fixed_dt
            max_dt = fixed_dt
        self.min_dt = min_dt
        self.max_dt = max_dt
        # defines cfl criterion, 0.165 in accordance with Hindmarsh 2001
        self.cfl_nr = cfl_nr

        # define indices for time step
        Nx = self.fls[0].nx
        self.k = np.arange(0, Nx)
        self.k_right = np.arange(1, Nx)
        self.k_left = np.arange(0, Nx - 1)
        self.k_stag = np.arange(0, Nx + 1)
        self.kp_stag = np.arange(1, Nx + 1)
        self.km_stag = np.arange(0, Nx)

    def step(self, dt):
        """Advance one step."""

        # Just a check to avoid useless computations
        if dt <= 0:
            raise InvalidParamsError('dt needs to be strictly positive')

        fl = self.fls[0]
        dx = fl.dx_meter
        dtype = self.fls[0].torch_type
        n = self.glen_n

        w = fl.widths_m
        H = fl.thick

        # get massbalance
        m_dot = torch.tensor(self.get_mb(fl.surface_h,
                                         self.yr,
                                         fl_id=0),
                             dtype=dtype,
                             requires_grad=False)

        S = fl.surface_h

        # Surface gradient on staggerd grid
        S_grad = torch.zeros(len(self.k_stag),
                             dtype=dtype,
                             requires_grad=False)
        S_grad[1:-1] = (S[self.k_right] - S[self.k_left]) / dx

        # Thickness on staggered grid
        H_stag = torch.zeros(len(self.k_stag),
                             dtype=dtype,
                             requires_grad=False)
        H_stag[1:-1] = (H[self.k_left] + H[self.k_right]) / 2

        # width on staggered grid
        w_stag = torch.zeros(len(self.k_stag),
                             dtype=dtype,
                             requires_grad=False)
        w_stag[1:-1] = (w[self.k_left] + w[self.k_right]) / 2

        # Diffusivity on staggered grid
        D = ((2 / (n + 2) * self.glen_a * H_stag ** 2 + self.fs) *
             w_stag * (self.rho * G) ** n * H_stag ** n *
             torch.abs(S_grad) ** (n - 1))

        # flux on staggered grid
        q = D * S_grad

        # gradient of flux on unstaggered grid
        q_grad = (q[self.kp_stag] - q[self.km_stag]) / dx

        # choose timestap as long as possible to fullfill cfl criterion
        dt_stab = self.cfl_nr * dx ** 2 / torch.max(torch.abs(D))

        dt_use = min(dt_stab, dt)

        # calculate new ice thickness
        H_new = H + dt_use * (m_dot + q_grad / w)

        # save new ice thickness in flowline
        fl.thick = torch.clamp(H_new, min=0)

        # Next step
        self.t += dt_use
        return dt_use


class FluxBasedModel(FlowlineModel):
    """The actual model used for COMBINE1D"""

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None, fs=None,
                 fixed_dt=None, min_dt=SEC_IN_DAY, max_dt=31 * SEC_IN_DAY,
                 inplace=False, cfl_nr=0.02, **kwargs):
        """ Instanciate.

        Parameters
        ----------

        Properties
        ----------
        #TODO: document properties
        """

        super(FluxBasedModel, self).__init__(flowlines,
                                             mb_model=mb_model,
                                             y0=y0, glen_a=glen_a,
                                             fs=fs,
                                             inplace=inplace,
                                             **kwargs)

        if len(self.fls) > 1:
            raise ValueError('Model does not work with tributaries.')

        # datatype for torch tensors
        self.torch_type = self.fls[0].torch_type
        self.device = self.fls[0].device

        self.dt_warning = False,
        if fixed_dt is not None:
            min_dt = fixed_dt
            max_dt = fixed_dt
        self.min_dt = torch.tensor(min_dt,
                                   dtype=self.torch_type,
                                   device=self.device,
                                   requires_grad=False)
        self.max_dt = torch.tensor(max_dt,
                                   dtype=self.torch_type,
                                   device=self.device,
                                   requires_grad=False)
        # defines cfl criterion
        self.cfl_nr = torch.tensor(cfl_nr,
                                   dtype=self.torch_type,
                                   device=self.device,
                                   requires_grad=False)

        # define indices for time step
        Nx = self.fls[0].nx
        # for calculation on staggered grid
        self.k_right = np.hstack([np.arange(0, Nx), Nx - 1])
        self.k_left = np.hstack([0, np.arange(0, Nx)])
        # for calculation on unstaggerd grid
        self.kp_stag = np.arange(1, Nx + 1)
        self.km_stag = np.arange(0, Nx)

        # define some numbers as tensors for calculation
        self.number_two = torch.tensor(2,
                                       dtype=self.torch_type,
                                       device=self.device,
                                       requires_grad=False)
        self.number_one = torch.tensor(1,
                                       dtype=self.torch_type,
                                       device=self.device,
                                       requires_grad=False)
        self.number_zero = torch.tensor(0,
                                        dtype=self.torch_type,
                                        device=self.device,
                                        requires_grad=False)
        self.number_ten = torch.tensor(10,
                                       dtype=self.torch_type,
                                       device=self.device,
                                       requires_grad=False)

    def step(self, dt):
        """Advance one step."""

        # Just a check to avoid useless computations
        if dt <= 0:
            raise InvalidParamsError('dt needs to be strictly positive')

        fl = self.fls[0]
        dx = fl.dx_meter
        n = self.glen_n

        H = fl.thick
        S = fl.surface_h
        CS = fl.section
        w = fl.widths_m

        # Surface gradient on staggerd grid
        S_grad = (S[self.k_right] - S[self.k_left]) / dx

        # Thickness on staggered grid
        H_stag = (H[self.k_left] + H[self.k_right]) / self.number_two

        # Section on staggered grid
        CS_stag = (CS[self.k_left] + CS[self.k_right]) / self.number_two

        # TODO: Implement shape factor function
        sf_stag = self.number_one

        # velocity on staggered grid
        u_stag = ((self.rho * G * S_grad) ** n *
                  (self._fd * H_stag ** (n + self.number_one) * sf_stag ** n +
                   self.fs * H_stag ** (n - self.number_one)))

        # flux on staggered grid
        q = u_stag * CS_stag

        # gradient of flux on unstaggered grid
        q_grad = (q[self.kp_stag] - q[self.km_stag]) / dx

        # choose timestap as long as possible to fullfill cfl criterion, and
        # check to be smaller than the maximum max_dt
        divisor = torch.max(torch.abs(u_stag))
        if divisor == 0:
            dt_cfl = self.max_dt
        else:
            dt_cfl = self.cfl_nr * dx / divisor

        if type(dt) != torch.Tensor:
            dt = torch.tensor(dt,
                              dtype=self.torch_type,
                              device=self.device,
                              requires_grad=False)
        dt_use = torch.clamp(torch.min(dt_cfl, dt),
                             self.number_zero,
                             self.max_dt)

        # check timestep that timestep is at least max_dt / 100, to avoid
        # memory overfolw and a break down of the program
        if (dt_use != dt) and (dt_use / self.max_dt < 0.001):
            raise MemoryError('Stopping dynamics run to avoid memory overflow')

        # get massbalance with COMBINE MassBalanceModel
        m_dot = self.get_mb(S,
                            year=self.yr,
                            fl_id=0)  # 0 because only one flowline

        # allow parabolic bed to grow
        m_dot_use = m_dot * torch.where((m_dot > 0.) & (w == 0),
                                        self.number_ten,
                                        w)

        # calculate new cross section area and filter for negative ones
        CS_new = torch.clamp(CS + dt_use * (q_grad + m_dot_use),
                             min=0)

        # save new section in flowline
        self.fls[0].section = CS_new

        # Next step
        self.t += dt_use

        return dt_use
