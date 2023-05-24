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
from oggm.core.flowline import Flowline as FlowlineOGGM
from oggm.core.flowline import FlowlineModel as FlowlineModelOGGM

# help function for gradient calculation
from combine1d.core.special_gradient_functions import para_width_from_thick,\
    para_thick_from_section, SolveBandedPyTorch

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
        surface_h: :py:class:`torch.Tensor`
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

    @property
    def volume_bsl_m3(self):
        # TODO: calving not implemented
        # return self._vol_below_level(water_level=0)
        raise NotImplementedError()

    @property
    def volume_bsl_km3(self):
        return self.volume_bsl_m3 * torch.tensor(1e-9,
                                                 dtype=self.torch_type,
                                                 device=self.device,
                                                 requires_grad=False)

    @property
    def volume_bwl_m3(self):
        # TODO: calving not implemented
        # return self._vol_below_level(water_level=self.water_level)
        raise NotImplementedError()

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
                 water_level=None, torch_type=torch.float, device='cpu'):
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
                                                   torch_type=torch_type,
                                                   device=device)

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
            val = torch.tensor(val, dtype=self.torch_type, device=self.device)
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
                 water_level=None, torch_type=torch.float, device='cpu'):
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
                                                     torch_type=torch_type,
                                                     device=device)

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
                 water_level=None, torch_type=torch.float, device='cpu'):
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
                                                     torch_type=torch_type,
                                                     device=device)

        if len(lambdas) == 1:
            lambdas = lambdas.repeat(len(bed_h))
        self._lambdas = torch.tensor(lambdas, dtype=self.torch_type,
                                     device=self.device)

        self._w0_m = torch.tensor(w0_m, dtype=self.torch_type,
                                  device=self.device)

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
                 rgi_id=None, water_level=None, torch_type=torch.double,
                 device='cpu'):
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
        if isinstance(bed_shape, torch.Tensor):
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


class FlowlineModelTorch(FlowlineModelOGGM):
    """Interface to the actual model"""

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None,
                 fs=None, inplace=True, is_tidewater=False,
                 is_lake_terminating=False, mb_elev_feedback='annual',
                 check_for_boundaries=False, water_level=None):
        """Create a new flowline model from the flowlines and a MB model.
        This calss is adapted from OGGMs FlowlineModel for the use of PyTorch

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

        try:
            len(flowlines)
        except TypeError:
            flowlines = [flowlines]

        self.torch_type = flowlines[0].torch_type
        self.device = flowlines[0].device

        super(FlowlineModelTorch, self).__init__(flowlines, mb_model=mb_model,
                                                 y0=y0, glen_a=glen_a, fs=fs,
                                                 inplace=inplace,
                                                 is_tidewater=is_tidewater,
                                                 is_lake_terminating=is_lake_terminating,
                                                 mb_elev_feedback=mb_elev_feedback,
                                                 check_for_boundaries=check_for_boundaries,
                                                 water_level=water_level)

        # update flowline spinup states at initialisation
        # TODO: is this still needed? (probably only for one-type flowlines)
        self.fls[0].spinup_sfc_h = self.fls[0].surface_h
        self.fls[0].spinup_widths = self.fls[0].widths_m

        # is needed so that run_until_and_store is working, because not
        # everything is supported right now
        cfg.PARAMS['dynamic_spinup_min_ice_thick'] = None
        cfg.PARAMS['store_diagnostic_variables'] = ['volume', 'area', 'length']

        # Defaults conversion to PyTorch tensor
        self.glen_a = torch.tensor(self.glen_a,
                                   dtype=self.torch_type,
                                   device=self.device,
                                   requires_grad=False)
        self.fs = torch.tensor(self.fs,
                               dtype=self.torch_type,
                               device=self.device,
                               requires_grad=False)
        self.glen_n = torch.tensor(self.glen_n,
                                   dtype=self.torch_type,
                                   device=self.device,
                                   requires_grad=False)
        self.rho = torch.tensor(self.rho,
                                dtype=self.torch_type,
                                device=self.device,
                                requires_grad=False)
        self.G = torch.tensor(G,
                              dtype=self.torch_type,
                              device=self.device,
                              requires_grad=False)

        # we keep glen_a as input, but for optimisation we stick to "fd"
        self._fd = torch.tensor(2.,
                                dtype=self.torch_type,
                                device=self.device,
                                requires_grad=False) / \
            (self.glen_n + torch.tensor(2.,
                                        dtype=self.torch_type,
                                        device=self.device,
                                        requires_grad=False)) * self.glen_a

        self.sec_in_year = torch.tensor(SEC_IN_YEAR,
                                        dtype=self.torch_type,
                                        device=self.device,
                                        requires_grad=False)

    def reset_y0(self, y0):
        """Reset the initial model time"""
        self.y0 = y0
        self.t = torch.tensor(0,
                              dtype=self.torch_type,
                              device=self.device)

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

        if self.required_model_steps == 'monthly':
            # We force timesteps to monthly frequencies for consistent results
            # among use cases (monthly or yearly output) and also to prevent
            # "too large" steps in the adaptive scheme.
            ts = utils.monthly_timeseries(self.yr.detach().to('cpu').numpy()
                                          if isinstance(self.yr, torch.Tensor)
                                          else self.yr, y1)

            # Add the last date to be sure we end on it
            ts = np.append(ts, y1)
        else:
            ts = np.arange(int(self.yr), int(y1 + 1))

        ts = torch.tensor(ts,
                          dtype=self.torch_type,
                          device=self.device,
                          requires_grad=False)

        # Loop over the steps we want to meet
        for y in ts:
            t = (y - self.y0) * self.sec_in_year
            # because of CFL, step() doesn't ensure that the end date is met
            # lets run the steps until we reach our desired date
            while self.t < t:
                self.step(t - self.t)

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

    def run_until_and_store(self, y1,
                            diag_path=None,
                            fl_diag_path=False,
                            geom_path=False,
                            store_monthly_step=None,
                            stop_criterion=None,
                            fixed_geometry_spinup_yr=None,
                            dynamic_spinup_min_ice_thick=None):
        ds = FlowlineModelOGGM.run_until_and_store(
            self, y1=y1, diag_path=None, fl_diag_path=fl_diag_path,
            geom_path=geom_path, store_monthly_step=store_monthly_step,
            stop_criterion=stop_criterion,
            fixed_geometry_spinup_yr=fixed_geometry_spinup_yr,
            dynamic_spinup_min_ice_thick=dynamic_spinup_min_ice_thick)

        # convert some values to numpy
        ds.attrs['glen_a'] = ds.attrs['glen_a'].detach().to('cpu').numpy()
        ds.attrs['fs'] = ds.attrs['fs'].detach().to('cpu').numpy()

        if diag_path is not None:
            ds.to_netcdf(diag_path)

        return ds


class RectangularBedDiffusiveFlowlineModel(FlowlineModelTorch):
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


class FluxBasedModel(FlowlineModelTorch):
    """Reimplemented and simplified version of OGGMs FluxBasedModel"""

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None, fs=None,
                 fixed_dt=None, min_dt=SEC_IN_DAY, max_dt=31 * SEC_IN_DAY,
                 cfl_nr=None, **kwargs):
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

        # factor for surface velocity
        self._surf_vel_fac = (self.glen_n + 2) / (self.glen_n + 1)

        self.u_stag = [torch.zeros(self.fls[0].nx + 1,
                                   dtype=self.torch_type,
                                   device=self.device,
                                   requires_grad=False)]
        self.flux_stag = [torch.zeros(self.fls[0].nx + 1,
                                      dtype=self.torch_type,
                                      device=self.device,
                                      requires_grad=False)]

        # defines cfl criterion
        if cfl_nr is None:
            cfl_nr = cfg.PARAMS['cfl_number']
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
        S_grad = (S[self.k_left] - S[self.k_right]) / dx

        # Thickness on staggered grid
        H_stag = (H[self.k_left] + H[self.k_right]) / self.number_two

        # Section on staggered grid
        CS_stag = (CS[self.k_left] + CS[self.k_right]) / self.number_two

        # TODO: Implement shape factor function
        sf_stag = self.number_one

        # velocity on staggered grid
        u_stag = ((self.rho * self.G * S_grad) ** n *
                  (self._fd * H_stag ** (n + self.number_one) * sf_stag ** n +
                   self.fs * H_stag ** (n - self.number_one)))

        # this is needed to use velocity as observation
        self.u_stag[0] = u_stag

        # flux on staggered grid
        q = u_stag * CS_stag
        self.flux_stag[0] = q

        # gradient of flux on unstaggered grid
        q_grad = (q[self.km_stag] - q[self.kp_stag]) / dx

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

        # check timestep that timestep is at least max_dt / 10000, to avoid
        # memory overfolw and a break down of the program
        if (dt_use != dt) and (dt_use / self.max_dt < 0.0001):
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


class SemiImplicitModel(FlowlineModelTorch):
    """Semi implicit flowline model from OGGM, adapted for the use with PyTorch

    It solves the same equation as the FluxBasedModel, but the ice flux q is
    implemented as q^t = D^t * (ds/dx)^(t+1).

    It supports only a single flowline (no tributaries) with bed shapes
    rectangular, trapezoidal or a mixture of both.

    The underlying equation is dC/dt = w*m + nable(q), with
    C ... cross section area
    t ... time
    w ... surface width
    m ... mass balance
    q ... ice flux
    nabla(q) ... flux divergence
    """

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None, fs=0.,
                 fixed_dt=None, min_dt=None, cfl_number=0.5, **kwargs):
        """ Instantiate. """

        super(SemiImplicitModel, self).__init__(flowlines,
                                                mb_model=mb_model,
                                                y0=y0, glen_a=glen_a,
                                                fs=fs, **kwargs)

        if len(self.fls) > 1:
            raise ValueError('Implicit model does not work with tributaries.')

        # check that whole flowline is trapezoidal or rectangular
        if ~np.all(
                np.logical_or(
                    self.fls[0].is_trapezoid.cpu().numpy().astype(bool),
                    self.fls[0].is_rectangular.cpu().numpy().astype(bool))):
            raise ValueError('Implicit model only works with a pure '
                             'trapezoidal flowline!')

        self.torch_type = self.fls[0].torch_type
        self.device = self.fls[0].device

        if fixed_dt is not None:
            min_dt = fixed_dt
            self.fixed_dt = torch.tensor(fixed_dt,
                                         dtype=self.torch_type,
                                         device=self.device,
                                         requires_grad=False)
        else:
            self.fixed_dt = None

        if min_dt is None:
            min_dt = cfg.PARAMS['cfl_min_dt']
        self.min_dt = torch.tensor(min_dt,
                                   dtype=self.torch_type,
                                   device=self.device,
                                   requires_grad=False)

        if cfl_number is None:
            cfl_number = cfg.PARAMS['cfl_number']
        self.cfl_number = torch.tensor(cfl_number,
                                       dtype=self.torch_type,
                                       device=self.device,
                                       requires_grad=False)

        # factor for surface velocity
        self._surf_vel_fac = (self.glen_n + 2) / (self.glen_n + 1)

        self.bed_h_exp = torch.cat((torch.tensor([self.fls[-1].bed_h[0]],
                                                 dtype=self.torch_type,
                                                 device=self.device),
                                    self.fls[-1].bed_h,
                                    torch.tensor([self.fls[-1].bed_h[-1]],
                                                 dtype=self.torch_type,
                                                 device=self.device)))
        self.dbed_h_exp_dx = (self.bed_h_exp[1:] - self.bed_h_exp[:-1]) / \
                             self.fls[0].dx_meter

        # test performance/memory consumption of .index_fill()
        # TODO: potential increase of performance here
        # self.u_stag = []
        # self.D_stag = []
        # self.D_stag.append(torch.zeros(nx + 1))
        # self.Amat_banded = np.zeros((3, nx))
        w0 = self.fls[0]._w0_m
        self.w0_stag = (w0[0:-1] + w0[1:]) / 2
        self.rhog = (self.rho * self.G) ** self.glen_n

        # define some numbers as tensors for calculation later
        self.number_two = torch.tensor(2,
                                       dtype=self.torch_type,
                                       device=self.device,
                                       requires_grad=False)
        self.number_one = torch.tensor(1,
                                       dtype=self.torch_type,
                                       device=self.device,
                                       requires_grad=False)

        # variables needed for the calculation of some diagnostics, this
        # calculations are done with @property, because they are not computed
        # on the fly during the dynamic run as in FluxBasedModel
        self._u_stag = torch.zeros(self.fls[0].nx + 1,
                                   dtype=self.torch_type,
                                   device=self.device,
                                   requires_grad=False)
        self._flux_stag = torch.zeros(self.fls[0].nx + 1,
                                      dtype=self.torch_type,
                                      device=self.device,
                                      requires_grad=False)
        self._slope_stag = torch.zeros(self.fls[0].nx + 1,
                                       dtype=self.torch_type,
                                       device=self.device,
                                       requires_grad=False)
        self._thick_stag = torch.zeros(self.fls[0].nx + 1,
                                       dtype=self.torch_type,
                                       device=self.device,
                                       requires_grad=False)
        self._section_stag = torch.zeros(self.fls[0].nx + 1,
                                         dtype=self.torch_type,
                                         device=self.device,
                                         requires_grad=False)

    @property
    def slope_stag(self):
        slope_stag = self._slope_stag

        surface_h = self.fls[0].surface_h
        dx = self.fls[0].dx_meter

        slope_stag[0] = 0
        slope_stag[1:-1] = (surface_h[0:-1] - surface_h[1:]) / dx
        slope_stag[-1] = slope_stag[-2]

        return [slope_stag]

    @property
    def thick_stag(self):
        thick_stag = self._thick_stag

        thick = self.fls[0].thick

        thick_stag[1:-1] = (thick[0:-1] + thick[1:]) / self.number_two
        thick_stag[[0, -1]] = thick[[0, -1]]

        return [thick_stag]

    @property
    def section_stag(self):
        section_stag = self._section_stag

        section = self.fls[0].section

        section_stag[1:-1] = (section[0:-1] + section[1:]) / self.number_two
        section_stag[[0, -1]] = section[[0, -1]]

        return [section_stag]

    @property
    def u_stag(self):
        u_stag = self._u_stag

        slope_stag = self.slope_stag[0]
        thick_stag = self.thick_stag[0]
        N = self.glen_n
        rhog = self.rhog

        rhogh = rhog * slope_stag ** N

        u_stag[:] = ((thick_stag ** (N + 1)) * self._fd * rhogh +
                     (thick_stag ** (N - 1)) * self.fs * rhogh)

        return [u_stag]

    @property
    def flux_stag(self):
        flux_stag = self._flux_stag

        section_stag = self.section_stag[0]
        u_stag = self.u_stag[0]

        flux_stag[:] = u_stag * section_stag

        return [flux_stag]

    def step(self, dt):
        """Advance one step."""

        # Just a check to avoid useless computations
        if dt <= 0:
            raise InvalidParamsError('dt needs to be strictly positive')

        # read out variables from current flowline
        fl = self.fls[0]
        dx = fl.dx_meter
        nx = fl.nx
        width = fl.widths_m
        thick = fl.thick
        surface_h = fl.surface_h

        # some variables needed later
        N = self.glen_n
        rhog = self.rhog

        # calculate staggered variables
        width_stag = (width[0:-1] + width[1:]) / self.number_two
        w0_stag = self.w0_stag
        thick_stag = (thick[0:-1] + thick[1:]) / self.number_two
        slope_stag = (surface_h[1:] - surface_h[0:-1]) / dx

        # calculate diffusivity (including sliding fs)
        # boundary condition D_stag_0 = D_stag_end = 0
        # TODO: potential optimisation possible, maybe with .index_fill()
        # d_stag = self.d_stag[0]
        d_stag = torch.zeros(nx + 1,
                             dtype=self.torch_type,
                             device=self.device)
        d_stag[1:-1] = ((self._fd * thick_stag ** (N + self.number_two) +
                         self.fs * thick_stag ** N) * rhog *
                        (w0_stag + width_stag) / self.number_two *
                        torch.abs(slope_stag) ** (N - self.number_one))

        # Time step
        if type(dt) != torch.Tensor:
            dt = torch.tensor(dt,
                              dtype=self.torch_type,
                              device=self.device,
                              requires_grad=False)
        if self.fixed_dt:
            # change only if step dt is larger than the chosen dt
            if self.fixed_dt < dt:
                dt = self.fixed_dt
        else:
            # use stability criterion dt < dx^2 / max(D/w)
            divisor = torch.max(torch.abs(d_stag[1:-1] / width_stag))
            if divisor > cfg.FLOAT_EPS:
                cfl_dt = self.cfl_number * dx ** self.number_two / divisor
            else:
                cfl_dt = dt

            if cfl_dt < dt:
                dt = cfl_dt
                if cfl_dt < self.min_dt:
                    raise MemoryError(
                        'CFL error: required time step smaller '
                        'than the minimum allowed: '
                        '{:.1f}s vs {:.1f}s. Happening at '
                        'simulation year {:.1f}, fl_id {}, '
                        'bin_id {} and max_D/w {:.3f} m2 yr-1. '
                        'To avoid memory overflow!'
                        ''.format(cfl_dt, self.min_dt, self.yr, 0,
                                  np.argmax(np.abs(d_stag.detach())),
                                  divisor * cfg.SEC_IN_YEAR))

        # calculate diagonals of Amat
        d0 = dt / dx ** self.number_two * (d_stag[:-1] + d_stag[1:]) / width
        dm = - dt / dx ** self.number_two * d_stag[:-1] / width
        dp = - dt / dx ** self.number_two * d_stag[1:] / width

        # construct banded form of the matrix, which is used during solving
        # (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_banded.html)
        # original matrix:
        # d_matrix = (np.diag(np.ones(len(d0)) + d0) +
        #             np.diag(dm[1:], -1) +
        #             np.diag(dp[:-1], 1))
        # TODO: potential optimisation possible
        # self.d_matrix_banded[0, 1:] = dp[:-1]
        # self.d_matrix_banded[1, :] = np.ones(len(d0)) + d0
        # self.d_matrix_banded[2, :-1] = dm[1:]
        d_matrix_banded = torch.zeros((3, nx),
                                      dtype=self.torch_type,
                                      device=self.device)
        d_matrix_banded[0, 1:] = dp[:-1]
        d_matrix_banded[1, :] = torch.ones(len(d0),
                                           dtype=self.torch_type,
                                           device=self.device) + d0
        d_matrix_banded[2, :-1] = dm[1:]

        # correction term for glacier bed (original equation is an equation for
        # the surface height s, which is transformed in an equation for h, as
        # s = h + b the term below comes from the '- b'
        b_corr = - d_stag * self.dbed_h_exp_dx

        # prepare rhs
        smb = self.get_mb(surface_h, year=self.yr, fl_id=0)
        rhs = thick + smb * dt + dt / width * (b_corr[:-1] - b_corr[1:]) / dx

        # solve matrix and ensure that new thickness >= 0
        thick_new = torch.clamp(SolveBandedPyTorch.apply(d_matrix_banded,
                                                         rhs),
                                min=0)

        # update flowline thickness
        fl.thick = thick_new

        # Next step
        self.t += dt

        return dt
