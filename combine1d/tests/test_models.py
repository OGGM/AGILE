import pytest
import numpy as np
import torch
from combine1d.core.massbalance import ConstantMassBalanceTorch
from oggm.core.massbalance import ConstantMassBalance
from combine1d.core.flowline import MixedBedFlowline
from combine1d.core.flowline import FluxBasedModel as combine_flux_model
from oggm.core.flowline import FluxBasedModel as oggm_flux_model
from oggm.utils._funcs import date_to_floatyear


pytestmark = pytest.mark.filterwarnings("ignore:<class 'combine1d.core.torch_interp1d.Interp1d'> "
                                        "should not be instantiated.:DeprecationWarning")


class TestModels:
    def test_constant_mb_model(self, hef_gdir):
        y0 = 2000
        halfsize = 15
        combine_mb_model = ConstantMassBalanceTorch(hef_gdir, y0=y0, halfsize=halfsize)
        oggm_mb_model = ConstantMassBalance(hef_gdir, y0=y0, halfsize=halfsize)
        test_height_shift = 100.
        combine_mb_model_shift = ConstantMassBalanceTorch(
            hef_gdir, y0=y0, halfsize=halfsize, height_shift=test_height_shift)

        fls = hef_gdir.read_pickle('model_flowlines')
        h = []
        for fl in fls:
            # We use bed because of overdeepenings
            h = np.append(h, fl.bed_h)
            h = np.append(h, fl.surface_h)
        zminmax = np.round([np.min(h), np.max(h)])
        heights = np.linspace(zminmax[0], zminmax[1], num=20)
        heights_torch = torch.tensor(heights, dtype=torch.double)
        heights_torch_shift = heights_torch + torch.tensor(test_height_shift,
                                                           dtype=torch.double)

        # test annual
        oggm_annual_mbs = oggm_mb_model.get_annual_mb(heights)
        combine_annual_mbs = combine_mb_model.get_annual_mb(heights_torch)
        combine_annual_mbs_shift = \
            combine_mb_model_shift.get_annual_mb(heights_torch_shift)

        assert np.allclose(oggm_annual_mbs, combine_annual_mbs)
        assert type(combine_annual_mbs) == torch.Tensor
        assert combine_annual_mbs.shape == heights_torch.shape
        assert np.allclose(combine_annual_mbs, combine_annual_mbs_shift)
        assert type(combine_annual_mbs_shift) == torch.Tensor
        assert combine_annual_mbs_shift.shape == heights_torch_shift.shape

        # test monthly
        for month in np.arange(12) + 1:
            yr = date_to_floatyear(0, month)
            oggm_monthly_mbs = oggm_mb_model.get_monthly_mb(heights, year=yr)
            combine_monthly_mbs = combine_mb_model.get_monthly_mb(heights_torch, year=yr)
            combine_monthly_mbs_shift = \
                combine_mb_model_shift.get_monthly_mb(heights_torch_shift, year=yr)

            assert np.allclose(oggm_monthly_mbs, combine_monthly_mbs)
            assert type(combine_monthly_mbs) == torch.Tensor
            assert combine_monthly_mbs.shape == heights_torch.shape
            assert np.allclose(combine_monthly_mbs, combine_monthly_mbs_shift)
            assert type(combine_monthly_mbs_shift) == torch.Tensor
            assert combine_monthly_mbs_shift.shape == heights_torch_shift.shape

    def test_fluxmodel(self, hef_gdir):
        # define flowlines
        oggm_fls = hef_gdir.read_pickle('model_flowlines')
        fl = oggm_fls[0]
        combine_fls = MixedBedFlowline(line=fl.line,
                                       dx=fl.dx,
                                       map_dx=fl.map_dx,
                                       surface_h=fl.surface_h,
                                       bed_h=fl.bed_h,
                                       section=fl.section,
                                       bed_shape=fl.bed_shape,
                                       is_trapezoid=fl.is_trapezoid,
                                       lambdas=fl._lambdas,
                                       w0_m=fl._w0_m,
                                       rgi_id=fl.rgi_id,
                                       water_level=fl.water_level,
                                       torch_type=torch.double,
                                       device="cpu")

        # define MassBalanceModels
        y0 = 2000
        halfsize = 15
        combine_mb_model = ConstantMassBalanceTorch(hef_gdir, y0=y0, halfsize=halfsize)
        oggm_mb_model = ConstantMassBalance(hef_gdir, y0=y0, halfsize=halfsize)

        # define FluxModels
        oggm_model = oggm_flux_model(oggm_fls, mb_model=oggm_mb_model, y0=0,
                                     cfl_number=0.01)
        combine_model = combine_flux_model(combine_fls, mb_model=combine_mb_model, y0=0)

        # Let models run
        oggm_model.run_until(30.)
        combine_model.run_until(30.)

        # Compare models
        def compare_mdls(combine_mdl, oggm_mdl):
            assert np.isclose(combine_mdl.area_m2, oggm_mdl.area_m2, rtol=1e-4)
            assert np.isclose(combine_mdl.area_km2, oggm_mdl.area_km2, rtol=1e-4)
            assert np.isclose(combine_mdl.volume_m3, oggm_mdl.volume_m3, rtol=1e-3)
            assert np.isclose(combine_mdl.volume_km3, oggm_mdl.volume_km3, rtol=1e-3)
            assert np.allclose(combine_mdl.fls[0].surface_h, oggm_mdl.fls[0].surface_h, rtol=1e-4)

        compare_mdls(combine_model, oggm_model)

        # test with year as torch tensor
        oggm_model.run_until(40.)
        combine_model.run_until(torch.tensor(40., dtype=torch.double))
        compare_mdls(combine_model, oggm_model)

        # Now run and compare in equilibrium
        oggm_model.run_until_equilibrium()
        combine_model.run_until_equilibrium()

        compare_mdls(combine_model, oggm_model)
