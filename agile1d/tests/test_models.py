import pytest
import numpy as np
import xarray as xr
from functools import partial
import torch
from torch.autograd import gradcheck
from agile1d.core.massbalance import ConstantMassBalanceTorch, MBModelTorchWrapper
from oggm.core.massbalance import ConstantMassBalance, MonthlyTIModel
from agile1d.core.flowline import MixedBedFlowline
from agile1d.core.flowline import FluxBasedModel as agile_flux_model
from agile1d.core.flowline import SemiImplicitModel as agile_impl_model
from oggm.core.flowline import FluxBasedModel as oggm_flux_model
from oggm.core.flowline import SemiImplicitModel as oggm_impl_model
from oggm.utils._funcs import date_to_floatyear
from oggm import cfg


pytestmark = [pytest.mark.filterwarnings("ignore:<class 'agile1d.core.torch_interp1d.Interp1d'> "
                                         "should not be instantiated.:DeprecationWarning"),
              pytest.mark.test_env("models")]


class TestModels:
    def test_constant_mb_model(self, hef_gdir):
        y0 = 2000
        halfsize = 15
        agile_mb_model = ConstantMassBalanceTorch(hef_gdir, y0=y0, halfsize=halfsize)
        oggm_mb_model = ConstantMassBalance(hef_gdir, y0=y0, halfsize=halfsize)
        test_height_shift = 100.
        agile_mb_model_shift = ConstantMassBalanceTorch(
            hef_gdir, y0=y0, halfsize=halfsize, height_shift=test_height_shift)

        fls = hef_gdir.read_pickle('model_flowlines',
                                   filesuffix='_parabola')
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

        # test annual, multiplication with 1e7 to compare 'large' numbers
        oggm_annual_mbs = oggm_mb_model.get_annual_mb(heights) * 1e7
        agile_annual_mbs = agile_mb_model.get_annual_mb(heights_torch) * 1e7
        agile_annual_mbs_shift = \
            agile_mb_model_shift.get_annual_mb(heights_torch_shift) * 1e7

        assert np.allclose(oggm_annual_mbs, agile_annual_mbs)
        assert isinstance(agile_annual_mbs, torch.Tensor)
        assert agile_annual_mbs.shape == heights_torch.shape
        assert np.allclose(agile_annual_mbs, agile_annual_mbs_shift)
        assert isinstance(agile_annual_mbs_shift, torch.Tensor)
        assert agile_annual_mbs_shift.shape == heights_torch_shift.shape

        # test monthly, multiplication with 1e7 to compare 'large' numbers
        for month in np.arange(12) + 1:
            yr = date_to_floatyear(0, month)
            oggm_monthly_mbs = oggm_mb_model.get_monthly_mb(heights, year=yr) * 1e7
            agile_monthly_mbs = agile_mb_model.get_monthly_mb(heights_torch, year=yr) * 1e7
            agile_monthly_mbs_shift = \
                agile_mb_model_shift.get_monthly_mb(heights_torch_shift, year=yr) * 1e7

            assert np.allclose(oggm_monthly_mbs, agile_monthly_mbs)
            assert isinstance(agile_monthly_mbs, torch.Tensor)
            assert agile_monthly_mbs.shape == heights_torch.shape
            assert np.allclose(agile_monthly_mbs, agile_monthly_mbs_shift)
            assert isinstance(agile_monthly_mbs_shift, torch.Tensor)
            assert agile_monthly_mbs_shift.shape == heights_torch_shift.shape

    def test_mb_model_wrapper(self, hef_gdir):
        # compare wrapper with constant mass balance implementation
        y0 = 2000
        halfsize = 15
        agile_mb_model = ConstantMassBalanceTorch(hef_gdir, y0=y0, halfsize=halfsize)
        oggm_mb_model = ConstantMassBalance(hef_gdir, y0=y0, halfsize=halfsize)
        agile_mb_model_wrapper = MBModelTorchWrapper(hef_gdir, mb_model=oggm_mb_model)

        fls = hef_gdir.read_pickle('model_flowlines',
                                   filesuffix='_parabola')
        h = []
        for fl in fls:
            h = np.append(h, fl.bed_h)
            h = np.append(h, fl.surface_h)
        zminmax = np.round([np.min(h), np.max(h)])
        heights = np.linspace(zminmax[0], zminmax[1], num=20)
        heights_torch = torch.tensor(heights, dtype=torch.double, requires_grad=True)

        # test annual, multiplication with 1e7 to compare 'large' numbers
        agile_annual_mbs = agile_mb_model.get_annual_mb(heights_torch) * 1e7
        wrapper_annual_mbs = agile_mb_model_wrapper.get_annual_mb(heights_torch) * 1e7

        np.testing.assert_allclose(wrapper_annual_mbs.detach().numpy(),
                                   agile_annual_mbs.detach().numpy(),
                                   rtol=3e-4)
        assert isinstance(agile_annual_mbs, torch.Tensor)
        assert isinstance(wrapper_annual_mbs, torch.Tensor)
        assert agile_annual_mbs.shape == heights_torch.shape
        assert wrapper_annual_mbs.shape == heights_torch.shape
        assert gradcheck(agile_mb_model.get_annual_mb, heights_torch)
        assert gradcheck(agile_mb_model_wrapper.get_annual_mb, heights_torch)

        # test monthly, multiplication with 1e7 to compare 'large' numbers
        for month in np.arange(12) + 1:
            yr = date_to_floatyear(0, month)
            agile_monthly_mbs = \
                agile_mb_model.get_monthly_mb(heights_torch, year=yr) * 1e7
            wrapper_monthly_mbs = \
                agile_mb_model_wrapper.get_monthly_mb(heights_torch, year=yr) * 1e7

            np.testing.assert_allclose(wrapper_monthly_mbs.detach().numpy(),
                                       agile_monthly_mbs.detach().numpy(),
                                       rtol=6e-3)
            assert isinstance(agile_monthly_mbs, torch.Tensor)
            assert agile_monthly_mbs.shape == heights_torch.shape
            assert gradcheck(agile_mb_model.get_monthly_mb, (heights_torch, yr))
            assert gradcheck(agile_mb_model_wrapper.get_monthly_mb, (heights_torch, yr))

        # test wrapper with MonthlyTIModel
        oggm_mb_model = MonthlyTIModel(hef_gdir)
        agile_mb_model = MBModelTorchWrapper(hef_gdir, mb_model=oggm_mb_model)

        # test annual, multiplication with 1e7 to compare 'large' numbers
        year = 2010
        oggm_annual_mbs = oggm_mb_model.get_annual_mb(heights, year=year) * 1e7
        agile_annual_mbs = \
            agile_mb_model.get_annual_mb(heights_torch, year=year) * 1e7

        np.testing.assert_allclose(oggm_annual_mbs,
                                   agile_annual_mbs.detach().numpy(),
                                   rtol=8e-4)
        assert isinstance(agile_annual_mbs, torch.Tensor)
        assert agile_annual_mbs.shape == heights_torch.shape
        assert gradcheck(partial(agile_mb_model.get_annual_mb,
                                 year=year),
                         heights_torch)

        # test monthly, multiplication with 1e7 to compare 'large' numbers
        for month in np.arange(12) + 1:
            yr = date_to_floatyear(year, month)
            oggm_monthly_mbs = oggm_mb_model.get_monthly_mb(heights, year=yr) * 1e7
            agile_monthly_mbs = \
                agile_mb_model.get_monthly_mb(heights_torch, year=yr) * 1e7

            np.testing.assert_allclose(oggm_monthly_mbs,
                                       agile_monthly_mbs.detach().numpy(),
                                       rtol=2e-2)
            assert isinstance(agile_monthly_mbs, torch.Tensor)
            assert agile_monthly_mbs.shape == heights_torch.shape
            assert gradcheck(partial(agile_mb_model.get_monthly_mb,
                                     year=yr),
                             heights_torch)

    @pytest.mark.parametrize('model_to_use', [('_parabola', agile_flux_model,
                                               oggm_flux_model),
                                              ('_trapezoidal',
                                               agile_impl_model,
                                               oggm_impl_model)])
    def test_DynamicModels(self, hef_gdir, model_to_use):
        oggm_fl_type, agile_dyn_model, oggm_dyn_model = model_to_use
        # define flowlines
        oggm_fls = hef_gdir.read_pickle('model_flowlines',
                                        filesuffix=oggm_fl_type)
        fl = oggm_fls[0]
        agile_fls = MixedBedFlowline(line=fl.line,
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
        agile_mb_model = ConstantMassBalanceTorch(hef_gdir, y0=y0, halfsize=halfsize)
        oggm_mb_model = ConstantMassBalance(hef_gdir, y0=y0, halfsize=halfsize)

        # define FluxModels
        oggm_model = oggm_dyn_model(oggm_fls, mb_model=oggm_mb_model, y0=0)
        agile_model = agile_dyn_model(agile_fls,
                                          mb_model=agile_mb_model, y0=0)

        # Let models run
        oggm_model.run_until(30.)
        agile_model.run_until(30.)

        # Compare models
        def compare_mdls(agile_mdl, oggm_mdl):
            # actually not needed to check m2 and km3, but it is a test that
            # the conversion is done right in agile
            assert np.isclose(agile_mdl.area_m2, oggm_mdl.area_m2,
                              rtol=1e-4, atol=0.001)
            assert np.isclose(agile_mdl.area_km2, oggm_mdl.area_km2,
                              rtol=1e-4, atol=0.001)
            assert np.isclose(agile_mdl.volume_m3, oggm_mdl.volume_m3,
                              rtol=1e-3, atol=0.001)
            assert np.isclose(agile_mdl.volume_km3, oggm_mdl.volume_km3,
                              rtol=1e-3, atol=0.001)
            assert np.allclose(agile_mdl.fls[0].surface_h,
                               oggm_mdl.fls[0].surface_h,
                               rtol=1e-3, atol=0.01)
            assert np.allclose(agile_mdl.u_stag[0], oggm_mdl.u_stag[0])
            assert np.allclose(agile_mdl.flux_stag[0], oggm_mdl.flux_stag[0])

        compare_mdls(agile_model, oggm_model)

        # test with year as torch tensor
        oggm_model.run_until(40.)
        agile_model.run_until(torch.tensor(40., dtype=torch.double))
        compare_mdls(agile_model, oggm_model)

        # Now run and compare in equilibrium
        oggm_model.run_until_equilibrium()
        agile_model.run_until_equilibrium()

        compare_mdls(agile_model, oggm_model)

        # test run_until_and_store
        agile_model = agile_dyn_model(agile_fls,
                                          mb_model=agile_mb_model,
                                          y0=100)

        ds_run = agile_model.run_until_and_store(
            200, diag_path=hef_gdir.get_filepath('model_diagnostics',
                                                 filesuffix='_test_run',
                                                 delete=True))

        assert ds_run.time[0] == 100
        assert ds_run.time[-1] == 200
        assert len(ds_run.volume_m3) == 101
        assert np.all(np.isfinite(ds_run.volume_m3))
        assert np.all(np.isfinite(ds_run.area_m2))
        assert np.all(np.isfinite(ds_run.length_m))
        assert ds_run.attrs['mb_model_rho'] == cfg.PARAMS['ice_density']
        assert ds_run.attrs['mb_model_device'] == 'cpu'
        assert ds_run.attrs['mb_model_class'] == 'ConstantMassBalanceTorch'

        # compare to what was saved on disk
        fp = hef_gdir.get_filepath('model_diagnostics', filesuffix='_test_run')
        with xr.open_dataset(fp) as ds:
            ds_disk = ds.load()
        assert np.all(ds_run.volume_m3 == ds_disk.volume_m3)
        assert np.all(ds_run.area_m2 == ds_disk.area_m2)
        assert np.all(ds_run.length_m == ds_disk.length_m)
