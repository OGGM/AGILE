import torch
import numpy as np

from cobbi.sia2d_adapted import Upstream2D as Up2D_new
from oggm.core.sia2d import Upstream2D as Up2D_old
from cobbi.utils import test_cases

from oggm import cfg
from oggm import utils

from os import path
import salem

from cobbi.utils.synthetic_ice_caps import NonRGIGlacierDirectory
from cobbi.utils.synthetic_ice_caps \
    import define_nonrgi_glacier_region, smooth_dem_borders
from cobbi.utils.massbalance_pytorch \
    import LinearMassBalance

cfg.initialize()

case = test_cases.arderin
y0 = 0
y_spinup_end = 2000
y_end = 2800

entity = {'min_x': case.extent[0, 0],
          'max_x': case.extent[1, 0],
          'min_y': case.extent[0, 1],
          'max_y': case.extent[1, 1],
          'name': case.name}

# Local working directory (where OGGM will write its output)
WORKING_DIR = path.join(path.expanduser('~'), 'tmp',
                        'synthetic_ice_cap', case.name)
utils.mkdir(WORKING_DIR, reset=False)
cfg.PATHS['working_dir'] = WORKING_DIR

gdir = NonRGIGlacierDirectory(entity)
define_nonrgi_glacier_region(gdir, dx=case.dx)
smooth_dem_borders(gdir, px_count=case.smooth_border_px)
ds = salem.GeoTiff(gdir.get_filepath('dem', filesuffix='_smooth_border'))
bed_2d = torch.tensor(ds.get_vardata(), dtype=torch.float, requires_grad=False)

mb = LinearMassBalance(case.ela_h, grad=case.mb_grad)
# Create glacier
with torch.no_grad():
    reference_model = Up2D_old(bed_2d.detach().numpy(),
                               dx=case.dx, mb_model=mb, y0=0,
                               glen_a=cfg.PARAMS['glen_a'],
                               ice_thick_filter=None)
    reference_model.run_until(y_spinup_end)
    start_surf = torch.tensor(reference_model.surface_h,
                                  dtype=torch.float, requires_grad=False)
    reference_model.run_until(y_end)
    reference_surf = torch.tensor(reference_model.surface_h,
                                  dtype=torch.float, requires_grad=False)
    ice_mask = (reference_surf > bed_2d)
    old = [start_surf, reference_surf, bed_2d]

with torch.no_grad():
    reference_model = Up2D_new(bed_2d, dx=case.dx, mb_model=mb, y0=0,
                               glen_a=cfg.PARAMS['glen_a'],
                               ice_thick_filter=None)
    reference_model.run_until(y_spinup_end)
    start_surf = torch.tensor(reference_model.surface_h,
                                  dtype=torch.float, requires_grad=False)
    reference_model.run_until(y_end)
    reference_surf = torch.tensor(reference_model.surface_h,
                                  dtype=torch.float, requires_grad=False)
    ice_mask = (reference_surf > bed_2d)
    new = [start_surf, reference_surf, bed_2d]

np.testing.assert_allclose(old[0], new[0])
np.testing.assert_allclose(old[1], new[1])