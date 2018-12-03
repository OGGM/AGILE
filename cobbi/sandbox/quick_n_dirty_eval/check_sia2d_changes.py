import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from cobbi.core.sia2d_adapted import Upstream2D as Up2D_new
from oggm.core.sia2d import Upstream2D as Up2D_old
from cobbi.core.test_cases import Giluwe, Borden
from cobbi.core.utils import NonRGIGlacierDirectory
from cobbi.core import gis
from cobbi.core.massbalance import ClippedLinearMassBalance
from cobbi.core.visualization import MidpointNormalize
from cobbi.core.visualization import plot_differences_discrete_cmap
from oggm import cfg
import salem

cfg.initialize()
basedir = '/home/philipp/sia2d'

case = Giluwe
y0 = 0
y_spinup_end = 2000
y_end = 2200

gdir = NonRGIGlacierDirectory(case, basedir)
gis.define_nonrgi_glacier_region(gdir)

ds = salem.GeoTiff(gdir.get_filepath('dem'))
bed_2d = torch.tensor(ds.get_vardata(), dtype=torch.float, requires_grad=False)

mb = case.get_mb_model()
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

np.save(os.path.join(basedir, '{:s}_old.npy'.format(case.name)), old[1])
np.save(os.path.join(basedir, '{:s}_new.npy'.format(case.name)), new[1])

diff = (new[1]-old[1]).detach().numpy()
cbar_min, cbar_max = np.floor(diff.min() * 2) / 2, np.ceil(diff.max() * 2) / 2
cbar_min_max = max(abs(cbar_min), abs(cbar_max))
norm = MidpointNormalize(midpoint=0., vmin=-cbar_min_max,
                         vmax=cbar_min_max)
plot_differences_discrete_cmap(diff, os.path.join(basedir,
                                        '{:s}_diff.pdf'.format(case.name)),
                     case, cbar_min=cbar_min, cbar_max=cbar_max, norm=norm,
                     cbar_label='Surface elevation difference (m)',
                               cmap='PuOr_r')

#np.testing.assert_allclose(old[0], new[0])
#np.testing.assert_allclose(old[1], new[1])
print('end')