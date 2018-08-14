import torch
torch.utils.backcompat.broadcast_warning.enabled = True
import matplotlib.pyplot as plt

from cobbi.sia2d_pytorch_full import Upstream2D
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
from cobbi.inversion import first_guess

# Initialize OGGM and set up the default run parameters
cfg.initialize()

case = test_cases.arderin
entity = {'min_x': case.extent[0, 0],
          'max_x': case.extent[1, 0],
          'min_y': case.extent[0, 1],
          'max_y': case.extent[1, 1],
          'name': case.name}

# Local working directory (where OGGM will write its output)
WORKING_DIR = path.join(path.expanduser('~'), 'tmp', 'synthetic_ice_cap')
utils.mkdir(WORKING_DIR, reset=False)
cfg.PATHS['working_dir'] = WORKING_DIR

gdir = NonRGIGlacierDirectory(entity)
define_nonrgi_glacier_region(gdir, dx=case.dx)
smooth_dem_borders(gdir, px_count=case.smooth_border_px)
ds = salem.GeoTiff(gdir.get_filepath('dem', filesuffix='_smooth_border'))
bed_2d = torch.tensor(ds.get_vardata(), dtype=torch.float)

mb = LinearMassBalance(case.ela_h, grad=case.mb_grad)

# Create glacier
y0 = 850
y1 = 1000
with torch.no_grad():
    reference_model = Upstream2D(bed_2d, dx=case.dx, mb_model=mb, y0=0,
                                 glen_a=cfg.A, ice_thick_filter=None)
    reference_model.run_until(y0)
    start_surf = torch.tensor(reference_model.surface_h,
                                  dtype=torch.float, requires_grad=False)
    reference_model.run_until(y1)
    reference_surf = torch.tensor(reference_model.surface_h,
                                  dtype=torch.float, requires_grad=False)
    ice_mask = (reference_surf > bed_2d)

# define cost_function
def cost_function(b):
    init_ice_thick = start_surf - b
    model = Upstream2D(b, dx=case.dx, mb_model=mb, y0=y0, glen_a=cfg.A,
                       ice_thick_filter=None, init_ice_thick=init_ice_thick)
    model.run_until(y1)
    s = model.surface_h
    cost = (reference_surf - s).pow(2).sum()
    return cost

bed_0 = first_guess(reference_surf.clone().detach().numpy(),
                    ice_mask.clone().detach().numpy(),
                    case.dx)
b_0 = torch.tensor(bed_0, dtype=torch.float, requires_grad=True)
c_0 = cost_function(b_0)
c_0.backward()
g = b_0.grad
g_0 = g.detach().numpy()
del c_0  # free resources?
del g
del b_0

b_0 = torch.tensor(bed_0, dtype=torch.float, requires_grad=False)

# #################################################
# finite difference gradient
import numpy as np
dbs = [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]
rel_diff1 = []
rel_diff2 = []
abs_diff = []
rmse = []

indices_to_check = np.argwhere(ice_mask.numpy())

for db in dbs:
    fin_dif_grad = np.zeros(bed_2d.shape)
    with torch.no_grad():
        J_ref = cost_function(b_0)
        for ind in indices_to_check:
            b_1 = b_0.clone()
            b_1[ind[0], ind[1]] = b_1[ind[0], ind[1]] + db
            J_dev = cost_function(b_1)
            fin_dif_grad[ind[0], ind[1]] = J_dev - J_ref

        fin_dif_grad = fin_dif_grad / db
    rel_diff1.append((fin_dif_grad - g_0) / g_0)
    rel_diff2.append((fin_dif_grad - g_0) / fin_dif_grad)
    abs_diff.append(fin_dif_grad - g_0)
    rmse.append((fin_dif_grad - g_0).pow(2).mean())

plt.figure()
plt.imshow(g_0)
plt.show()

plt.figure()
plt.imshow(fin_dif_grad)
plt.show()

# TODO: evaluate rel_diff, abs_diff and/or rmse systematically for convergence


print('end')

