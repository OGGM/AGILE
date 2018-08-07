import torch
torch.utils.backcompat.broadcast_warning.enabled = True
import matplotlib.pyplot as plt

from cobbi.sia2d_pytorch import Upstream2D
from cobbi.sia2d_pytorch_full import Upstream2D as \
    Up2D
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

print(c_0)
print(g)

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

def cost_function2(b):
    init_ice_thick = start_surf - b
    model = Up2D(b, dx=case.dx, mb_model=mb, y0=y0, glen_a=cfg.A,
                       ice_thick_filter=None, init_ice_thick=init_ice_thick)
    model.run_until(y1)
    s = model.surface_h
    cost = (reference_surf - s).pow(2).sum()
    return cost

b_1 = torch.tensor(bed_0, dtype=torch.float, requires_grad=True)
c_1 = cost_function2(b_1)
c_1.backward()

g = b_1.grad
g_1 = g.detach().numpy()

fig = plt.figure()
plt.imshow(reference_model.ice_thick.clone().detach())
plt.show()

fig = plt.figure()
plt.imshow(g_0)
plt.show()

fig = plt.figure()
plt.imshow(g_1)
plt.show()

fig = plt.figure()
plt.imshow(bed_2d)
plt.show()

print((g_0 - g_1).min())
print((g_0 - g_1).max())
print(g_0.min())
print(g_0.max())

print('end')

