import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from cobbi.core import gis, test_cases
from cobbi.core.utils import NonRGIGlacierDirectory
from cobbi.core.first_guess import compile_first_guess
from cobbi.core.inversion import InversionDirectory
from cobbi.core.dynamics import create_glacier
from cobbi.core.cost_function import create_cost_func
from cobbi.core.inversion import InversionDirectory
from cobbi.core import data_logging
import salem
from oggm import cfg
import matplotlib.colors as colors

np.seed = 0  # needs to be fixed for reproducible results with noise

cfg.initialize()

basedir = '/path/to/example'
basedir = '/data/philipp/thesis_test2/Giluwe/plot_case'

# Choose a case
case = test_cases.Giluwe
gdir = NonRGIGlacierDirectory(case, basedir)
# only needed once:
gis.define_nonrgi_glacier_region(gdir)

# create settings for inversion
lambdas = np.zeros(4)

minimize_options = {
    'maxiter': 300,
    'ftol': 0.5e-3,
    #'xtol': 1e-30,
    'gtol': 1e-4,
    #'maxcor': 5,
    #'maxls': 10,
    'disp': True
}

gdir.write_inversion_settings(mb_spinup=None,
                              yrs_spinup=2000,
                              yrs_forward_run=200,
                              reg_parameters=lambdas,
                              solver='L-BFGS-B',
                              minimize_options=minimize_options,
                              inversion_subdir='0',
                              fg_shape_factor=1.,
                              fg_slope_cutoff_angle=2.5,
                              fg_min_height=-30,
                              bounds_min_max=(2, 1000)
                              )
y_spinup = gdir.inversion_settings['yrs_spinup']
y_end = gdir.inversion_settings['yrs_forward_run'] + y_spinup
# Optional, if not reset=True and already ran once
# only needed once:
create_glacier(gdir)
first_bed_guess = compile_first_guess(gdir)


spinup_surf = salem.GeoTiff(gdir.get_filepath('spinup_dem')).get_vardata()
reference_surf = salem.GeoTiff(gdir.get_filepath('ref_dem')).get_vardata()
bed_2d = salem.GeoTiff(gdir.get_filepath('dem')).get_vardata()

ref_ice_mask = np.load(gdir.get_filepath('ref_ice_mask'))
ice_mask_for_plot = np.ma.masked_array(np.full(ref_ice_mask.shape, 1),
                                       mask=np.logical_not(ref_ice_mask))
ref_it = np.load(gdir.get_filepath('ref_ice_thickness'))
spinup_it = np.load(gdir.get_filepath('spinup_ice_thickness'))
masked_ice_thick_end = np.ma.masked_array(ref_it,
                                          mask=np.logical_not(ref_ice_mask))
masked_ice_thick_start = np.ma.masked_array(spinup_it,
                                            mask=np.logical_not(ref_ice_mask))
masked_reference_surf = np.ma.masked_array(reference_surf,
                                           mask=np.logical_not(ref_ice_mask))

plt.ioff()

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    #see: https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('terrain')
new_cmap = truncate_colormap(cmap, 0.3, 0.8)

f = plt.figure()
im_b = plt.imshow(bed_2d, cmap=new_cmap)
cbar = plt.colorbar(im_b)
cbar.set_label('Bed  height A.S.L (m)')
plt.title('Bed of case {:s}, dx={:d}m'.format(case.name, case.dx))
fname = '{:s}_bed.png'.format(case.name)
plt.savefig(os.path.join(gdir.dir, fname))
plt.clf()

f = plt.figure()
im_b = plt.imshow(first_bed_guess, cmap=new_cmap)
cbar = plt.colorbar(im_b)
cbar.set_label('Bed  height A.S.L (m)')
plt.title('First Bed Guess of case {:s}, dx={:d}m'.format(case.name, case.dx))
fname = '{:s}_first_guess.png'.format(case.name)
plt.savefig(os.path.join(gdir.dir, fname))
plt.clf()


class MidpointNormalize(colors.Normalize):
    # see: https://matplotlib.org/users/colormapnorms.html#custom-normalization-two-linear-ranges
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


db = first_bed_guess - bed_2d
f = plt.figure()
im_b = plt.imshow(db, cmap='RdBu_r',
                  norm=MidpointNormalize(midpoint=0., vmin=db.min(),
                                         vmax=db.max()))
cbar = plt.colorbar(im_b)
cbar.set_label('$\Delta$Bed  height A.S.L (m)')
plt.title('Error of First Guess of case {:s}, dx={:d}m'.format(case.name,
                                                               case.dx))
fname = '{:s}_first_guess_error.png'.format(case.name)
plt.savefig(os.path.join(gdir.dir, fname))
plt.clf()

f = plt.figure()
im_b = plt.imshow(bed_2d, cmap=new_cmap)
cbar = plt.colorbar(im_b)
cbar.set_label('Bed  height A.S.L (m)')
plt.title('Surface of case {:s}, dx={:d}m, t={:d}a'.format(case.name, case.dx,
                                                          y_end))
plt.imshow(ice_mask_for_plot, 'binary', alpha=0.5)
fname = '{:s}_surf_y{:d}.png'.format(case.name, y_end)
plt.savefig(os.path.join(gdir.dir, fname))
plt.clf()

cmap = plt.get_cmap('Blues_r')
cut_gray_cmap = truncate_colormap(cmap, 0.5, 0.99)

for y in [y_spinup, y_end]:
    f = plt.figure()
    im_b = plt.imshow(bed_2d, cmap=new_cmap)
    plt.title('Ice surface height, {:s}, dx={:d}m, t={:d}a'.format(
        case.name, case.dx, y))
    im_i = plt.imshow(masked_reference_surf, cut_gray_cmap)
    cbar = plt.colorbar(im_i)
    cbar.set_label('Surface height A.S.L. (m)')
    fname = '{:s}_surf_height_y{:d}.png'.format(case.name, y)
    plt.savefig(os.path.join(gdir.dir, fname))
    plt.clf()

#For evaluation
gradients = np.gradient(bed_2d, case.dx)
gradients = np.sqrt(gradients[0] ** 2 + gradients[1] ** 2)
angle = np.rad2deg(np.arctan(gradients))
print(np.max(angle))
print(np.min(angle))

plt.ion()

#inner_mask = torch.zeros(ice_mask.shape)
#inner_mask[1:-1, 1:-1] = torch.conv2d(torch.tensor([[ice_mask]], dtype=torch.float), torch.ones((1, 1, 3,3))) == 9
#plt.figure()
#plt.imshow(ice_mask - inner_mask)
#plt.show()