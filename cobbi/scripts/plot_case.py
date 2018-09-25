import torch
import numpy as np
import matplotlib.pyplot as plt
from cobbi.utils import test_cases
from cobbi.inversion import *
import matplotlib.colors as colors

case = test_cases.blafell

y0 = 0
y_spinup_end = 2000
y_end = 2800
start_surf, reference_surf, ice_mask, mb, bed_2d = spin_up(case, y_spinup_end,
                                                           y_end)

start_surf = start_surf.detach().numpy()
reference_surf = reference_surf.detach().numpy()
ice_mask = ice_mask.detach().numpy()
bed_2d = bed_2d.detach().numpy()

ref_ice_mask = (reference_surf - bed_2d) == 0

ice_mask_for_plot = np.ma.masked_array(np.full(ice_mask.shape, 1),
                                       mask=ref_ice_mask)

masked_ice_thick_end = np.ma.masked_array(reference_surf - bed_2d,
                                          mask=ref_ice_mask)
masked_ice_thick_start = np.ma.masked_array(start_surf - bed_2d,
                                            mask=ref_ice_mask)
masked_reference_surf = np.ma.masked_array(reference_surf,
                                           mask=ref_ice_mask)

basedir = '/data/philipp/tests/cases/'
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
plt.savefig(basedir + '{:s}_bed.png'.format(case.name))
plt.clf()


f = plt.figure()
im_b = plt.imshow(bed_2d, cmap=new_cmap)
cbar = plt.colorbar(im_b)
cbar.set_label('Bed  height A.S.L (m)')
plt.title('Surface of case {:s}, dx={:d}m, y={:d}'.format(case.name, case.dx,
                                                          y_end))
plt.imshow(ice_mask_for_plot, 'binary', alpha=0.5)

plt.savefig(basedir + '{:s}_surf_y{:d}.png'.format(case.name, y_end))
plt.clf()

cmap = plt.get_cmap('Blues_r')
cut_gray_cmap = truncate_colormap(cmap, 0.5, 0.95)

f = plt.figure()
im_b = plt.imshow(bed_2d, cmap=new_cmap)
plt.title('Height of ice surface, {:s}, dx={:d}m, y={:d}'.format(
    case.name, case.dx, y_end))
im_i = plt.imshow(masked_reference_surf, cut_gray_cmap)
cbar = plt.colorbar(im_i)
cbar.set_label('Surface height A.S.L. (m)')

plt.savefig(basedir + '{:s}_surf_height_y{:d}.png'.format(case.name, y_end))
plt.clf()

plt.ion()

inner_mask = torch.zeros(ice_mask.shape)
inner_mask[1:-1, 1:-1] = torch.conv2d(torch.tensor([[ice_mask]], dtype=torch.float), torch.ones((1, 1, 3,3))) == 9
plt.figure()
plt.imshow(ice_mask - inner_mask)
plt.show()