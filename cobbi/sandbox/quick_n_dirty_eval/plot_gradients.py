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
from cobbi.sandbox.gradver.cost_function import create_cost_func
from cobbi.core.inversion import InversionDirectory
from cobbi.core import data_logging
from cobbi.core.visualization import plot_gradient
from cobbi.core.visualization import MidpointNormalize, truncate_colormap,\
    imshow_ic, plot_glacier_contours, add_colorbar, get_axes_coords


output_dir = '/media/philipp/Daten/Dokumente/Studium/Master/Masterarbeit' \
           '/Thesis/figs/gradient_verification'
basedir = '/home/philipp/thesis/gradient_verification'
file_extension = 'pdf'

case = test_cases.Giluwe
figsize = (4.5, 3)
#for case in [test_cases.Giluwe, test_cases.Borden]:
bdir = os.path.join(basedir, case.name)
pytorch_grad = np.load(os.path.join(bdir, 'pytorch.npy'))
fin_diff_grad = np.load(os.path.join(bdir, 'fd_db_0.1.npy'))
ice_mask = np.load(os.path.join(bdir, 'ref_ice_mask.npy'))
cbar_min = min(pytorch_grad.min(), fin_diff_grad.min())
cbar_max = max(pytorch_grad.max(), fin_diff_grad.max())
cbar_min_max = 0.75
#cbar_min_max = max(abs(cbar_min), abs(cbar_max))

my_cmap = plt.get_cmap('BrBG')

fig = plt.figure(figsize=figsize)
ax = fig.add_axes(get_axes_coords(case))
norm = MidpointNormalize(midpoint=0., vmin=-cbar_min_max, vmax=cbar_min_max)
im_p = imshow_ic(ax, pytorch_grad, case, cmap=my_cmap, ticks=False,
                 norm=norm, vmin=-cbar_min_max, vmax=cbar_min_max)
cbar = add_colorbar(fig, ax, im_p, norm=norm, extend='both')
cbar.set_label('gradient of cost function (m$^{-1}$)')
cbar.remove()
plot_glacier_contours(ax, ice_mask, case)
fname = '{:s}_pytorch_grad.{:s}'.format(case.name, file_extension)
plt.savefig(os.path.join(output_dir, fname))
plt.close(fig)

fig = plt.figure(figsize=figsize)
ax = fig.add_axes(get_axes_coords(case))
norm = MidpointNormalize(midpoint=0., vmin=-cbar_min_max, vmax=cbar_min_max)
im_f = imshow_ic(ax, pytorch_grad, case, cmap=my_cmap, ticks=False,
                 norm=norm, vmin=-cbar_min_max, vmax=cbar_min_max)
cbar = add_colorbar(fig, ax, im_f, norm=norm, extend='both')
cbar.set_label('gradient of cost function (m$^{-1}$)')
plot_glacier_contours(ax, ice_mask, case)
fname = '{:s}_fin_diff_grad.{:s}'.format(case.name, file_extension)
plt.savefig(os.path.join(output_dir, fname))
plt.close(fig)


abs_diff = pytorch_grad - fin_diff_grad
cbar_min_max = max(abs(abs_diff.min()), abs(abs_diff.max()))
fig = plt.figure(figsize=figsize)
ax = fig.add_axes(get_axes_coords(case))
norm = MidpointNormalize(midpoint=0., vmin=-cbar_min_max, vmax=cbar_min_max)
im_f = imshow_ic(ax, abs_diff, case, cmap=my_cmap, ticks=False,
                 norm=norm, vmin=-cbar_min_max, vmax=cbar_min_max)
cbar = add_colorbar(fig, ax, im_f, norm=norm)#, extend='both')
plot_glacier_contours(ax, ice_mask, case)
cbar.set_label('$\Delta$ gradient of cost function (m$^{-1}$)')
fname = '{:s}_abs_diff_grad.{:s}'.format(case.name, file_extension)
plt.savefig(os.path.join(output_dir, fname))
plt.close(fig)