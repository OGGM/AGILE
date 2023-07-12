# import matplotlib
# matplotlib.use('Qt5Agg')
# import seaborn as sns
# sns.set_style('ticks')
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from agile2d.core.arithmetics import RMSE, mean_BIAS
from agile2d.core import test_cases
from agile2d.core.visualization import MidpointNormalize, imshow_ic, \
    plot_glacier_contours, add_colorbar, get_axes_coords

output_dir = '/home/philipp/final/plots/gradient_verification'
basedir = '/home/philipp/final/gradient_verification'
file_extension = 'pdf'

case = test_cases.Giluwe
figsize = (4.5, 3)
#for case in [test_cases.Giluwe, test_cases.Borden]:
bdir = os.path.join(basedir, case.name)
pytorch_grad = np.load(os.path.join(bdir, 'pytorch.npy'))
fin_diff_grad = np.load(os.path.join(bdir, 'fd_db_0.1.npy'))
ice_mask = np.load(os.path.join(bdir, 'ref_ice_mask.npy'))
cbar_min = -1.1
cbar_max = 1.1
cbar_min_max = max(abs(cbar_min), abs(cbar_max))
#cbar_min_max = max(abs(cbar_min), abs(cbar_max))

my_cmap = plt.get_cmap('BrBG')
cbar_steps = 12
cmap = ListedColormap(my_cmap(np.linspace(0, 1, cbar_steps - 1,
                                          endpoint=True)))
cbar_min_max = max(abs(cbar_min), abs(cbar_max))
bounds = np.linspace(-cbar_min_max, cbar_min_max, cbar_steps)
bounds_step = bounds[1] - bounds[0]
bounds = bounds[
    np.logical_and(bounds + bounds_step >= cbar_min,
                   bounds - bounds_step <= cbar_max)]

fig = plt.figure(figsize=figsize)
ax = fig.add_axes(get_axes_coords(case))
norm = MidpointNormalize(midpoint=0., vmin=-cbar_min_max, vmax=cbar_min_max)
im_p = imshow_ic(ax, pytorch_grad, case, cmap=cmap, ticks=False,
                 norm=norm, vmin=-cbar_min_max, vmax=cbar_min_max)
cbar = add_colorbar(fig, ax, im_p, norm=norm, extend='both', boundaries=bounds)
cbar.set_label('Gradient of cost function (m$^{-1}$)')
cbar.remove()
plot_glacier_contours(ax, ice_mask, case)
fname = '{:s}_pytorch_grad.{:s}'.format(case.name, file_extension)
plt.savefig(os.path.join(output_dir, fname))
plt.close(fig)

fig = plt.figure(figsize=figsize)
ax = fig.add_axes(get_axes_coords(case))
norm = MidpointNormalize(midpoint=0., vmin=-cbar_min_max, vmax=cbar_min_max)
im_f = imshow_ic(ax, pytorch_grad, case, cmap=cmap, ticks=False,
                 norm=norm, vmin=-cbar_min_max, vmax=cbar_min_max)
cbar = add_colorbar(fig, ax, im_f, norm=norm, extend='both', boundaries=bounds)
cbar.set_label('Gradient of cost function (m$^{-1}$)')
plot_glacier_contours(ax, ice_mask, case)
fname = '{:s}_fin_diff_grad.{:s}'.format(case.name, file_extension)
plt.savefig(os.path.join(output_dir, fname))
plt.close(fig)

# cmap = ListedColormap(my_cmap(np.linspace(0, 1, cbar_steps - 1,
#                                          endpoint=True)))
# cbar_min_max = max(abs(cbar_min), abs(cbar_max))
# bounds = np.linspace(-cbar_min_max, cbar_min_max, cbar_steps)
# bounds_step = bounds[1] - bounds[0]
# bounds = bounds[
#    np.logical_and(bounds + bounds_step >= cbar_min,
#                   bounds - bounds_step <= cbar_max)]
abs_diff = pytorch_grad - fin_diff_grad
#cbar_min_max = max(abs(abs_diff.min()), abs(abs_diff.max()))
fig = plt.figure(figsize=figsize)
ax = fig.add_axes(get_axes_coords(case))
norm = MidpointNormalize(midpoint=0., vmin=-cbar_min_max, vmax=cbar_min_max)
im_f = imshow_ic(ax, abs_diff, case, cmap=cmap, ticks=False,
                 norm=norm, vmin=-cbar_min_max, vmax=cbar_min_max)
cbar = add_colorbar(fig, ax, im_f, norm=norm, boundaries=bounds, extend='both')
plot_glacier_contours(ax, ice_mask, case)
cbar.set_label('$\Delta$ Gradient of cost function (m$^{-1}$)')
#cbar.set_clim(abs_diff.min(), abs_diff.max())
fname = '{:s}_abs_diff_grad.{:s}'.format(case.name, file_extension)
plt.savefig(os.path.join(output_dir, fname))
plt.close(fig)

x_data = fin_diff_grad.flatten()
y_data = pytorch_grad.flatten()
lin_fit = np.polyfit(x=x_data, y=y_data, deg=1)
print(lin_fit)
poly = np.poly1d(lin_fit)

fig, ax = plt.subplots(figsize=(4.5, 3.5))
ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.scatter(x_data, y_data, alpha=0.8, s=7)
ax.plot(x_data, poly(x_data), color='r', linestyle='dotted',
        label='$y \\approx {:.3f} x - {:.3f}$\n(fitted)'.format(poly[1],
                                                                -poly[0]))
min = np.min([pytorch_grad, fin_diff_grad])
max = np.max([pytorch_grad, fin_diff_grad])
ax.plot([min, max], [min, max], color='k', linestyle='dotted',
        label='$y = x$')
ax.legend()
ax.set_xlabel('Finite difference derivative (m$^{-1}$)')
ax.set_ylabel('PyTorch derivative (m$^{-1}$)')
# plt.axis('equal')
ax.yaxis.set_ticks_position('right')
ax.yaxis.set_label_position('right')
plt.tight_layout()
# plt.show()
fname = '{:s}_scatter_plot_grad.{:s}'.format(case.name, file_extension)
plt.savefig(os.path.join(output_dir, fname))
plt.close(fig)

print('Correlation coefficient:')
print(np.corrcoef(x_data, y_data))
print('Bias:')
print(mean_BIAS(x_data, y_data))
print('RMSE:')
print(RMSE(x_data, y_data))
print('Abs mean:')
print(np.abs(x_data).mean())
print(np.abs(y_data).mean())
y_data2 = 1 / poly[1] * y_data
print('Bias2:')
print(mean_BIAS(x_data, y_data2))
print('RMSE2:')
print(RMSE(x_data, y_data2))

print('end')
