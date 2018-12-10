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
from cobbi.core.table_creation import create_case_table
import salem
from oggm import cfg
import matplotlib.colors as colors
from cobbi.core.visualization import MidpointNormalize, truncate_colormap,\
    imshow_ic, plot_glacier_contours, add_colorbar, get_axes_coords

cfg.initialize()

basedir = '/home/philipp/HR_01'
output_dir = '/home/philipp/HR_01'
file_extension = 'pdf'

case = test_cases.Trikora
figsize = (4.5, 3)
for case in [test_cases.BordenHR]:  # [test_cases.Borden, test_cases.Giluwe]:
    gdir = NonRGIGlacierDirectory(case, basedir)

    spinup_surf = salem.GeoTiff(gdir.get_filepath('spinup_dem')).get_vardata()
    reference_surf = salem.GeoTiff(gdir.get_filepath('ref_dem')).get_vardata()
    bed_2d = salem.GeoTiff(gdir.get_filepath('dem')).get_vardata()

    ref_ice_mask = np.load(gdir.get_filepath('ref_ice_mask'))
    ref_it = np.load(gdir.get_filepath('ref_ice_thickness'))
    spinup_it = np.load(gdir.get_filepath('spinup_ice_thickness'))
    masked_ice_thick_end = np.ma.masked_array(ref_it,
                                              mask=np.logical_not(ref_ice_mask))
    #masked_ice_thick_start = np.ma.masked_array(spinup_it,
    #                                            mask=np.logical_not(ref_ice_mask))
    masked_reference_surf = np.ma.masked_array(reference_surf,
                                               mask=np.logical_not(ref_ice_mask))

    #plt.ioff()

    cmap = plt.get_cmap('terrain')
    terrain_cmap = truncate_colormap(cmap, 0.3, 0.8)
    # ---------------------------------------------------------------
    # plot true bed height
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(get_axes_coords(case))
    im_b = imshow_ic(ax, bed_2d, case, cmap=terrain_cmap, ticks='scalebar')
    cbar = add_colorbar(fig, ax, im_b)
    cbar.set_label('bed elevation A.S.L (m)')
    #plt.title('Bed of case {:s}, dx={:d}m'.format(case.name, case.dx))
    fname = '{:s}_bed.{:s}'.format(case.name, file_extension)
    #plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname))
    plt.close(fig)

    cmap = plt.get_cmap('Blues_r')
    cut_gray_cmap = truncate_colormap(cmap, 0.3, 0.9)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(get_axes_coords(case))
    im_b = imshow_ic(ax, bed_2d, case, cmap=terrain_cmap, ticks='scalebar')
    #plt.title('Ice surface height, {:s}, dx={:d}m, t={:d}a'.format(
    #    case.name, case.dx, y))
    im_i = imshow_ic(ax, masked_reference_surf, case, cmap=cut_gray_cmap,
                     ticks='scalebar')
    cbar = add_colorbar(fig, ax, im_i)
    cbar.set_label('ice surface elevation A.S.L. (m)')
    plot_glacier_contours(ax, ref_ice_mask, case) #, linestyles='solid')
    fname = '{:s}_surf_height.{:s}'.format(case.name, file_extension)
    #plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname))
    plt.close(fig)

    cmap = plt.get_cmap('gist_heat')
    it_cmap = truncate_colormap(cmap, 0.05, 0.9)
    if not np.all(np.equal(ref_it, spinup_it)):
        print('not in equilibrium! consider also plotting spinup')
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(get_axes_coords(case))
    im_it = imshow_ic(ax, masked_ice_thick_end, case, cmap=it_cmap, ticks='scalebar')
    cbar = add_colorbar(fig, ax, im_it)
    cbar.set_label('ice thickness (m)')
    fname = '{:s}_ice_thickness.{:s}'.format(case.name, file_extension)
    plot_glacier_contours(ax, ref_ice_mask, case)
    #plt.show()
    #plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname))
    plt.close(fig)

    #For evaluation
    gradients = np.gradient(bed_2d, case.dx)
    gradients = np.sqrt(gradients[0] ** 2 + gradients[1] ** 2)
    angle = np.rad2deg(np.arctan(gradients))
    print(np.max(angle))
    print(np.min(angle))

    #plt.ion()

#inner_mask = torch.zeros(ice_mask.shape)
#inner_mask[1:-1, 1:-1] = torch.conv2d(torch.tensor([[ice_mask]], dtype=torch.float), torch.ones((1, 1, 3,3))) == 9
#plt.figure()
#plt.imshow(ice_mask - inner_mask)
#plt.show()