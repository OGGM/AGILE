import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import numpy as np
import glob
import os
import salem
from cobbi.core.data_logging import load_pickle
from cobbi.sandbox.quick_n_dirty_eval import experiment_naming_engine
from cobbi.core.visualization import MidpointNormalize, truncate_colormap,\
    imshow_ic, plot_glacier_contours, add_colorbar, get_axes_coords,\
    plot_bed_difference, plot_surf_difference, plot_differences
from cobbi.core import test_cases
from cobbi.core.utils import NonRGIGlacierDirectory
from oggm import cfg
from scipy.signal import convolve2d


cfg.initialize()

output_dir = '/media/philipp/Daten/Dokumente/Studium/Master/Masterarbeit' \
             '/iteration_plots'
basedir = '/home/philipp/thesis/identical_twin/'
file_extension = 'png'
case = test_cases.Giluwe
dx = case.dx
gdir = NonRGIGlacierDirectory(case, basedir)

figsize = (4.5, 3)

dl = load_pickle(os.path.join(gdir.dir, 'identical twin', 'data_logger.pkl'))
ref_surf = salem.GeoTiff(gdir.get_filepath('ref_dem')).get_vardata()
ref_ice_mask = np.load(gdir.get_filepath('ref_ice_mask'))
#true_bed = salem.GeoTiff(gdir.get_filepath('dem')).get_vardata()

ref_inner_mask = compute_inner_mask(ref_ice_mask)
#plt.figure()
#plt.imshow(ref_ice_mask)
#plt.imshow(ref_inner_mask, cmap='RdBu')
#plt.show()
inversion_settings = load_pickle(os.path.join(gdir.dir, 'identical twin',
                                              'inversion_settings.pkl'))
reg_parameters = inversion_settings['reg_parameters']
margin = np.logical_xor(ref_ice_mask, ref_inner_mask)

def get_costs_arr(reg_parameters, ref_surf, ref_ice_mask, ref_inner_mask,
                  guessed_bed, model_surf, model_ice_mask, model_inner_mask,
                  dx):

    margin = np.logical_xor(ref_ice_mask, ref_inner_mask)
    cost = np.zeros(len(reg_parameters) + 1).tolist()

    # TODO recheck all indices for reg_parameters and cost
    cost[-1] = ((ref_surf - model_surf) * np.logical_not(margin))**2  # Big
    #  TODO
    cost[0] = reg_parameters[0] * \
              ((ref_surf - model_surf) * margin)**2

    if reg_parameters[1] != 0:
        # penalizes ice thickness, where ice thickness should be 0
        cost[1] = reg_parameters[1] * (((model_surf - guessed_bed)
                                        * np.logical_not(ref_ice_mask))**2)

    if reg_parameters[2] != 0:
        # penalize large derivatives of bed under glacier
        # -> avoids numerical instabilites
        db_dx1 = (guessed_bed[:, :-2] - guessed_bed[:, 1:-1]) / dx
        db_dx2 = (guessed_bed[:, 1:-1] - guessed_bed[:, 2:]) / dx
        db_dy1 = (guessed_bed[:-2, :] - guessed_bed[1:-1, :]) / dx
        db_dy2 = (guessed_bed[1:-1, :] - guessed_bed[2:, :]) / dx
        db_dx_sq = 0.5 * (db_dx1**2 + db_dx2**2) * ref_ice_mask[:,
                                                           1:-1]
        db_dx_sq_full = np.zeros(guessed_bed.shape)
        db_dx_sq_full[:, 1:-1] = db_dx_sq
        db_dy_sq = 0.5 * (db_dy1**2 + db_dy2**2) * ref_ice_mask[1:-1,
                                                           :]
        db_dy_sq_full = np.zeros(guessed_bed.shape)
        db_dy_sq_full[1:-1, :] = db_dy_sq
        cost[2] = reg_parameters[2] * 0.5 * (db_dx_sq_full + db_dy_sq_full)
        # TODO: think about first squaring forward and backward and then adding vs adding and then squaring
        # then an additional .abs() is required for db_dx1, ...

    if reg_parameters[3] != 0:
        # penalize high curvature of bed exactly at boundary pixels of
        # glacier for a smooth transition from glacier-free to glacier
        ddb_dx = (guessed_bed[:, :-2] + guessed_bed[:, 2:]
                  - 2 * guessed_bed[:, 1:-1]) / dx ** 2
        ddb_dy = (guessed_bed[:-2, :] + guessed_bed[2:, :]
                  - 2 * guessed_bed[1:-1, :]) / dx ** 2
        ddb_dx = ddb_dx * np.logical_xor(model_ice_mask, model_inner_mask)[:, 1:-1]
        ddb_dx_full = np.zeros(guessed_bed.shape)
        ddb_dx_full[:, 1:-1] = ddb_dx
        ddb_dy = ddb_dy * np.logical_xor(model_ice_mask, model_inner_mask)[1:-1, :]
        ddb_dy_full = np.zeros(guessed_bed.shape)
        ddb_dy_full[1:-1, :] = ddb_dy
        cost[3] = reg_parameters[3] * (ddb_dx_full**2 + ddb_dy_full**2)

    if len(reg_parameters) > 4 and reg_parameters[4] != 0:
        # penalize high curvature of surface in glacier bounds
        dds_dx = (model_surf[:, :-2] + model_surf[:, 2:]
                  - 2 * model_surf[:, 1:-1]) / dx ** 2
        dds_dy = (model_surf[:-2, :] + model_surf[2:, :]
                  - 2 * model_surf[1:-1, :]) / dx ** 2
        dds_dx = dds_dx * model_inner_mask[:, 1:-1]
        dds_dy = dds_dy * model_inner_mask[1:-1, :]
        cost[4] = reg_parameters[4] \
                  * (dds_dx**2 + dds_dy**2)

    if len(reg_parameters) > 5 and reg_parameters[5] != 0:
        # penalize large derivatives of surface
        # -> avoids numerical instabilites
        ds_dx1 = (model_surf[:, :-2] - model_surf[:, 1:-1]) / dx
        ds_dx2 = (model_surf[:, 1:-1] - model_surf[:, 2:]) / dx
        ds_dy1 = (model_surf[:-2, :] - model_surf[1:-1, :]) / dx
        ds_dy2 = (model_surf[1:-1, :] - model_surf[2:, :]) / dx
        ds_dx_sq = 0.5 * (ds_dx1**2
                          + ds_dx2**2) * model_inner_mask[:, 1:-1]
        ds_dy_sq = 0.5 * (ds_dy1**2
                          + ds_dy2**2) * model_inner_mask[1:-1, :]
        cost[5] = reg_parameters[5] * 0.5 * ((ds_dx_sq + ds_dy_sq))
        # TODO: think about first squaring forward and backward and then adding vs adding and then squaring
        # then an additional .abs() is required for db_dx1, ...
    return cost



interesting_costs = [0, 1, 2, 3, -1]
cost_names = ['J{:d}'.format(j) for j in range(6)] + ['Raw']

for i in dl.step_indices:
    base_plotpath = 'iteration{:03d}_{:s}.{:s}'.format(i, '{:s}',
                                                      file_extension)
    model_surf = dl.surfs[i]
    guessed_bed = dl.beds[i]
    surf_diff = model_surf - dl.ref_surf
    bed_diff = guessed_bed - dl.true_bed
    model_ice_mask = (model_surf - guessed_bed) > 0
    model_inner_mask = compute_inner_mask(model_ice_mask)
    costs_arr = get_costs_arr(reg_parameters, ref_surf, ref_ice_mask,
                              ref_inner_mask, guessed_bed, model_surf,
                              model_ice_mask, model_inner_mask, dx)

    cmap_bed_diff = plt.get_cmap('seismic')
    cmap_surf_diff = plt.get_cmap('PuOr_r')
    cmap_list = [sns.diverging_palette(240, 0, l=40, s=99, as_cmap=True),
                 sns.diverging_palette(240, 60, l=40, s=99, as_cmap=True),
                 sns.diverging_palette(240, 120, l=40, s=99, as_cmap=True),
                 sns.diverging_palette(240, 180, l=40, s=99, as_cmap=True),
                 None,
                 None,
                 sns.diverging_palette(240, 240, l=40, s=99, as_cmap=True)]

    cbar_min = bed_diff.min()
    cbar_max = bed_diff.max()
    cbar_min_max = max(abs(cbar_min), abs(cbar_max))
    norm = MidpointNormalize(midpoint=0., vmin=-cbar_min_max,
                             vmax=cbar_min_max)
    plotpath = os.path.join(output_dir, 'bed_error',
                            base_plotpath.format('bed_error'))
    plot_bed_difference(bed_diff, plotpath, case,
                        ice_mask=ref_ice_mask,
                        bed_measurements=None,
                        cbar_min=cbar_min, cbar_max=cbar_max,
                        show_cbar=True, norm=norm, cmap=cmap_bed_diff)

    cbar_min = surf_diff.min()
    cbar_max = surf_diff.max()
    cbar_min_max = max(abs(cbar_min), abs(cbar_max))
    norm = MidpointNormalize(midpoint=0., vmin=-cbar_min_max,
                             vmax=cbar_min_max)
    plotpath = os.path.join(output_dir, 'surf_error',
                            base_plotpath.format('surf_error'))
    plot_surf_difference(surf_diff, plotpath, case,
                        ice_mask=ref_ice_mask,
                        bed_measurements=None,
                        cbar_min=cbar_min, cbar_max=cbar_max,
                        show_cbar=True, norm=norm, cmap=cmap_surf_diff)

    for j in interesting_costs:
        cbar_min = 0
        cbar_max = costs_arr[j].max()
        cbar_min_max = max(abs(cbar_min), abs(cbar_max))
        norm = MidpointNormalize(midpoint=0., vmin=-cbar_min_max,
                                 vmax=cbar_min_max)
        plotpath = os.path.join(output_dir, 'cost_' + cost_names[j],
                                base_plotpath.format('cost_' + cost_names[j]))
        plot_differences(costs_arr[j], plotpath, case, ice_mask=ref_ice_mask,
                         bed_measurements=None, cbar_min=cbar_min,
                         cbar_max=cbar_max, show_cbar=True, norm=norm,
                         cmap=cmap_list[j],
                         cbar_label='{:s} (m$^2$)'.format(cost_names[j])
                         )