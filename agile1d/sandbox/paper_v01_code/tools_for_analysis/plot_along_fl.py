# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python [conda env:oggm_env]
#     language: python
#     name: conda-env-oggm_env-py
# ---

# # Imports

# +
import numpy as np
import copy

from get_reference_glaciers import get_fl_at_year


# -

# # Plot bed_h, volume 1980 and 2020

def add_bed_h_and_volume_along_fl(axs, ds, glacier, glacier_state,
                                   iterations=np.arange(0, 21, 10),
                                   title='', scale_by_length=True,
                                  ):
    fl_ref_1980 = get_fl_at_year(glacier, glacier_state, 1980)
    fl_ref_2020 = get_fl_at_year(glacier, glacier_state, 2020)
    
    def get_diff_per_iteration(iteration, ds):
        if iteration in ds.iteration:
            iteration = iteration
        else:
            iteration = ds.iteration[-1].item()

        fl_iter = ds.flowlines.loc[{'iteration': iteration}].item()
        sfc_h_start = ds.sfc_h_start.loc[{'iteration': iteration}].values
        dis_along_fl = fl_iter.dis_on_line / fl_iter.dx * fl_iter.dx_meter
        
        def get_volume(fl):
            return fl.section * fl.dx_meter
        
        bed_h_diff = fl_iter.bed_h - fl_ref_1980.bed_h
        volume_2020_diff = (get_volume(fl_iter) - get_volume(fl_ref_2020)) / 1e9
        if scale_by_length:
            volume_2020_diff /= fl_iter.dx_meter
    
        fl_iter_1980 = copy.deepcopy(fl_iter)
        fl_iter_1980.surface_h = sfc_h_start
        volume_1980_diff = (get_volume(fl_iter_1980) - get_volume(fl_ref_1980)) / 1e9
        if scale_by_length:
            volume_1980_diff /= fl_iter_1980.dx_meter
    
        return dis_along_fl, bed_h_diff, volume_1980_diff, volume_2020_diff, iteration
    
    def set_max_xlim(data, max_xlim):
        current_max = np.argmax(data == 0)
    
        if current_max > max_xlim:
            return current_max
        else:
            return max_xlim
    
    max_xlim = 0
    
        
    for iter in iterations:
        dis_along_fl, bed_h_diff, volume_1980_diff, volume_2020_diff, iter = get_diff_per_iteration(iter, ds)

        dis_along_fl /= 1000
        if iter == 0:
            label = '0 (fg)'
        else:
            label = iter
    
        axs[0].plot(dis_along_fl,
                    bed_h_diff,
                    label=label)
        axs[1].plot(dis_along_fl, 
                    volume_1980_diff,
                    label=label)
        axs[2].plot(dis_along_fl,
                    volume_2020_diff,
                    label=label)
    
        max_xlim = set_max_xlim(bed_h_diff, max_xlim)
        max_xlim = set_max_xlim(volume_1980_diff, max_xlim)
        max_xlim = set_max_xlim(volume_2020_diff, max_xlim)

    axs[0].set_title(title)

    axs[0].set_ylabel('DIF_BED\n[m]')
    if scale_by_length:
        ylabel_1980 = 'DIF_V_1980\n[km³] per\nunit length'
    else:
        ylabel_1980 = 'DIF_V_1980\n[km³]'
    if scale_by_length:
        ylabel_2020 = 'DIF_V_2020\n[km³] per\nunit length'
    else:
        ylabel_2020 = 'DIF_V_2020\n[km³]'
    axs[1].set_ylabel(ylabel_1980)
    axs[2].set_ylabel(ylabel_2020)

    axs[2].set_xlabel('Distance along flowline [km]')
        
    for i, ax in enumerate(axs):
        ax.set_xlim([0, dis_along_fl[max_xlim + 2]])

        if i in [0, 1]:
            ax.set_xticklabels([])

    axs[1].legend(title='Iteration',
                  loc='center left',
                  bbox_to_anchor=(1, 0.5)
                 )
