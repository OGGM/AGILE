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
#     display_name: Python [conda env:agile_env]
#     language: python
#     name: conda-env-agile_env-py
# ---

# +
from calculate_statistics import get_stats_for_all_iterations, create_stat_matrices
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import seaborn as sns

from plot_labels import get_multiple_short_labels_obs
from experiment_file_fg_oggm import experiment_options


# -

# # Plot for statistics for each iteration for one setting

def plot_stats_per_iteration(ax, ds, glacier, glacier_state, xticks, ds_ref=None):
    mad_today_volume, mad_bed_h, mad_past_volume = get_stats_for_all_iterations(
        ds,
        glacier,
        glacier_state,
        ds_ref,
    )
    
    ax.plot(mad_today_volume[0],
                  '.-',
                  label=('MAD_V_2020'),
                 )
    
    ax.plot(mad_bed_h[0],
                  '.-',
                  label=('MAD_BED'),
                 )
    
    ax.plot(mad_past_volume[0],
                  '.-',
                  label=('MAD_V_1980'),
                 )
    ax.set_ylabel('Scaled by OGGM\nfirst guess MAD')

    # add number of forward model runs
    fig = plt.gcf()
    text_positions = []
    
    t = ax.text(1, 0.01,
                  'Nr. model runs:',
                  va='center',
                  ha='right',
                  c='black',
                transform=ax.get_xaxis_transform(),
                zorder=4,
                #transform=ax.transAxes,
             bbox=dict(
                    boxstyle="round,pad=0.2",  # Rounded edges with padding
                    edgecolor="black",         # Border color
                    facecolor="white",         # Background color
                    alpha=0.8                 # Transparency level
                )
                 )
    text_positions.append(t.get_window_extent(fig.canvas.get_renderer()
                                             ).transformed(fig.transFigure.inverted()))
    for call in ds.fct_calls:
        if call.iteration in list(xticks[1:]):
            t = ax.text(call.iteration,
                          0.01,
                          str(call.item()),
                          va='center',
                          ha='center',
                          c='black',
                        transform=ax.get_xaxis_transform(),
                         zorder=4,
                        #transform=ax.transAxes,
             bbox=dict(
                    boxstyle="round,pad=0.2",  # Rounded edges with padding
                    edgecolor="black",         # Border color
                    facecolor="white",         # Background color
                    alpha=0.8                 # Transparency level
                )
                         )
            text_positions.append(t.get_window_extent(fig.canvas.get_renderer()
                                             ).transformed(fig.transFigure.inverted()))

    x0 = min(b.x0 for b in text_positions)
    y0 = min(b.y0 for b in text_positions)
    x1 = max(b.x1 for b in text_positions)
    y1 = max(b.y1 for b in text_positions)

    #box = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
    #                     transform=fig.transFigure,  # Use figure coordinates
    #                     boxstyle="round,pad=0.4",
    #                     edgecolor="black",
    #                     facecolor="white",
    #                     alpha=0.8,
    #                     linewidth=1,
    #                     zorder=3,
    #                    )
    #ax.add_patch(box)


# # plot for all settings for one iteration

def get_custom_diverging_cmap(color1, color2, name='custom_diverging'):
    return LinearSegmentedColormap.from_list(
        "custom_diverging",
        [color1, "white", color2])


def plot_stat_metrics(ax, main_results_dir,
                      glacier, glacier_state,
                      experiment_options,
                      iteration,
                      fg_method='fg_oggm',
                      title=None, lw_grid=1):

    (mad_today_volume_matrix, mad_bed_h_matrix,
     lambda_values, observation_values) = create_stat_matrices(
         main_results_dir, glacier,
         glacier_state, iteration,
         experiment_options,
         fg_method)
    
    stat1_matrix = mad_bed_h_matrix
    stat1_label = 'MAD_BED scaled by OGGM first guess'
    stat2_matrix = mad_today_volume_matrix
    stat2_label = 'MAD_V_2020 scaled by OGGM first guess'

    
    
    # Define the color maps for the two statistics
    colorblind_palette = sns.color_palette("colorblind")
    cmap1 = get_custom_diverging_cmap(
        color1=colorblind_palette[0],
        color2=colorblind_palette[3],
        name='custom_diverging_1'
    )  # Divergent colormap for statistic 1
    cmap2 = get_custom_diverging_cmap(
        color1=colorblind_palette[2],
        color2=colorblind_palette[4],
        name='custom_diverging_2'
    )# Different divergent colormap for statistic 2
    
    # Normalize the values for color mapping
    def get_vmin_vmax(matrix):
        if np.max(matrix) < 1:
            vmax = 1.1 #np.abs(np.min(matrix))
        else:
            vmax = np.max(matrix)

        vmax = 2

        if np.min(matrix) > 1:
            vmin = np.abs(np.max(matrix)) * (-1)
        else:
            vmin = np.min(matrix)

        vmin = 0

        return vmax, vmin

    vmax1, vmin1 = get_vmin_vmax(stat1_matrix)
    vmax2, vmin2 = get_vmin_vmax(stat2_matrix)
        
    norm1 = mcolors.TwoSlopeNorm(vmin=vmin1, vcenter=1, vmax=vmax1)
    norm2 = mcolors.TwoSlopeNorm(vmin=vmin2, vcenter=1, vmax=vmax2)
    
    # Draw each cell in the matrix
    for i in range(stat1_matrix.shape[0]):
        for j in range(stat1_matrix.shape[1]):
            # Get the values for each statistic
            stat1_value = stat1_matrix[i, j]
            stat2_value = stat2_matrix[i, j]
            
            # Calculate the position of each cell
            x = j
            y = len(lambda_values) - i - 1
            
            # Draw the left half for stat1
            rect1 = Rectangle((x, y), 0.5, 1, color=cmap1(norm1(stat1_value)))
            ax.add_patch(rect1)
            
            # Draw the right half for stat2
            rect2 = Rectangle((x + 0.5, y), 0.5, 1, color=cmap2(norm2(stat2_value)))
            ax.add_patch(rect2)

            # Add the text for stat1 (left side)
            ax.text(x + 0.25, y + 0.5, f'{stat1_value:.1f}',
                    va='center', ha='center', fontsize=8, color='black')
            
            # Add the text for stat2 (right side)
            ax.text(x + 0.75, y + 0.5, f'{stat2_value:.1f}',
                    va='center', ha='center', fontsize=8, color='black')
    
    # Set the axis labels and ticks
    ax.set_xticks(np.arange(len(observation_values)) + 0.5)
    ax.set_yticks(np.arange(len(lambda_values)) + 0.5)
    
    ax.set_xticklabels(
        get_multiple_short_labels_obs(experiment_options['observations']),
        rotation=0
    )
    ax.set_yticklabels([f"{experiment_options['cost_lambda'][lam]:.1e}".replace('e-0', 'e-').replace('e+0', 'e+')
                        for lam in reversed(lambda_values)])
    
    # Set labels
    ax.set_xlabel('Observations')
    ax.set_ylabel(r'$\lambda$')
    ax.set_title(title)
    
    # Set the aspect ratio to ensure squares have the same size
    #ax.set_aspect('equal')
    
    # Draw vertical and horizontal lines between cells
    for x in range(len(observation_values) + 1):
        ax.vlines(x, ymin=0, ymax=len(lambda_values), color='black', linewidth=lw_grid)
    for y in range(len(lambda_values) + 1):
        ax.hlines(y, xmin=0, xmax=len(observation_values), color='black', linewidth=lw_grid)
    
    #fig.subplots_adjust(bottom=0.25)
    
    # Add colorbars for both statistics
    sm1 = plt.cm.ScalarMappable(cmap=cmap1, norm=norm1)
    sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)

    cbar1 = plt.colorbar(sm1, ax=ax,
                         orientation='horizontal',
                         location='top',
                         #fraction=0.02,
                         #shrink=1.5,
                         aspect=18,
                         pad=-0.12)
    cbar1_ticks = [0, 1, 2]
    cbar1.set_ticks(cbar1_ticks)
    cbar1.set_ticklabels(['<1 (better)',
                          '1 = OGGM first guess',
                          '>1 (worse)'])
    
    cbar2 = plt.colorbar(sm2, ax=ax,
                         orientation='horizontal',
                         location='top',
                         #fraction=0.02,
                         aspect=18,
                         pad=0.04)
    cbar2_ticks = [0, 1, 2]
    cbar2.set_ticks([]) #cbar2_ticks)
    cbar2.set_ticklabels([]) #['<1 (better than fg)', '1 (first guess)', '>1 (worse than fg)'])

    cbar1.ax.text(0.5, 0.45, stat1_label,
                  va='center', ha='center',
                  fontsize=10, color='black',
                  transform=cbar1.ax.transAxes)
    cbar2.ax.text(0.5, 0.45, stat2_label,
                  va='center', ha='center',
                  fontsize=10, color='black',
                  transform=cbar2.ax.transAxes)
    
    # Adjust ylim to match the number of lambda values
    ax.set_ylim(0, len(lambda_values))
    ax.set_xlim(0, len(observation_values))

# # concise plot for all settings of one iteration



