import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
import os
from cobbi.core.data_logging import load_pickle
from cobbi.core.test_cases import Giluwe, Borden
from cobbi.core.utils import NonRGIGlacierDirectory
from cobbi.core.inversion import InversionDirectory
from cobbi.core.visualization import plot_iterative_behaviour
from cobbi.core.arithmetics import *
from oggm import cfg;

cfg.initialize()
import salem

sns.set_palette(sns.color_palette('Set1'))
# plt.style.use('seaborn-ticks')
# import seaborn as sns
# sns.set_style('ticks')

basedir = '/home/philipp/zweitversuch2'
    #'/media/philipp/Daten/Dokumente/Studium/Master/Masterarbeit/Ergebnisse'

output_dir = '/media/philipp/Daten/Dokumente/Studium/Master/Masterarbeit' \
             '/Thesis/figs/iteration'

experiment = 'identical-twin'
case = Borden

gdir = NonRGIGlacierDirectory(case, basedir)
#plot_iterative_behaviour(gdir, experiment)

spinup_surf = salem.GeoTiff(gdir.get_filepath('spinup_dem')).get_vardata()
reference_surf = salem.GeoTiff(gdir.get_filepath('ref_dem')).get_vardata()
bed_2d = salem.GeoTiff(gdir.get_filepath('dem')).get_vardata()

ref_ice_mask = np.load(gdir.get_filepath('ref_ice_mask'))
ref_it = np.load(gdir.get_filepath('ref_ice_thickness'))
spinup_it = np.load(gdir.get_filepath('spinup_ice_thickness'))
masked_ice_thick_end = np.ma.masked_array(ref_it,
                                          mask=np.logical_not(ref_ice_mask))
# masked_ice_thick_start = np.ma.masked_array(spinup_it,
#                                            mask=np.logical_not(ref_ice_mask))
masked_reference_surf = np.ma.masked_array(reference_surf,
                                           mask=np.logical_not(ref_ice_mask))

dl = load_pickle(os.path.join(gdir.dir, experiment, 'data_logger.pkl'))
inversion_settings = load_pickle(os.path.join(gdir.dir, experiment,
                                              'inversion_settings.pkl'))
reg_parameters = inversion_settings['reg_parameters']

bed_differences = dl.get_bed_differences()
surf_differences = dl.get_surf_differences()
bed_rmses = [RMSE(bd, 0, ref_ice_mask) for bd in bed_differences]
surf_rmses = [RMSE(sd, 0, ref_ice_mask) for sd in surf_differences]
costs = dl.costs
c_terms = np.array(dl.c_terms).T
all_costs = []
cost_labels = []
i = 0

iteration_index = np.arange(1, len(bed_rmses) + 1)
df = pd.DataFrame(index=iteration_index)
# df.set_index('Iteration #')
# df['Surface'] = surf_rmses
# df['Bed'] = bed_rmses
for c, rp in zip(c_terms[:-1], reg_parameters[:-1]):
    if rp > 0:
        df['c' + str(i)] = c
        all_costs.append(c)
        cost_labels.append('$J_{reg, ' + str(i) + '}$')
    i = i + 1

all_costs.append([c_terms[-1], ])
cost_labels.append('Surface misfit')
df['c-1'] = c_terms[-1]


def log_vals_labels(x, pos):
    begin = str(x)[0]
    if begin in '1, 2, 3, 5':
        return '{:.0f}'.format(x)


formatter = FuncFormatter(log_vals_labels)

available_fill_colors = ['sienna', 'olivedrab', 'mediumpurple',
                         'cornflowerblue']

fig, (ax1, ax2) = plt.subplots(figsize=(8, 5), nrows=2)
l1 = ax1.semilogy(iteration_index, surf_rmses, label='Surface RMSE',
                  color='cornflowerblue', linewidth=2)
l2 = ax1.semilogy(iteration_index, bed_rmses, label='Bed RMSE',
                  color='olivedrab', linewidth=2)
ax1.set_ylabel('RMSE (m)')
ax1.set_xlabel('')
ax1.yaxis.set_minor_formatter(formatter)
ax1.yaxis.set_major_formatter(formatter)
ax1.set_xticklabels([])
ax1.xaxis.set_ticks_position('both')
ax1.tick_params(labeltop=False, labelbottom=False)
ax1.tick_params(direction='in', which='both')
# ax1.grid(which='minor')
# ax1.yaxis.set_ticks_position('both')
# ax1.tick_params(labelleft=True, labelright=False)
ax1.set_xlim(iteration_index[0], iteration_index[-1])
ax1.tick_params(axis='y', which='minor')
ax1b = ax1.twinx()
l3 = ax1b.semilogy(iteration_index, costs, label='Cost', color='crimson',
                   linewidth=2, linestyle='dashed')
ax1b.tick_params(direction='in', which='both')
ax1b.set_ylabel('Cost (m$^2$)')
lns = l1 + l2 + l3
labs = [l.get_label() for l in lns]
ax1b.legend(lns, labs, loc='center right')

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(sns.color_palette(flatui))
my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())
# ax2.stackplot(iteration_index[1:], all_costs, labels=cost_labels)
# plt.yscale('log')
df_perc = df.divide(df.sum(axis=1), axis=0) * 100
df_perc.plot.area(ax=ax2, stacked=True, cmap=my_cmap,
                  linewidth=2)  # , colors=)
# df.plot(ax=ax2)
# plt.yscale('log')
ax2.set_ylabel('Contribution to cost (%)')
ax2.set_xlabel('Iteration #')
ax2.legend(labels=cost_labels, frameon=True)  # , loc='upper
# center',
# ncol=len(df.columns))
ax2.xaxis.set_ticks_position('both')
ax2.yaxis.set_ticks_position('both')
ax2.tick_params(labeltop=False, labelbottom=True, labelleft=True,
                labelright=True)
ax2.set_xlim(iteration_index[0], iteration_index[-1])
ax2.set_ylim(0, 100)
ax2.tick_params(direction='in', which='both', zorder=5000)

plt.tight_layout()
plt.show()
