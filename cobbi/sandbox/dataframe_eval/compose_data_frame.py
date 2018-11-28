import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import salem
from cobbi.core.utils import NonRGIGlacierDirectory
from cobbi.core.test_cases import Borden, Giluwe
from cobbi.core.arithmetics import RMSE, mean_BIAS
from cobbi.core.data_logging import load_pickle
from cobbi.sandbox.quick_n_dirty_eval.experiment_naming_engine import \
    get_experiment_group, get_experiment_subgroup
from oggm import cfg;

cfg.initialize()

basedir = '/home/philipp/final3'
outputdir = '/home/philipp/final3'

case = Borden
gdir = NonRGIGlacierDirectory(case, basedir)
ref_ice_mask = np.load(gdir.get_filepath('ref_ice_mask'))
true_bed = salem.GeoTiff(gdir.get_filepath('dem')).get_vardata()
true_surf = salem.GeoTiff(gdir.get_filepath('ref_dem')).get_vardata()
filepaths = glob.glob(os.path.join(gdir.dir, '*', 'data_logger.pkl'))
filepaths = sorted(filepaths)
columns = [
    'experiment',
    'experimentgroup',
    'experimentsubgroup',
    'subgroupindex',
    'optimizedbed',
    'optimizedsurf',
    'optimizedicethick',
    'firstguess',
    'beds',
    'surfs',
    'costs',
    'cterms',
    'optimizedbederror',
    'optimizedsurferror',
    'optimizedbedrmse',
    'optimizedsurfrmse',
    'optimizedbedbias',
    'optimizedsurfbias',
    'firstguessrmse',
    'firstguessbias'
    'surfacenoise',
    'surfacenoisermse',
    'surfacenoisebias',
    'bedmeasurements',
    'bedmeasurementsrmse',
    'bedmeasurementsbias',
    'iterations',
    'R',
    'dV',
    'warning',
    'dir_path'
]
df = pd.DataFrame(columns=columns)
for path in filepaths:
    dl = load_pickle(path)
    inv_subdir = os.path.split(path)[0]
    inv_settings = load_pickle(os.path.join(gdir.dir, inv_subdir,
                                            'inversion_settings.pkl'))
    experiment = inv_settings['inversion_subdir']
    surface_noise = np.zeros(true_bed.shape)
    if os.path.exists(os.path.join(gdir.dir, inv_subdir, 'dem_noise.npy')):
        surface_noise = np.load(os.path.join(gdir.dir, inv_subdir,
                                             'dem_noise.npy'))
    bed_measurements = np.ma.masked_all(true_bed.shape)
    if os.path.exists(os.path.join(gdir.dir, inv_subdir,
                                   'bed_measurements.pkl')):
        bed_measurements = np.load(os.path.join(gdir.dir, inv_subdir,
                                                'bed_measurements.pkl'))
    warning_found = False
    if os.path.exists(os.path.join(gdir.dir, inv_subdir,
                                   'warning.txt')):
        warning_found = True

    if len(dl.step_indices) > 0:
        final_bed = dl.beds[-1]
        final_surf = dl.surfs[-1]
        final_it = dl.surfs[-1] - dl.beds[-1]
        bed_rmse = RMSE(dl.beds[-1], true_bed, ref_ice_mask)
        bed_bias = mean_BIAS(dl.beds[-1], true_bed, ref_ice_mask)
        bed_error = final_bed - true_bed
        surf_rmse = RMSE(dl.surfs[-1], true_surf, ref_ice_mask)
        surf_bias = mean_BIAS(dl.surfs[-1], true_surf, ref_ice_mask)
        surf_error = final_surf - true_surf
        dV = (((dl.surfs[-1] - dl.beds[-1]).sum())
              - (true_surf - true_bed).sum()) / (true_surf - true_bed).sum()
    else:
        final_bed = np.ma.masked_all(true_bed.shape)
        final_surf = np.ma.masked_all(true_bed.shape)
        final_it = np.ma.masked_all(true_bed.shape)
        bed_error = np.ma.masked_all(true_bed.shape)
        bed_rmse = np.nan
        bed_bias = np.nan
        surf_error = np.ma.masked_all(true_bed.shape)
        surf_rmse = np.nan
        surf_bias = np.nan
        dV = np.nan

    new_row = {
        'experiment': experiment,
        'experimentgroup': get_experiment_group(experiment),
        'experimentsubgroup': get_experiment_subgroup(experiment),
        'subgroupindex': '',
        'optimizedbed': final_bed,
        'optimizedsurf': final_surf,
        'optimizedicethick': final_it,
        'firstguess': dl.first_guessed_bed,
        # 'beds': dl.beds,
        # 'surfs': dl.surfs,
        # 'costs': dl.costs,
        # 'cterms': dl.c_terms,
        'optimizedbederror': bed_error,
        'optimizedbedrmse': bed_rmse,
        'optimizedbedbias': bed_bias,
        'optimizedsurferror': surf_error,
        'optimizedsurfrmse': surf_rmse,
        'optimizedsurfbias': surf_rmse,
        'firstguessrmse': RMSE(dl.first_guessed_bed, true_bed, ref_ice_mask),
        'firstguessbias': mean_BIAS(dl.first_guessed_bed, true_bed,
                                    ref_ice_mask),
        'surfacenoise': surface_noise,
        'surfacenoisermse': RMSE(surface_noise, 0, ref_ice_mask),
        'surfacenoisebias': mean_BIAS(surface_noise, 0, ref_ice_mask),
        'bedmeasurements': bed_measurements,
        'bedmeasurementsrmse': RMSE(bed_measurements, 0, ref_ice_mask),
        'bedmeasurementsbias': mean_BIAS(bed_measurements, 0, ref_ice_mask),
        'iterations': len(dl.step_indices),
        'dx': case.dx,
        'dV': dV,
        'warning': warning_found,
        'dir_path': inv_subdir
    }
    if new_row['experimentgroup'] == 'fg rmse':
        new_row['subgroupindex'] = new_row['firstguessrmse']
    elif new_row['experimentgroup'] == 'fg bias':
        new_row['subgroupindex'] = new_row['firstguessbias']
    elif new_row['experimentgroup'] == 'promised land':
        new_row['subgroupindex'] = new_row['surfacenoisermse']
    elif new_row['experimentgroup'] == 'bed measurements':
        new_row['subgroupindex'] = new_row['bedmeasurementsrmse']

    # df_row = pd.DataFrame(new_row)
    df = df.append(new_row, ignore_index=True)

df = df.sort_values(['experimentgroup', 'experimentsubgroup', 'subgroupindex',
                     'experiment'])

df.to_pickle(os.path.join(basedir, '{:s} dataframe.pkl'.format(case.name)))
# store = pd.HDFStore(os.path.join(basedir,
#                                 '{:s} dataframe.h5'.format(case.name)))
# store['df'] = df
# store.close()
cols_to_drop = [
    'optimizedbed',
    'optimizedsurf',
    'optimizedicethick',
    'firstguess',
    'beds',
    'surfs',
    'costs',
    'cterms',
    'optimizedbederror',
    'optimizedsurferror',
    'surfacenoise',
    'bedmeasurements'
]

small_df = df.copy()
small_df.drop(cols_to_drop, inplace=True, axis=1)
small_df = small_df.to_csv(
    os.path.join(basedir, '{:s} dataframe small.csv'.format(case.name)))

print('end')

fig, ax = plt.subplots()
sdf = df.loc[df['experimentsubgroup'] == 'fg bias']
sdf.plot('subgroupindex', 'optimizedbedrmse', ax=ax)
sdf.plot('subgroupindex', 'firstguessrmse', ax=ax)
plt.show()

fig, ax = plt.subplots()
sdf = df.loc[df['experimentgroup'] == 'fg rmse']
for key, grp in sdf.groupby(['experimentsubgroup']):
    grp.plot('subgroupindex', 'optimizedbedrmse', ax=ax, label='optimized '
                                                               + str(key))
    grp.plot('subgroupindex', 'firstguessrmse', ax=ax, label='fg rmse ' + str(
        key))
    grp.plot('subgroupindex', 'firstguessbias', ax=ax, label='fg bias ' + str(
        key))
plt.show()
