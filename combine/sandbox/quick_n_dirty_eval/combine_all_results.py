import glob
import os
from combine.sandbox.quick_n_dirty_eval import experiment_naming_engine

basedir = '/home/philipp/HR_01/'
filepaths = glob.glob(os.path.join(basedir, '*/*/results.csv'))

results = ''
with open(filepaths[0], 'r') as f:
    results += f.readline()
results = results[:-1] + ',subgroupindex,rmse' + results[-1]
filepaths = sorted(filepaths)

for path in filepaths:
    print('\'' + os.path.split(os.path.split(path)[0])[1] + '\'')
    with open(path, 'r') as f:
        f.readline()
        line = f.readline()
        cols = line.split(',')
        base, rmse = experiment_naming_engine.get_base_rmse_from_experiment(
            cols[1])
        cols.append(str(base))
        cols.append(str(rmse))
        line = ','.join(cols)
        results += '\n' + line

results = results.replace('Borden Peninsula', 'Borden')
#results = results.replace('$', '')
results = results.replace(' star', '*')

print(results)