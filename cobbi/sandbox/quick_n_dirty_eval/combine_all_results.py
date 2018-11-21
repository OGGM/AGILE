import numpy as np
import glob
import os
from cobbi.sandbox.quick_n_dirty_eval import experiment_naming_engine

basedir = '/data/philipp/erstabgabe/'
filepaths = glob.glob(os.path.join(basedir, '*/*/results.csv'))

results = ''
with open(filepaths[0], 'r') as f:
    results += f.readline()
filepaths = sorted(filepaths)

for path in filepaths:
    with open(path, 'r') as f:
        f.readline()
        line = f.readline()
        cols = line.split(',')
        #cols[1] = experiment_naming_engine.get_experiment_name(cols[1])
        #if cols[1] is not None:
        #    line = ','.join(cols)
        results += '\n' + line

results = results.replace('Borden Peninsula', 'Borden')
#results = results.replace('$', '')
results = results.replace(' star ', '*')

print(results)