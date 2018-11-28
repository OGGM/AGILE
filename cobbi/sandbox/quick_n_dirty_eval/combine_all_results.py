import glob
import os

basedir = '/home/philipp/final4/'
filepaths = glob.glob(os.path.join(basedir, '*/*/results.csv'))

results = ''
with open(filepaths[0], 'r') as f:
    results += f.readline()
filepaths = sorted(filepaths)

for path in filepaths:
    print('\'' + os.path.split(os.path.split(path)[0])[1] + '\'')
    with open(path, 'r') as f:
        f.readline()
        line = f.readline()
        # cols = line.split(',')
        #cols[1] = experiment_naming_engine.get_experiment_name(cols[1])
        #if cols[1] is not None:
        #    line = ','.join(cols)
        results += '\n' + line

results = results.replace('Borden Peninsula', 'Borden')
#results = results.replace('$', '')
results = results.replace(' star', '*')

print(results)