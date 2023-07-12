import numpy as np
import matplotlib.pyplot as plt
import salem

from agile2d.core import gis, test_cases
from agile2d.core.utils import NonRGIGlacierDirectory
from agile2d.core.inversion import InversionDirectory
from agile2d.core.data_logging import load_pickle

from oggm import cfg
cfg.initialize()

basedir = '/data/philipp/thesis_test/Giluwe/perfect'

case = test_cases.Giluwe
gdir = NonRGIGlacierDirectory(case, basedir)
idir = InversionDirectory(gdir)
idir.inv_settings['inversion_counter'] = 1001

true_bed = salem.GeoTiff(gdir.get_filepath('dem')).get_vardata()

data_logger = load_pickle(idir.get_subdir_filepath('data_logger'))

plt.figure()
plt.imshow(data_logger.beds[-1] - true_bed)

subsampled_beds = data_logger.beds[-3:]

for i in [3, 4, 5, 6, 7, 8, 9, 10]:
    yfrom = 6
    yend = 8
    y = i
    xfrom = 0
    xend = -1

    plt.figure()
    for bed in subsampled_beds:
        plt.plot(bed[y, xfrom:xend])

    plt.plot(true_bed[y, xfrom:xend], 'k')
    plt.title('Y-Crossection at {:d}'.format(i))

print('end')