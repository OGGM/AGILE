import torch
import matplotlib.pyplot as plt
import numpy
from cobbi.utils.data_logging import *
import os
from oggm import cfg
import re

cfg.initialize()

basedir = '/data/philipp/tests/lcurve/reg_3_LBFGSB_noise_all/'

reg_vals = []
cost_vals = []
bed_rmses = []

for dir in range(24, 30):
    dl_file = os.path.join(basedir, str(dir), 'data_logger.pkl')
    dl = load_pickle(dl_file)
    reg_vals.append(np.log(dl.c_terms[-1][3]))
    cost_vals.append(np.log(dl.c_terms[-1][-1]))
    bed_rmses.append(dl.get_bed_rmses()[-1])

plt.figure()
plt.plot(reg_vals)  # , reg_vals)
plt.show()

plt.figure()
plt.semilogy(bed_rmses)
plt.show()

print('end')
