import torch
from cobbi.utils import test_cases
from cobbi.utils.optimization import *
import matplotlib.pyplot as plt
import numpy as np

import os
import psutil
process = psutil.Process(os.getpid())
print(process.memory_info().rss / 1024. / 1024.)

case = test_cases.Blafell
case.dx = 800
case.smooth_border_px = 1

y0 = 0
y_spinup_end = 1000
y_end = 1200

process = psutil.Process(os.getpid())
print(process.memory_info().rss / 1024. / 1024.)

lambs = torch.tensor([0.2, 0.2, 1.5, 2., 2., 2., 0., 10., 0.])

test = LCurveTest(case, y0, y_spinup_end, y_end)
test.lambdas = lambs

#test.maxiter = 20
#test.run_minimize()

test.optim_log = ''

dl = DataLogger(test)
test.data_logger = dl


process = psutil.Process(os.getpid())
print(process.memory_info().rss / 1024. / 1024.)

test.cost_func = create_cost_function(test.start_surf,
                                      test.reference_surf,
                                      test.ice_mask, test.case.dx,
                                      test.mb, test.y_spinup_end,
                                      test.y_end,
                                      test.lambdas,
                                      dl)

process = psutil.Process(os.getpid())
print(process.memory_info().rss / 1024. / 1024.)

#res = minimize(fun=self.cost_func,
#               x0=self.first_guess.astype(np.float64).flatten(),
#               method=self.solver, jac=True,
#               options=self.minimize_options,
#               callback=self.iteration_info_callback)

c1, g1 = test.cost_func(test.first_guess)

process = psutil.Process(os.getpid())
print(process.memory_info().rss / 1024. / 1024.)

c2, g2 = test.cost_func(test.first_guess)

process = psutil.Process(os.getpid())
print(process.memory_info().rss / 1024. / 1024.)
test.cost_func = None

process = psutil.Process(os.getpid())
print(process.memory_info().rss / 1024. / 1024.)
tensors = []
import gc
import inspect
oo = gc.get_objects()
with open('/home/philipp/log.txt', 'w') as f:
    for o in oo:
        if getattr(o, "__class__", None):
            txt = str(o.__class__)
            if 'torch.Tensor' in txt: # or 'torch.Tensor' in txt:
                filename = inspect.getabsfile(o.__class__)
                f.write("Object :" + str(o) + "...")
                f.write("Class  :" + txt + "...")
                f.write("defined:" + filename + "\n")
                f.write("shape:" + str(o.shape) + "\n")
                try:
                    plt.figure()
                    plt.imshow(o.detach().numpy())
                    plt.show()
                except:
                    pass
                tensors.append(o)
del oo
del o
                #f.write(str(o) + '\n')



print('end')