import os
from cobbi.utils import test_cases
from cobbi.inversion import *
from cobbi.utils.data_logging import DataLogger
from cobbi.utils import data_logging
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

class LCurveTest(object):

    def __init__(self, case, y0, y_spinup_end, y_end):

        self.basedir = '/data/philipp/tests/lcurve/'

        self.case = case
        self.y0 = y0
        self.y_spinup_end = y_spinup_end
        self.y_end = y_end

        su = spin_up(self.case, self.y_spinup_end, self.y_end)
        self.start_surf = su[0]
        self.reference_surf = su[1]
        self.ice_mask = su[2]
        self.mb = su[3]
        self.bed_2d = su[4]

        self.lambdas = torch.zeros(9, dtype=torch.float)
        self.first_guess = get_first_guess(self.reference_surf, self.ice_mask,
                                           self.case.dx)
        
        self.cost_func = None
        self.optim_log = ''
        self.optimization_counter = 0
        self.data_logger = DataLogger(self.bed_2d.detach().numpy(),
                                      self.reference_surf.detach().numpy(),
                                      self.first_guess.detach().numpy())
        self.maxiter = 100

    def iteration_info_callback(self, x0):
        i = len(self.data_logger.costs) - 1
        if i >= 0:
            dl = self.data_logger
            dl.step_indices.append(i)
            b = self.bed_2d.detach().numpy()
            log_entry = ''
            log_entry += '----------------------------------------------\n'
            log_entry += 'Function Call: {:d}\n'.format(i)
            log_entry += 'Iteration: {:d}\n'.format(
                len(dl.step_indices))
            log_entry += 'Cost: {:g}\n'.format(dl.costs[i])
            log_entry += 'Bed RMSE: {:g}\n'.format(RMSE(dl.beds[i], b))
            log_entry += 'Bed Max_diff: {:g}\n'.format(
                np.max(np.abs(dl.beds[i] - b)))
            log_entry += 'Surface RMSE: {:g}\n'.format(
                  RMSE(dl.surfs[i], self.reference_surf.detach().numpy()))
            log_entry += 'Surface Max_diff: {:g}\n'.format(
                  np.max(np.abs(dl.surfs[i]
                                - self.reference_surf.detach().numpy())))
            print(log_entry)
            self.optim_log += log_entry

    def run_minimize(self, write_specs=True):
        self.optim_log = ''
        if write_specs:
            self.clear_dir(self.get_current_basedir())
            self.write_string_to_file('settings.txt',
                                      self.get_setting_as_string())
        dl = DataLogger(self.bed_2d.detach().numpy(),
                        self.reference_surf.detach().numpy(),
                        self.first_guess.detach().numpy())
        self.data_logger  = dl
        
        self.cost_func = create_cost_function(self.start_surf,
                                              self.reference_surf,
                                              self.ice_mask, self.case.dx,
                                              self.mb, self.y_spinup_end,
                                              self.y_end,
                                              self.lambdas,
                                              dl)

        res = minimize(fun=self.cost_func,
                       x0=self.first_guess.detach().numpy().astype(np.float64).flatten(),
                       method='L-BFGS-B', jac=True,
                       options={'maxiter': self.maxiter,
                                # 'ftol': 1e-4,
                                # 'xtol': 1e-8,
                                'maxcor': 20,
                                'disp': True},
                       callback=self.iteration_info_callback)

        if write_specs:
            self.write_string_to_file('log.txt', self.optim_log)
            dir = self.get_current_basedir()
            dl.filter_data_from_optimization()
            data_logging.write_pickle(dl, dir + 'data_logger.pkl')
            dl.plot_all(dir)
            plt.close('all')
        self.optimization_counter += 1

    def write_string_to_file(self, filename, text):
        dir = self.get_current_basedir()
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        with open(dir + filename, 'w') as f:
            f.write(text)

    def get_setting_as_string(self):
        s = ''
        s += 'basedir = \'{:s}\''.format(self.basedir)
        s += '\ncase = ' + self.case.name
        s += '\ncase.dx = {:d}'.format(self.case.dx)
        s += '\ny0 = {:d}'.format(self.y0)
        s += '\ny_spinup_end = {:d}'.format(self.y_spinup_end)
        s += '\nself.y_end = {:d}'.format(self.y_end)
        s += '\nlambdas = ' + str(self.lambdas)
        s += '\noptimization_counter = {:d}'.format(self.optimization_counter)
        return s

    def get_current_basedir(self):
        return self.basedir + '{:d}/'.format(self.optimization_counter)

    def clear_dir(self, dir):
        if os.path.exists(dir):
            for f in os.listdir(dir):
                os.remove(os.path.join(dir, f))
        else:
            if not os.path.exists(dir):
                os.makedirs(dir, exist_ok=True)
