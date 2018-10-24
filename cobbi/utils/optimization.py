import os
from cobbi.utils import test_cases
from cobbi.inversion import *
from cobbi.utils.data_logging import DataLogger
from cobbi.utils import data_logging
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import interpolation

class LCurveTest(object):

    def __init__(self, case, y0, y_spinup_end, y_end, minimize_options=None,
                 solver='L-BFGS-B'):
        self.basedir = '/data/philipp/tests/lcurve/'

        self.case = case
        self.y0 = y0
        self.y_spinup_end = y_spinup_end
        self.y_end = y_end

        su = spin_up(self.case, self.y_spinup_end, self.y_end)
        self.start_surf = su[0].detach().numpy()
        self.exact_surf = su[1].detach().numpy()
        self.reference_surf = self.exact_surf
        self.surface_noise = None
        self.ice_mask = su[2].detach().numpy()
        self.mb = su[3]
        self.bed_2d = su[4].detach().numpy()

        self.lambdas = torch.zeros(9, dtype=torch.float, requires_grad=False)
        self.first_guess = get_first_guess(self.reference_surf, self.ice_mask,
                                           self.case.dx)
        
        self.cost_func = None
        self.optim_log = ''
        self.optimization_counter = 0

        self.solver = solver
        self.maxiter = 100
        self.minimize_options = minimize_options
        if minimize_options is None:
            self.minimize_options = {'maxiter': self.maxiter,
                                     # 'ftol': 1e-4,
                                     #  'xtol': 1e-8,
                                     'maxcor': 20,
                                     'disp': True}

        self.data_logger = DataLogger(self)
        self.optimization_bounds = None

    def iteration_info_callback(self, x0):
        i = len(self.data_logger.costs) - 1
        if i >= 0:
            dl = self.data_logger
            dl.step_indices.append(i)
            b = self.bed_2d
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
                  RMSE(dl.surfs[i], self.reference_surf))
            if self.surface_noise is not None:
                log_entry += 'Surface RMSE (from exact): {:g}\n'.format(
                    RMSE(dl.surfs[i], self.exact_surf))
            log_entry += 'Surface Max_diff: {:g}\n'.format(
                  np.max(np.abs(dl.surfs[i]
                                - self.reference_surf)))
            print(log_entry)
            self.optim_log += log_entry

    def run_minimize2(self, update_scaling=0.5, write_specs=True):
        self.optim_log = ''
        if write_specs:
            self.clear_dir(self.get_current_basedir())
            self.write_string_to_file('settings.txt',
                                      self.get_setting_as_string())
        dl = DataLogger(self)
        self.data_logger = dl

        self.cost_func = create_cost_function(self.start_surf,
                                              self.reference_surf,
                                              self.ice_mask, self.case.dx,
                                              self.mb, self.y_spinup_end,
                                              self.y_end,
                                              self.lambdas,
                                              dl)
        guessed_bed = self.first_guess.copy()
        for i in range(self.minimize_options['maxiter']):
            cost, grad = self.cost_func(guessed_bed)
            self.iteration_info_callback(guessed_bed)
            if cost < 10:
                break
            grad = grad.reshape(guessed_bed.shape)
            gradmax_index = np.unravel_index(np.argmax(np.abs(grad)), grad.shape)
            surf_diff = np.abs(dl.surfs[-1][gradmax_index]
                                    - self.reference_surf[gradmax_index])
            k = surf_diff / np.abs(grad[gradmax_index])
            guessed_bed = guessed_bed - update_scaling * k * grad


        #res = minimize(fun=self.cost_func,
        #               x0=self.first_guess.astype(np.float64).flatten(),
        #               method=self.solver, jac=True,
        #               options=self.minimize_options,
        #               callback=self.iteration_info_callback)

        if write_specs:
            self.write_string_to_file('log.txt', self.optim_log)
            dir = self.get_current_basedir()
            dl.filter_data_from_optimization()
            data_logging.write_pickle(dl, dir + 'data_logger.pkl')
            dl.plot_all(dir)
            plt.close('all')
        del self.cost_func
        self.cost_func = None
        self.optimization_counter += 1

    def run_minimize(self, write_specs=True):
        self.optim_log = ''
        if write_specs:
            self.clear_dir(self.get_current_basedir())
            self.write_string_to_file('settings.txt',
                                      self.get_setting_as_string())
        dl = DataLogger(self)
        self.data_logger = dl
        
        self.cost_func = create_cost_function(self.start_surf,
                                              self.reference_surf,
                                              self.ice_mask, self.case.dx,
                                              self.mb, self.y_spinup_end,
                                              self.y_end,
                                              self.lambdas,
                                              dl)

        res = minimize(fun=self.cost_func,
                       x0=self.first_guess.astype(np.float64).flatten(),
                       method=self.solver, jac=True,
                       bounds=self.optimization_bounds,
                       options=self.minimize_options,
                       callback=self.iteration_info_callback)

        if write_specs:
            self.write_string_to_file('log.txt', self.optim_log)
            dir = self.get_current_basedir()
            dl.filter_data_from_optimization()
            data_logging.write_pickle(dl, dir + 'data_logger.pkl')
            dl.plot_all(dir)
            plt.close('all')
        del self.cost_func
        self.cost_func = None
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
        s += '\ncase.smooth_border_px = {:d}'.format(self.case.smooth_border_px)
        s += '\ny0 = {:d}'.format(self.y0)
        s += '\ny_spinup_end = {:d}'.format(self.y_spinup_end)
        s += '\nself.y_end = {:d}'.format(self.y_end)
        s += '\nlambdas = ' + str(self.lambdas)
        s += '\noptimization_counter = {:d}'.format(self.optimization_counter)
        s += '\nsolver = ' + self.solver
        s += '\nminimize_options = ' + str(self.minimize_options)
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

    def remove_surface_noise(self):
        self.reference_surf = self.exact_surf
        self.surface_noise = None

    def add_surface_noise(self, std=3, zoom=-1, glacier_only=True):
        noise = std * np.random.standard_normal(self.exact_surf.shape)
        if zoom > 0:
            noise = interpolation.zoom(noise, zoom)[0:self.exact_surf.shape[0],
                                                    0:self.exact_surf.shape[1]]
        if glacier_only:
            noise = noise * self.ice_mask
        self.surface_noise = noise
        self.reference_surf = self.exact_surf + noise

