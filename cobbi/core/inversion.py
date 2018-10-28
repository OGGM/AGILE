import torch
import os
import numpy as np
import logging
import salem
import rasterio
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from cobbi.core import data_logging
from cobbi.core.data_logging import DataLogger
from cobbi.core.arithmetics import rmse
from cobbi.core.utils import NonRGIGlacierDirectory
from cobbi.core.cost_function import create_cost_func
from cobbi.core.data_logging import write_pickle
# -------------------------------
# Further initialization / extended import tasks
# Module logger
log = logging.getLogger(__name__)


class InversionDirectory(object):

    def __init__(self, gdir: NonRGIGlacierDirectory):
        self.gdir = gdir
        self.inv_settings = gdir.inversion_settings
        # TODO: copy these inversion settings to subdirectory
        self.true_bed = None
        self.first_guessed_bed = None
        self.ref_surf = None
        self.minimize_log = ''
        self.cost_func = None
        self.data_logger = None

    def iteration_info_callback(self, x0):
        i = len(self.data_logger.costs) - 1
        if i >= 0:
            dl = self.data_logger
            dl.step_indices.append(i)
            b = self.true_bed
            log_entry = '''
            ----------------------------------------------
            Function Call: {func_call:d}
            Iteration: {iteration:d}
            Cost: {cost:g}
            Bed RMSE: {bed_rmse:g}
            Bed Max_diff: {bed_maxdiff:g}
            Surface RMSE: {surf_rmse:g}
            Surface Max_diff: {surf_maxdiff:g}
            '''
            myargs = {
                'func_call': i,
                'iteration': len(dl.step_indices),
                'cost': dl.costs[i],
                'bed_rmse': rmse(dl.beds[i], b),
                'bed_maxdiff': np.max(np.abs(dl.beds[i] - b)),
                'surf_rmse': rmse(dl.surfs[i], self.ref_surf),
                'surf_maxdiff': np.max(np.abs(dl.surfs[i] - self.ref_surf))
            }

            #if self.surface_noise is not None:
            #    log_entry += 'Surface RMSE (from exact): {:g}\n'.format(
            #        rmse(dl.surfs[i], self.exact_surf))
            log_entry = log_entry.format(**myargs)
            print(log_entry)
            self.minimize_log += log_entry

    def write_string_to_file(self, filename, text):
        dir = self.get_current_basedir()
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, filename), 'w') as f:
            f.write(text)

    def get_current_basedir(self):
        return os.path.join(self.gdir.dir,
                            '{:d}/'.format(self.inv_settings[
                                               'inversion_counter']))

    def clear_dir(self, dir):
        if os.path.exists(dir):
            for f in os.listdir(dir):
                os.remove(os.path.join(dir, f))
        else:
            if not os.path.exists(dir):
                os.makedirs(dir, exist_ok=True)

    def _read_all_data(self):
        """
        Reads all necessary information from files in gdir for
        minimization/optimization and logging.
        """
        self.true_bed = salem.GeoTiff(
            self.gdir.get_filepath('dem')).get_vardata()
        self.ref_surf = salem.GeoTiff(
            self.gdir.get_filepath('ref_dem')).get_vardata()
        self.first_guessed_bed = salem.GeoTiff(
            self.gdir.get_filepath('first_guessed_bed')).get_vardata()

    def get_subdir_filepath(self, filename, filesuffix=None):
        original_path = self.gdir.get_filepath(filename, filesuffix=filesuffix)
        original_path = os.path.split(original_path)
        return os.path.join(self.get_current_basedir(), original_path[1])

    def run_minimize(self):
        """
        Here the actual minimization of the cost_function is done via
        scipy.optimize.minimize

        Parameters
        ----------
        cost_func: function
            Cost function to minimize. Accepts a bed array of size of the
            spinup_surface as input and outputs a scalar variable, which is
            the cost for this bed.

        Returns
        -------
        Result of minimization as scipy.optimize.minimize returns

        """
        self._read_all_data()
        self.minimize_log = ''
        self.data_logger = None
        callback = None

        if self.inv_settings['log_minimize_steps']:
            self.clear_dir(self.get_current_basedir())  # TODO: really
            # neccessary?
            dl = DataLogger(self)
            self.data_logger = dl
            callback = self.iteration_info_callback

        write_pickle(self.inv_settings,
                     self.get_subdir_filepath('inversion_settings'))

        self.cost_func = create_cost_func(self.gdir, self.data_logger)
        res = minimize(fun=self.cost_func,
                       x0=self.first_guessed_bed.astype(np.float64).flatten(),
                       method=self.inv_settings['solver'], jac=True,
                       bounds=None,  # TODO: allow bounds
                       options=self.inv_settings['minimize_options'],
                       callback=callback)

        # TODO: save 'inverted_bed' in directory: Attention: save to
        # subdirectory
        inverted_bed = res.x.reshape(self.first_guessed_bed.shape)
        with rasterio.open(self.gdir.get_filepath('dem')) as src:
            profile = src.profile
        profile['dtype'] = 'float64'
        with rasterio.open(self.get_subdir_filepath('inverted_bed'),
                           'w', **profile) as dst:
            dst.write(inverted_bed, 1)

        if self.inv_settings['log_minimize_steps']:
            self.write_string_to_file('log.txt', self.minimize_log)
            dir = self.get_current_basedir()
            dl.filter_data_from_optimization()  # Optional, if we want to

            data_logging.write_pickle(dl,
                                      self.get_subdir_filepath('data_logger'))
            dl.plot_all(dir)
            plt.close('all')

        return res

    def run_minimize2(self, update_scaling=0.5, write_specs=True):
        self.minimize_log = ''
        if write_specs:
            self.clear_dir(self.get_current_basedir())
            self.write_string_to_file('settings.txt',
                                      self.get_setting_as_string())
        dl = DataLogger(self)
        self.data_logger = dl

        self.cost_func = create_cost_func(self.gdir, self.lambdas,
                                          self.yrs_to_run, self.case.dx,
                                          self.mb_model, self.data_logger)

        first_guessed_bed = salem.GeoTiff(self.gdir.get_filepath(
            'first_guessed_bed'))

        for i in range(self.minimize_options['maxiter']):
            cost, grad = self.cost_func(guessed_bed)
            self.iteration_info_callback(guessed_bed)

            if cost < 10:
                break
            grad = grad.reshape(guessed_bed.shape)
            surf_diff = dl.surfs[-1] - self.reference_surf
            # TODO: reference_surf, ice_mask
            locally_to_compensate = surf_diff**2 / self.ice_mask.sum()
            n = locally_to_compensate / grad
            n[np.isinf(n)] = np.nan
            n = np.nanmax(np.abs(n))
            guessed_bed = guessed_bed - update_scaling * n * grad


        #res = minimize(fun=self.cost_func,
        #               x0=self.first_guess.astype(np.float64).flatten(),
        #               method=self.solver, jac=True,
        #               options=self.minimize_options,
        #               callback=self.iteration_info_callback)

        if write_specs:
            self.write_string_to_file('log.txt', self.minimize_log)
            dir = self.get_current_basedir()
            dl.filter_data_from_optimization()
            data_logging.write_pickle(dl, dir + 'data_logger.pkl')
            dl.plot_all(dir)
            plt.close('all')
        del self.cost_func
        self.cost_func = None
        self.optimization_counter += 1