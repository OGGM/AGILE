import torch
import os
import shutil
import numpy as np
import logging
import salem
import rasterio
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from cobbi.core import data_logging
from cobbi.core.data_logging import DataLogger
from cobbi.core.arithmetics import RMSE, mean_BIAS
from cobbi.core.utils import NonRGIGlacierDirectory
from cobbi.core.cost_function import create_cost_func
from cobbi.core.data_logging import write_pickle, load_pickle
# -------------------------------
# Further initialization / extended import tasks
# Module logger
log = logging.getLogger(__name__)


class InversionDirectory(object):

    def __init__(self, gdir: NonRGIGlacierDirectory):
        self.gdir = gdir
        self.inv_settings = gdir.inversion_settings
        self.true_bed = None
        self.first_guessed_bed = None
        self.surf_noise = None
        self.ref_surf = None
        self.ice_mask = None
        self.minimize_log = ''
        self.cost_func = None
        self.data_logger = None
        if not 'minimize_bounds' in self.inv_settings:
            self.inv_settings['minimize_bounds'] = None

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
            Bed Bias: {bed_bias:g}
            Bed Max_diff: {bed_maxdiff:g}
            Surface RMSE: {surf_rmse:g}
            Surface Max_diff: {surf_maxdiff:g}
            '''
            myargs = {
                'func_call': i,
                'iteration': len(dl.step_indices),
                'cost': dl.costs[i],
                'bed_rmse': RMSE(dl.beds[i], b),
                'bed_bias': mean_BIAS(dl.beds[i], b, np.sum(self.ice_mask)),
                'bed_maxdiff': np.max(np.abs(dl.beds[i] - b)),
                'surf_rmse': RMSE(dl.surfs[i], self.ref_surf),
                'surf_maxdiff': np.max(np.abs(dl.surfs[i] - self.ref_surf))
            }

            if self.surf_noise is not None:
                log_entry += 'RMSE to perturbed surf: {:g}\n'.format(
                    RMSE(dl.surfs[i], self.ref_surf + self.surf_noise))
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
                            self.inv_settings['inversion_subdir'])

    def clear_dir(self, dir):
        if os.path.exists(dir):
            for f in os.listdir(dir):
                if not str.endswith(f, '.py'):
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
            self.get_subdir_filepath('first_guessed_bed')).get_vardata()
        self.ice_mask = np.load(self.gdir.get_filepath('ref_ice_mask'))
        if os.path.exists(self.gdir.get_filepath('dem_noise')):
            shutil.copy(self.gdir.get_filepath('dem_noise'),
                        self.get_subdir_filepath('dem_noise'))
            self.surf_noise = np.load(self.get_subdir_filepath('dem_noise'))
        else:
            self.surf_noise = None

    def get_subdir_filepath(self, filename, filesuffix=None):
        """
        Gets the filepath for a file with a given name (without extension).
        Works as and is based on get_filepath in GlacierDirectory,
        but returns filepath in this inversion directory.

        Parameters
        ----------
        filename: str
            name of the file
        filesuffix: str
            optional filesuffix to the filename

        Returns
        -------
        Entire path to this file in this inversion directory

        """
        original_path = self.gdir.get_filepath(filename, filesuffix=filesuffix)
        original_path = os.path.split(original_path)
        return os.path.join(self.get_current_basedir(), original_path[1])

    def get_bounds(self):
        """
        Creates bounds for the minimization on the current domain. If
        'bounds_min_max' in inversion settings is None, no bounds are set.
        Else, in Areas without ice, upper and lower bound are exactly as the
        observed surface, otherwise min and max values for ice thickness are
        taken from 'bounds_min_max' and give bounds in glacierized areas. (
        min ice thickness determines upper bound and max ice thickness
        determines lower bound)

        Returns
        -------
        bounds for this domain and this inversion settings
        """
        bounds = None
        if self.inv_settings['bounds_min_max'] is not None:
            min_ice_thickness = self.inv_settings['bounds_min_max'][0]
            max_ice_thickness = self.inv_settings['bounds_min_max'][1]
            lower_bounds = self.ref_surf - max_ice_thickness * self.ice_mask
            upper_bounds = self.ref_surf - min_ice_thickness * self.ice_mask
            bounds = np.c_[lower_bounds.flatten(), upper_bounds.flatten()]
        return bounds

    def run_minimize(self):
        """
        Here the actual minimization of the cost_function is done via
        scipy.optimize.minimize.
        First, data from the glacier directory is read and optionally a
        DataLogger is created. The inversion settings used for this
        particular inversion are saved in this subdirectory. Bounds for the
        minimization are derived. Then the cost function is created and the
        minimization of this cost function started. In the end, the result is
        written to disk and optionally, further information is written to disk.

        The whole process is dominated by the set inversion settings

        Returns
        -------
        Result of minimization as scipy.optimize.minimize returns (res.x
        gives flattened ndarray with bed, needs to be reshaped)

        """

        # Copy first_guessed_bed to inversion directory
        if self.inv_settings['log_minimize_steps']:
            # TODO: really useful? -> respect reset argument in gdir?
            self.clear_dir(self.get_current_basedir())

        with rasterio.open(self.gdir.get_filepath('first_guessed_bed')) as src:
            profile = src.profile
            data = src.read(1)
        with rasterio.open(self.get_subdir_filepath('first_guessed_bed'),
                           'w', **profile) as dst:
            dst.write(data, 1)
        if os.path.exists(self.gdir.get_filepath('first_guessed_bed_noise')):
            shutil.copy(self.gdir.get_filepath('first_guessed_bed_noise'),
                        self.get_subdir_filepath('first_guessed_bed_noise'))

        write_pickle(self.inv_settings,
                     self.get_subdir_filepath('inversion_settings'))
        # Write out reg_parameters to check easier later on
        self.write_string_to_file(self.get_subdir_filepath('reg_parameters'),
                                  str(self.inv_settings['reg_parameters']))
        self.inv_settings = load_pickle(
            self.get_subdir_filepath('inversion_settings'))
        self._read_all_data()
        self.minimize_log = ''
        self.data_logger = None
        callback = None

        if self.inv_settings['log_minimize_steps']:
            dl = DataLogger(self)
            self.data_logger = dl
            callback = self.iteration_info_callback

        # ----------------------------------------------------------------------
        # Core: things are happening here:
        bounds = self.get_bounds()

        self.cost_func = create_cost_func(self.gdir, self.data_logger,
                                          self.surf_noise)
        res = minimize(fun=self.cost_func,
                       x0=self.first_guessed_bed.astype(np.float64).flatten(),
                       method=self.inv_settings['solver'], jac=True,
                       bounds=bounds,
                       options=self.inv_settings['minimize_options'],
                       callback=callback)

        inverted_bed = res.x.reshape(self.first_guessed_bed.shape)
        # ----------------------------------------------------------------------

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