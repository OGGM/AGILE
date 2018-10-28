import torch
import os
import shutil
import json
import numpy as np
from oggm import cfg
from oggm.utils import GlacierDirectory, mkdir
from cobbi.core.data_logging import write_pickle, load_pickle  # TODO: move

EXTENDED_BASENAMES = {
    'first_guessed_bed': 'first_guessed_bed.tiff',
    'inversion_settings': 'inversion_settings.pkl',
    'ice_mask': 'ice_mask.npy',
    'ice_thickness': 'ice_thickness.npy',
    'inverted_bed': 'inverted_bed.tiff',
    'ref_dem': 'ref_dem.tiff',
    'ref_ice_mask': 'ref_ice_mask.npy',
    'ref_ice_thickness': 'ref_ice_thickness.npy',
    'spinup_dem': 'spinup_dem.tiff',
    'spinup_ice_mask': 'spinup_ice_mask.npy',
    'spinup_ice_thickness': 'spinup_ice_thickness.npy',
    'data_logger': 'data_logger.pkl'
}

DEFAULT_REG_PARAMETERS = np.array([0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

class NonRGIGlacierDirectory(GlacierDirectory):
    """GlacierDirectory for synthetic cases and glaciers, which are not
    registered with RGI"""

    def __init__(self, case, base_dir=None, reset=False):
        """
        Initializes the glacier directory

        Parameters
        ----------
        case : TestCase
            TestCase providing name, geographical information, information
            for mb, ...
        base_dir : str
            path to the directory where to open the directory.
            Defaults to `cfg.PATHS['working_dir'] + /per_glacier/`
        reset : bool, default=False
            empties the directory at construction (careful!)
        """

        self.case = case

        if base_dir is None:
            if not cfg.PATHS.get('working_dir', None):
                raise ValueError("Need a valid PATHS['working_dir']!")
            base_dir = os.path.join(cfg.PATHS['working_dir'], case.name)

        # entity = {'min_x': case.extent[0, 0],
        #          'max_x': case.extent[1, 0],
        #          'min_y': case.extent[0, 1],
        #          'max_y': case.extent[1, 1],
        #          'name': case.name}

        self.extent_ll = case.extent.T
        # [[entity['min_x'], entity['max_x']],
        #                  [entity['min_y'], entity['max_y']]]

        # Could choose different dummy RGI as well
        self.rgi_id = 'RGI00-00.00000'
        self.glims_id = '00'
        #self.rgi_area_km2 = 0.rgi_area_km2
        self.cenlon = (case.extent[1, 0] + case.extent[0, 0]) / 2.
        self.cenlat = (case.extent[1, 1] + case.extent[0, 1]) / 2.
        self.rgi_region = '00'
        self.rgi_subregion = '00-00'
        self.rgi_version = '00'

        self.name = case.name
        self.glacier_type = 'Ice Cap'
        self.terminus_type = 'Land-terminating'

        self.is_tidewater = self.terminus_type in ['Marine-terminating',
                                                   'Lake-terminating']
        self.inversion_calving_rate = 0.
        self.is_icecap = self.glacier_type == 'Ice cap'

        # Hemisphere
        self.hemisphere = 'sh' if self.cenlat < 0 else 'nh'

        # The divides dirs are created by gis.define_glacier_region, but we
        # make the root dir
        self.dir = os.path.join(base_dir, self.name)
        if reset and os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        mkdir(self.dir)

        # logging file
        self.logfile = os.path.join(self.dir, 'log.txt')

        # Optimization
        #self._mbdf = None

    def __repr__(self):

        summary = ['<oggm.NonRGIGlacierDirectory>']
        summary += ['  RGI id: ' + self.rgi_id]
        summary += ['  Region: ' + self.rgi_region_name]
        summary += ['  Subregion: ' + self.rgi_subregion_name]
        if self.name :
            summary += ['  Name: ' + self.name]
        summary += ['  Glacier type: ' + str(self.glacier_type)]
        summary += ['  Terminus type: ' + str(self.terminus_type)]
        summary += ['  Area: ' + str(self.rgi_area_km2) + ' km2']
        summary += ['  Lon, Lat: (' + str(self.cenlon) + ', ' +
                    str(self.cenlat) + ')']
        if os.path.isfile(self.get_filepath('glacier_grid')):
            summary += ['  Grid (nx, ny): (' + str(self.grid.nx) + ', ' +
                        str(self.grid.ny) + ')']
            summary += ['  Grid (dx, dy): (' + str(self.grid.dx) + ', ' +
                        str(self.grid.dy) + ')']
        return '\n'.join(summary) + '\n'

    def get_filepath(self, filename, delete=False, filesuffix=''):
        """Absolute path to a specific file.

        Parameters
        ----------
        filename : str
            file name (must be listed in cfg.BASENAME)
        delete : bool
            delete the file if exists
        filesuffix : str
            append a suffix to the filename (useful for model runs). Note
            that the BASENAME remains same.

        Returns
        -------
        The absolute path to the desired file
        """

        if filename not in EXTENDED_BASENAMES and \
                filename not in cfg.BASENAMES:
            raise ValueError(filename + ' not in known BASENAMES.')

        if filename in EXTENDED_BASENAMES:
            fname = EXTENDED_BASENAMES[filename]
        else:
            fname = cfg.BASENAMES[filename]
        if filesuffix:
            fname = fname.split('.')
            assert len(fname) == 2
            fname = fname[0] + filesuffix + '.' + fname[1]
        out = os.path.join(self.dir, fname)
        if delete and os.path.isfile(out):
            os.remove(out)
        return out

    def copy_to_basedir(self, base_dir, setup='run'):
        """

        Parameters
        ----------
        base_dir
        setup

        Returns
        -------

        """
        raise NotImplementedError

    def get_ref_mb_data(self):
        """

        Returns
        -------

        """
        raise NotImplementedError

    def get_ref_length_data(self):
        """

        Returns
        -------

        """
        raise NotImplementedError

    def get_inversion_flowline_hw(self):
        """

        Returns
        -------

        """
        raise NotImplementedError

    @property
    def inversion_settings(self):
        """Dictionary with settings for the inversion"""
        return load_pickle(self.get_filepath('inversion_settings'))

    def write_inversion_settings(self, mb_spinup=None,
                                 mb_forward_run=None, yrs_spinup=2000,
                                 yrs_forward_run=500,
                                 fg_slope_cutoff_angle=5.0,
                                 fg_shape_factor=1.0,
                                 reg_parameters=DEFAULT_REG_PARAMETERS,
                                 solver='L-BFGS-B', minimize_options=None,
                                 inversion_counter=0, log_minimize_steps=True):
        """
        TODO

        Parameters
        ----------
        mb_spinup
        mb_forward_run
        yrs_spinup
        yrs_forward_run
        reg_parameters
        solver
        minimize_options
        inversion_counter
        log_minimize_steps

        Returns
        -------
        Nothing
        """

        inv_settings = locals()
        inv_settings.pop('self')
        inv_settings['case'] = self.case

        if type(reg_parameters) is np.ndarray:
            inv_settings['reg_parameters'] = reg_parameters.tolist()
        elif type(reg_parameters) is torch.Tensor:
            inv_settings['reg_parameters'] = \
                reg_parameters.detach().numpy.tolist()
        write_pickle(inv_settings, self.get_filepath('inversion_settings'))