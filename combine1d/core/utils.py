import torch
import os
import pyproj
from functools import partial
from shapely.ops import transform as shp_trafo
import geopandas as gpd
import salem
from salem import wgs84
from salem.gis import transform_proj
import shutil
import json
import numpy as np
from oggm import cfg
from oggm.utils import GlacierDirectory, mkdir
from oggm.utils._funcs import multipolygon_to_polygon
from oggm.exceptions import InvalidParamsError
from combine2d.core.data_logging import write_pickle, load_pickle
# TODO: move pickle methods


class InversionGlacierDirectory(GlacierDirectory):
    """GlacierDirectory for synthetic cases and glaciers, which are not
    registered with RGI"""

    def __init__(self, rgi_entity, base_dir=None, reset=False, from_tar=False, delete_tar=False):
        """
        Initializes the glacier directory for COMBINE

        TODO
        """

        super(InversionGlacierDirectory, self).__init__(rgi_entity, base_dir, reset, from_tar, delete_tar)

        # add aditional BASENAMES used by the Inversion
        cfg.add_to_basenames('inversion_settings', 'inversion_settings.pkl', 'Contains the needed information for the '
                                                                             'COMBINE inversion. TODO: doc needed '
                                                                             'information')


    def _reproject_and_write_shapefile(self, entity, filesuffix=''):

        # Make a local glacier map
        params = dict(name='tmerc', lat_0=0., lon_0=self.cenlon,
                      k=0.9996, x_0=0, y_0=0, datum='WGS84')
        proj4_str = "+proj={name} +lat_0={lat_0} +lon_0={lon_0} +k={k} " \
                    "+x_0={x_0} +y_0={y_0} +datum={datum}".format(**params)

        # Reproject
        proj_in = pyproj.Proj("epsg:4326", preserve_units=True)
        proj_out = pyproj.Proj(proj4_str, preserve_units=True)

        # transform geometry to map
        project = partial(transform_proj, proj_in, proj_out)
        geometry = shp_trafo(project, entity['geometry'])
        geometry = multipolygon_to_polygon(geometry, gdir=self)

        # Save transformed geometry to disk
        entity = entity.copy()
        entity['geometry'] = geometry

        # Do we want to use the RGI area or ours?
        if not cfg.PARAMS['use_rgi_area']:
            # Update Area
            area = geometry.area * 1e-6
            entity['Area'] = area

        # Avoid fiona bug: https://github.com/Toblerity/Fiona/issues/365
        for k, s in entity.iteritems():
            if type(s) in [np.int32, np.int64]:
                entity[k] = int(s)
        towrite = gpd.GeoDataFrame(entity).T
        towrite.crs = proj4_str

        # Write shapefile
        self.write_shapefile(towrite, 'outlines', filesuffix=filesuffix)

        # Also transform the intersects if necessary
        gdf = cfg.PARAMS['intersects_gdf']
        if len(gdf) > 0:
            gdf = gdf.loc[((gdf.RGIId_1 == self.rgi_id) |
                           (gdf.RGIId_2 == self.rgi_id))]
            if len(gdf) > 0:
                gdf = salem.transform_geopandas(gdf, to_crs=proj_out)
                if hasattr(gdf.crs, 'srs'):
                    # salem uses pyproj
                    gdf.crs = gdf.crs.srs
                self.write_shapefile(gdf, 'intersects')
        else:
            # Sanity check
            if cfg.PARAMS['use_intersects']:
                raise InvalidParamsError(
                    'You seem to have forgotten to set the '
                    'intersects file for this run. OGGM '
                    'works better with such a file. If you '
                    'know what your are doing, set '
                    "cfg.PARAMS['use_intersects'] = False to "
                    "suppress this error.")

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

    def write_inversion_settings(self,
                                 control_vars=['bed_h'],
                                 mb_models_settings={'MB1': {'type': 'constant', 'years': np.array([1950, 2016])}},
                                 min_w0_m=10.,
                                 observations=None,  # {'Area', {'2010': np.array([23])}}
                                 reg_parameters=None,  # [0, 0.1, 10]
                                 solver='L-BFGS-B', minimize_options=None,
                                 inversion_subdir='0', log_minimize_steps=True,
                                 bounds_min_max=None):
        """
        TODO: Documentation

        Parameters
        ----------
        control_vars
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

        if control_vars is None:
            control_vars = ['bed_h']
        inv_settings = locals()
        inv_settings.pop('self')

        write_pickle(inv_settings, self.get_filepath('inversion_settings'))