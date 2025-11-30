# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.5
#   kernelspec:
#     display_name: Python [conda env:agile_env]
#     language: python
#     name: conda-env-agile_env-py
# ---

from oggm import cfg, utils, tasks, workflow
from oggm.core.flowline import FileModel
from agile1d.sandbox.glaciers_for_idealized_experiments import experiment_glaciers
import os
import gzip
import pickle
import numpy as np

cfg.initialize()
working_dir = os.path.abspath("../../../../../")
cfg.PATHS['working_dir'] = working_dir
gdirs = workflow.init_glacier_directories()

working_dir
fp_gdirs = {
    'RGI60-16.02444': os.path.join(working_dir, "per_glacier/RGI60-16/RGI60-16.02/RGI60-16.02444/"),
    'RGI60-11.01450': os.path.join(working_dir, "per_glacier/RGI60-11/RGI60-11.01/RGI60-11.01450/"),
    'RGI60-02.05098': os.path.join(working_dir, "per_glacier/RGI60-02/RGI60-02.05/RGI60-02.05098/"),
    'RGI60-14.06794': os.path.join(working_dir, "per_glacier/RGI60-14/RGI60-14.06/RGI60-14.06794/"),
}


# # Return full gdirs

def get_experiment_gdirs():
    return gdirs


def get_rgi_id_for_glacier(glacier):
    return experiment_glaciers[glacier]['rgi_id']


def get_name_from_rgi_id(rgi_id):
    for glacier in experiment_glaciers:
        if experiment_glaciers[glacier]['rgi_id'] == rgi_id:
            return glacier

    raise ValueError(f'No glacier with rgi_id {rgi_id} defined!')


# # Return reference flowline of experiment

def read_pickle(glacier, filename, use_compression=None, filesuffix=''):
    """Reads a pickle located in the directory.

    Parameters
    ----------
    filename : str
        file name (must be listed in cfg.BASENAME)
    use_compression : bool
        whether or not the file ws compressed. Default is to use
        cfg.PARAMS['use_compression'] for this (recommended)
    filesuffix : str
        append a suffix to the filename (useful for experiments).

    Returns
    -------
    An object read from the pickle
    """

    use_comp = (use_compression if use_compression is not None
                else cfg.PARAMS['use_compression'])
    _open = gzip.open if use_comp else open
    fp = os.path.join(
            fp_gdirs[get_rgi_id_for_glacier(glacier)],
            f'{filename}{filesuffix}.pkl'
    )
    with _open(fp, 'rb') as f:
        try:
            out = pickle.load(f)
        except ModuleNotFoundError as err:
            if err.name == "shapely.io":
                err.msg = "You need shapely version 2.0 or higher for this to work."
            raise

    # Some new attrs to add to old pre-processed directories
    if filename == 'model_flowlines':
        if getattr(out[0], 'map_trafo', None) is None:
            try:
                # This may fail for very old gdirs
                grid = self.grid
            except InvalidWorkflowError:
                return out

            # Add the trafo
            trafo = partial(grid.ij_to_crs, crs=salem.wgs84)
            for fl in out:
                fl.map_trafo = trafo

    return out


def get_fl_geometry_experiment(glacier, glacier_state):
    fmod = FileModel(os.path.join(
        fp_gdirs[get_rgi_id_for_glacier(glacier)],
        f'model_geometry_agile_true_end_{glacier_state}.nc'
    ))
    return fmod


def get_fl_at_year(glacier, glacier_state, year):
    if year in np.arange(2000, 2021):
        fmod = get_fl_geometry_experiment(glacier, glacier_state)
        fmod.run_until(year)
        fl = fmod.fls[0]

    elif year == 1980:
        fl = read_pickle(glacier, 'model_flowlines',
                         filesuffix=f'_creation_spinup_{glacier_state}')[0]
    return fl



