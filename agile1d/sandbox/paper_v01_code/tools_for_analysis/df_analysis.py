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
#     display_name: Python [conda env:oggm_env]
#     language: python
#     name: conda-env-oggm_env-py
# ---

import numpy as np


# # Function for filtering dataframe

# Usage with e.g.:
#
# ```python
# filter_dict = {
#     'glacier': ['Aletsch'],
#     'glacier_state': ['retreating'],
# }
# ```

def filter_dataframe(df, filter_dict_keep=None, filter_dict_remove=None):
    """
    usage example
    filter_dict = {
        'glacier': ['Aletsch'],
        'glacier_state': ['retreating'],
    }
    """
    filtered_df = df.copy()
    if filter_dict_keep is not None:
        for column, values in filter_dict_keep.items():
            values = np.atleast_1d(values)
            filtered_df = filtered_df[filtered_df[column].isin(values)]

    if filter_dict_remove is not None:
        for column, values in filter_dict_remove.items():
            values = np.atleast_1d(values)
            filtered_df = filtered_df[~filtered_df[column].isin(values)]
        
    return filtered_df


# # Function to sort df

def sort_dataframe(df, var):
    df = df.sort_values(by=var)
    df = df.reset_index(drop=True)
    return df


# # Function to return settings from abbreviations

def return_settings(df, settings):
    lam = settings['cost_lambda'][df['lambda']]
    obs = settings['observations'][df['observations']]
    reg = settings['regularisation_terms'][df['regularisation']]

    print(f"lambda {df['lambda']}: {lam}")
    print(f"observations {df['observations']}: {obs}")
    print(f"regularisation {df['regularisation']}: {reg}")


