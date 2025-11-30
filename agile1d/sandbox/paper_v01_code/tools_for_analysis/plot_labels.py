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

# # short labels for observations

short_labels_obs = {
    'fl_surface_h:m': r'$sfc$',
    'dmdtda:kg m-2 yr-1': r'$\Delta M$',
    'volume:km3': r'$V$'
}


# # multiple short labels for observaitons

def get_multiple_short_labels_obs(dict_all_obs_settings):
    all_labels = []

    for obs_all in dict_all_obs_settings:
        obs_all = list(dict_all_obs_settings[obs_all].keys())
        obs_label = short_labels_obs[obs_all[0]]
        if len(obs_all) > 1:
            for other_obs in obs_all[1:]:
                obs_label += f'\n{short_labels_obs[other_obs]}'

        all_labels.append(obs_label)

    return all_labels


