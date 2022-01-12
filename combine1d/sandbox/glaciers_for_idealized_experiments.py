# Here are the glaciers defined which are used for the experiments,
# t_bias_limits must be set by hand (set to None and use
# create_idealized_experiment function), and also set t_bias_spinup to speed
# up the creation of the experiments
experiment_glaciers = {
    'Baltoro': {
        'rgi_id': 'RGI60-14.06794',
        't_bias_spinup_limits': (-4, -3),
        't_bias_spinup': -3.903,
    },
    'Hintereisferner': {
        'rgi_id': 'RGI60-11.00897',
        't_bias_spinup_limits': (-1, 0),
        't_bias_spinup': -0.4671875,
    },
    'Aletsch': {
        'rgi_id': 'RGI60-11.01450',
        't_bias_spinup_limits': (-1, 0),
        't_bias_spinup': -0.4151348039215686,
    },
    'Artesonraju': {
        'rgi_id': 'RGI60-16.02444',
        't_bias_spinup_limits': (-1, 0),
        't_bias_spinup': -0.5234025753158407,
    },
    'Shallap': {
        'rgi_id': 'RGI60-16.02207',
        't_bias_spinup_limits': (-2, -1),
        't_bias_spinup': -1.9,
    },
}
