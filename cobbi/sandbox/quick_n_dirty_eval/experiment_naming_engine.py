
def get_experiment_name(exp_folder_name):
    exp_dict = {
        'identical twin': 'identical-twin a',
        'identical twin control perturb surf': 'identical-twin b',
        'identical twin control perturb surf 2': 'identical-twin c',

        'first guess bias 1': 'first guess bias 1',
        'first guess bias 2': 'first guess bias 2',
        'first guess rmse 1': 'first guess rmse 1',
        'first guess rmse 2': 'first guess rmse 2',

        'fin 01 scaling 02 0e7': 'promised land 1a',
        'fin 10 scaling 02 0e7': 'promised land 1b',
        'fin 10 scaling 02 1e7': 'promised land 1c',
        'fin 01 scaling 06 0e7': 'promised land 2a',
        'fin 10 scaling 06 0e7': 'promised land 2b',
        'fin 10 scaling 06 1e7': 'promised land 2c',
        'fin 01 scaling 10 0e7': 'promised land 3a',
        'fin 10 scaling 10 0e7': 'promised land 3b',
        'fin 10 scaling 10 1e7': 'promised land 3c',

        'fin 01 scaling 02 0e7': 'promised land 1a',
        'fin 200 scaling 02 0e7': 'promised land 1b',
        'fin 200 scaling 02 1e7': 'promised land 1c',
        'fin 01 scaling 06 0e7': 'promised land 2a',
        'fin 200 scaling 06 0e7': 'promised land 2b',
        'fin 200 scaling 06 1e7': 'promised land 2c',
        'fin 01 scaling 10 0e7': 'promised land 3a',
        'fin 200 scaling 10 0e7': 'promised land 3b',
        'fin 200 scaling 10 1e7': 'promised land 3c',

        '3c std 30 l5 1e0': 'promised land 3c plus bed',
        '3c std 30 l5 1e-0': 'promised land 3c plus bed',
        'upper tongue std 30 l5 1e-0': 'identical-twin a plus bed'
    }
    if exp_folder_name in exp_dict:
        return exp_dict[exp_folder_name]
    else:
        return None