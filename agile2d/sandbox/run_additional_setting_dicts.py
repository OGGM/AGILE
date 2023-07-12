from agile2d.sandbox import bed_measurement_masks

first_guess_bias_1_dict = {'use': True, 'desired_mean_bias': 20}
first_guess_bias_2_dict = {'use': True, 'desired_mean_bias': -20}
first_guess_rmse_1_dict = {'use': True, 'desired_rmse': 5, 'octaves': 4,
                           'base':2, 'freq': 3, 'glacier_only': True}
first_guess_rmse_2_dict_giluwe = {'use': True, 'desired_rmse': 20.2038,
                                  'octaves': 4, 'base':2, 'freq': 3,
                                  'glacier_only': True}
first_guess_rmse_2_dict_borden = {'use': True, 'desired_rmse': 20.1911,
                                  'octaves': 4, 'base':2, 'freq': 3,
                                  'glacier_only': True}

# surface noise dicts --
# TODO: not forget to adapt regularization set to a, b, c!
promised_land_1_dict = {'use': True, 'desired_rmse': 2, 'octaves': 4,
                            'base': 2, 'freq': 3, 'glacier_only': True}
promised_land_2_dict = {'use': True, 'desired_rmse': 6, 'octaves': 4,
                            'base': 2, 'freq': 3, 'glacier_only': True}
promised_land_3_dict = {'use': True, 'desired_rmse': 10, 'octaves': 4,
                            'base': 2, 'freq': 3, 'glacier_only': True}


# TODO: not forget to adapt regularization set to a, b, c!
giluwe_identical_twin_bed_measurement_dict = {
    'use': True, 'measurement_mask': bed_measurement_masks.Giluwe_upper_tongue,
    'std': 30, 'seed': 0
    }

giluwe_promised_land_3c_bed_measurement_dict = {
    'use': True, 'measurement_mask': bed_measurement_masks.Giluwe_cross,
    'std': 30, 'seed': 0
    }

borden_promised_land_3c_bed_measurement_dict = {
    'use': True, 'measurement_mask': bed_measurement_masks.Borden_horizontal,
    'std': 30, 'seed': 0
    }