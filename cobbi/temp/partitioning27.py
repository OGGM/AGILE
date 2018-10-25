from partitioning import core

# get path to the input data
input_shp = '/home/philipp/tmp/synthetic_ice_cap/Nanisivik Arctic Bay/per_glacier/Nanisivik Arctic Bay/ice_mask_1500m_2800a.shp'
input_dem = '/home/philipp/tmp/synthetic_ice_cap/Nanisivik Arctic Bay/per_glacier/Nanisivik Arctic Bay/surface_1500m_2800a_40m_res.tiff'

# filter options
f_area = True
f_alt_range = False
f_perc_alt_range = True

core.dividing_glaciers(input_dem, input_shp, filter_area=f_area,
                       filter_alt_range=f_alt_range,
                       filter_perc_alt_range=f_perc_alt_range)
print('end')