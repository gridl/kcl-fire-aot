import numpy as np

# set up filepaths and similar
root = '/Users/dnf/projects/kcl-fire-aot/data/'
#root = '/Volumes/INTENSO/'
orac_file_path = root + 'processed/orac_proc/'
goes_frp_file_path = root + 'processed/goes_frp/goes13_2014_fire_frp_atm.csv'
plume_mask_file_path = root + 'processed/plume_masks/myd021km_plumes_df.pickle'
plume_background_file_path = root + 'processed/plume_masks/myd021km_bg_df.pickle'
lc_file_path = root + 'external/land_cover/GLOBCOVER_L4_200901_200912_V2.3.tif'

# resampling parameters
padding = 10  # number of pixels around plume to additionally extract
res = 0.01  # grid resolution in degrees
radius_of_influence = 9000
fill_value = 0
