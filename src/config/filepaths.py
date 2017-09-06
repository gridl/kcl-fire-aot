'''
Contains the various file paths for the Python scripts
'''

import sensor

# set region based on sensor
if sensor.sensor == 'goes':
    region = 'Americas'
elif sensor.sensor == 'himawari':
    region = 'Asia'

# root path to data folder
#root_path = '/Users/dnf/Projects/kcl-fire-aot/data/{0}/'.format(region)
root_path = '/Users/danielfisher/Projects/kcl-fire-aot/data/{0}/'.format(region)

# processed data paths
path_to_processed_orac = root_path + 'processed/orac_proc/'
path_to_goes_frp = root_path + 'processed/goes_frp/goes13_2014_fire_frp_atm.csv'
path_to_smoke_plume_masks = root_path + 'processed/plume_masks/myd021km_plumes_df.pickle'
path_to_smoke_plume_masks_csv = root_path + 'processed/plume_masks/myd021km_plumes_df.csv'
path_to_ml_smoke_plume_masks = root_path + 'processed/plume_masks/myd021km_plumes_ml_df.pickle'
path_to_ml_smoke_free_masks = root_path + 'processed/plume_masks/myd021km_smoke_free_ml_df.pickle'
path_to_background_masks = root_path + 'processed/plume_masks/myd021km_bg_df.pickle'
path_to_processed_filelist = root_path + 'processed/plume_masks/processed_filelist.txt'

# raw data and data transfer paths
path_to_transfer_file = root_path + 'raw/rsync_file_list/files_to_transfer.txt'
path_to_modis_l1b = root_path + 'raw/modis/l1b/'
path_to_modis_geo = root_path + 'raw/modis/geo'
path_to_modis_frp = root_path + 'raw/modis/frp/'
path_to_modis_aod_3k = root_path + 'raw/modis/MYD04_3K/'
path_to_modis_aod = root_path + 'raw/modis/MYD04/'
path_to_goes_l1b = root_path + 'raw/goes/'
path_to_viirs_aod = root_path + 'raw/viirs/
path_to_viirs_geo = root_path + 'raw/viirs/

# FTP paths (MODIS / Ladsweb)
path_to_ladsweb_ftp = "ladsweb.nascom.nasa.gov"
path_to_myd03 = 'allData/6/MYD03/'
path_to_myd021km = 'allData/6/MYD021KM/'
path_to_myd14 = 'allData/6/MYD14/'
path_to_myd04_3K = 'allData/6/MYD04_3K/'
path_to_myd04 = 'allData/6/MYD04_L2/'
path_to_VAOTIP_L2 = 'allData/3110/NPP_VAOTIP_L2/'
path_to_GMTCO = 'allData/3110/'


# HTTPS paths (GOES / class)
path_to_class_https_a = 'https://download.class.ncdc.noaa.gov/download/'
path_to_class_https_b = 'https://download.class.ngdc.noaa.gov/download/'

# temporary file paths
path_to_goes_tmp = root_path + 'tmp/goes/'
path_to_modis_tmp = root_path + 'tmp/modis/'

# paths to frp
if sensor.sensor == 'goes':
    path_to_goes_frp = root_path + 'processed/goes_frp/'
elif sensor.sensor == 'himawari':
    path_to_himawari_frp = root_path + 'processed/himawari/'

# features filepaths
path_to_plume_classification_features = root_path + 'interim/classification_features.pickle'
path_to_reduced_plume_classification_features = root_path + 'interim/classification_features_20pc.pickle'

# model filepaths
path_to_rf_model = root_path + 'models/rf_model_64_trees.pickle'

# visualisation filepaths
path_to_orac_visuals = root_path + 'visualisations/orac/'

# filelist paths
path_to_filelists = root_path + 'filelists/'
