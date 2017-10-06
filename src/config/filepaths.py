'''
Contains the various file paths for the Python scripts
'''

sensor = 'himawari'  # goes

# set region based on sensor
if sensor == 'goes':
    region = 'Americas'
elif sensor == 'himawari':
    region = 'Asia'

# root path to data folder
#root_path = '/Users/dnf/Projects/kcl-fire-aot/data/{0}/'.format(region)
#root_path = '/Users/danielfisher/Projects/kcl-fire-aot/data/{0}/'.format(region)
root_path = '/Volumes/dfisher/data/{0}/'.format(region)

# processed data paths
path_to_processed_orac = root_path + 'processed/orac_proc/'
path_to_goes_frp = root_path + 'processed/goes_frp/goes13_2014_fire_frp_atm.csv'
path_to_smoke_plume_polygons = root_path + 'processed/plume_masks/myd021km_plumes_df.pickle'
path_to_smoke_plume_polygons_csv = root_path + 'processed/plume_masks/myd021km_plumes_df.csv'
path_to_processed_filelist = root_path + 'processed/plume_masks/processed_filelist.txt'

# raw data and data transfer paths
path_to_transfer_file = root_path + 'raw/rsync_file_list/files_to_transfer.txt'
path_to_modis_l1b = root_path + 'raw/modis/l1b/'
path_to_modis_geo = root_path + 'raw/modis/geo'
path_to_modis_frp = root_path + 'raw/modis/frp/'
path_to_modis_aod = root_path + 'raw/modis/aod/'
path_to_orac_aod = root_path + 'processed/orac/'
path_to_goes_l1b = root_path + 'raw/goes/'
path_to_himawari_l1b = root_path + 'raw/himawari/'

# FTP paths (MODIS / Ladsweb)
path_to_ladsweb_ftp = "ladsweb.nascom.nasa.gov"
path_to_myd03 = 'allData/6/MYD03/'
path_to_myd021km = 'allData/6/MYD021KM/'
path_to_myd14 = 'allData/6/MYD14/'
path_to_myd04_3K = 'allData/6/MYD04_3K/'
path_to_myd04 = 'allData/6/MYD04_L2/'
path_to_VAOTIP_L2 = 'allData/3110/NPP_VAOTIP_L2/'  # VIIRS 750m IP AOD
path_to_VMAE_L1 = 'allData/3110/NPP_VMAE_L1/'  # VIIRS 750m GEO for AOD

# HTTPS paths (GOES / class)
path_to_class_https_a = 'https://download.class.ncdc.noaa.gov/download/'
path_to_class_https_b = 'https://download.class.ngdc.noaa.gov/download/'

# temporary file paths for data downloads
path_to_goes_tmp = root_path + 'tmp/goes/'
path_to_modis_tmp = root_path + 'tmp/modis/'

# features filepaths
path_to_plume_classification_features = root_path + 'interim/classification_features.pickle'
path_to_reduced_plume_classification_features = root_path + 'interim/classification_features_20pc.pickle'

# model filepaths
path_to_rf_model = root_path + 'models/rf_model_64_trees.pickle'

# visualisation filepaths
path_to_orac_visuals = root_path + 'visualisations/orac/'
path_to_him_visualisations = root_path + 'visualisations/himawari/'

# filelist paths
path_to_filelists = root_path + 'filelists/'

# paths to frp
if sensor == 'goes':
    path_to_goes_frp = root_path + 'processed/goes_frp/'
elif sensor == 'himawari':
    path_to_himawari_frp = root_path + 'processed/himawari/'


