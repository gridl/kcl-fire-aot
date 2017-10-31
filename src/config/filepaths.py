'''
Contains the various file paths for the Python scripts
'''

# TODO replace all + with os.path.join

sensor = 'himawari'  # goes

# set region based on sensor
if sensor == 'goes':
    region = 'Americas'
elif sensor == 'himawari':
    region = 'Asia'

# root path to data folder
#root_path = '/Users/dnf/Projects/kcl-fire-aot/data/{0}/'.format(region)
#root_path = '/Users/danielfisher/Projects/kcl-fire-aot/data/{0}/'.format(region)
#root_path = '/Volumes/dfisher/data/{0}/'.format(region)
root_path = '/Volumes/INTENSO/{0}/'.format(region)

# processed data paths
path_to_processed_orac = root_path + 'processed/orac_proc/'
path_to_smoke_plume_polygons = root_path + 'processed/plume_masks/myd021km_plumes_df.pickle'
path_to_smoke_plume_polygons_csv = root_path + 'processed/plume_masks/myd021km_plumes_df.csv'
path_to_processed_filelist = root_path + 'processed/plume_masks/processed_filelist.txt'
if sensor == 'goes':
    path_to_goes_frp = root_path + 'processed/goes_frp/'
elif sensor == 'himawari':
    path_to_himawari_frp = root_path + 'processed/himawari/'

# raw data and data transfer paths
path_to_transfer_file = root_path + 'raw/rsync_file_list/files_to_transfer.txt'
path_to_modis_l1b = root_path + 'raw/modis/l1b/'
path_to_modis_geo = root_path + 'raw/modis/geo'
path_to_modis_frp = root_path + 'raw/modis/frp/'
path_to_modis_aod = root_path + 'raw/modis/aod/'
path_to_orac_aod = root_path + 'processed/orac/'
path_to_goes_l1b = root_path + 'raw/goes/'
path_to_himawari_l1b = root_path + 'raw/himawari/'
path_to_landcover = root_path.replace(region, 'Global') + 'land_cover/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.nc'

# FTP paths (MODIS / Ladsweb)
path_to_ladsweb_ftp = "ladsweb.nascom.nasa.gov"
path_to_all_data = 'allData/6/'

# HTTPS paths (GOES / class)
path_to_class_https_a = 'https://download.class.ncdc.noaa.gov/download/'
path_to_class_https_b = 'https://download.class.ngdc.noaa.gov/download/'

# temporary file paths for data downloads
path_to_goes_tmp = root_path + 'tmp/goes/'
path_to_modis_tmp = root_path + 'tmp/modis/'

# features filepaths
path_to_plume_classification_features = root_path + 'interim/classification_features.pickle'
path_to_reduced_plume_classification_features = root_path + 'interim/classification_features_20pc.pickle'
path_to_frp_tpm_features = root_path + 'interim/fre_tpm_features/'

# model filepaths
path_to_rf_model = root_path + 'models/rf_model_64_trees.pickle'

# visualisation filepaths
path_to_orac_visuals = root_path + 'visualisations/orac/'
path_to_plume_tracking_visualisations = root_path + 'visualisations/plume_tracking/'
path_to_plume_tracking_visualisations = root_path + 'visualisations/plume_tracking/'

# filelist paths
path_to_filelists = root_path + 'filelists/'



