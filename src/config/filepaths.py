'''
Contains the various file paths for the Python scripts
'''

import os


# root path to data folder
#root_path = '/Users/dnf/Projects/kcl-fire-aot/data/{0}/'.format(region)
#root_path = '/Users/danielfisher/Projects/kcl-fire-aot/data/{0}/'.format(region)
#root_path = '/Volumes/dfisher/data/{0}/'.format(region)
root_path = '/Volumes/INTENSO/kcl-fire-aot/ODA/'

# processed data paths
path_to_processed_orac = root_path + 'processed/orac_proc/'
path_to_smoke_plume_polygons_viirs = root_path + 'processed/plume_masks/viirs_plumes_sumatra_df.pickle'
path_to_smoke_plume_polygons_viirs_csv = root_path + 'processed/plume_masks/viirs_plumes_sumatra_df.csv'
path_to_processed_filelist_viirs = root_path + 'processed/plume_masks/processed_filelist_viirs.txt'

# raw data and data transfer paths
path_to_viirs_aod = root_path + 'raw/viirs/unprojected/aod/'
path_to_viirs_orac = root_path + 'processed/orac/viirs/'
path_to_aeronet = os.path.join(root_path, 'external/aeronet')
path_to_himawari_l1b = root_path + 'raw/himawari/'

# resampled viirs for digitsing
digi_path = os.path.join(root_path, 'raw/viirs/sumatra_roi/resampled')
path_to_viirs_sdr_resampled_peat = os.path.join(digi_path, 'peat/')
path_to_viirs_aod_resampled = os.path.join(digi_path, 'aod/')
path_to_viirs_aod_flags_resampled = os.path.join(digi_path, 'aod_flags/')
path_to_viirs_orac_resampled = os.path.join(digi_path, 'orac/')
path_to_viirs_orac_cost_resampled = os.path.join(digi_path, 'orac_cost/')


# temporary file paths for data downloads
path_to_viirs_tmp = root_path + 'tmp/viirs/'

# features filepaths
path_to_frp_tpm_features = root_path + 'interim/fre_tpm_features/'

# visualisation filepaths
path_to_plume_tracking_visualisations_viirs = root_path + 'visualisations/viirs/plume_tracking/'
path_to_aeronet_visuals = os.path.join(root_path, 'visualisations/aeronet/')

# filelist paths
path_to_filelists = root_path + 'filelists/'

# dataframe paths
path_to_dataframes = os.path.join(root_path, 'interim', 'dataframes')





