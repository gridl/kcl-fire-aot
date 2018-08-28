'''
Contains the various file paths for the Python scripts
'''

import os


# root path to data folder
root = '/Volumes/INTENSO/kcl-fire-aot/ODA/'
#root = '/Users/danielfisher/Projects/kcl-fire-aot/data/ODA/'
roi = 'combined'

# select list of files to process
analysis_filelist_path = os.path.join(root, 'filelists', '{0}_files.txt'.format(roi))

# digitisation paths
root_pm = os.path.join(root, 'processed', 'plume_masks')
plume_polygon_path = os.path.join(root_pm, '{0}_plumes_df.p'.format(roi))
plume_polygon_path_csv = os.path.join(root_pm, '{0}_plumes_df.csv'.format(roi))
processed_filelist_path = os.path.join(root_pm, '{0}_processed_filelist.txt'.format(roi))

# resampled viirs for digitsing
root_digi = os.path.join(root, 'processed', 'viirs', 'resampled')
path_to_viirs_sdr_resampled_peat = os.path.join(root_digi, 'peat/')
path_to_viirs_sdr_resampled_no_peat = os.path.join(root_digi, 'no_peat/')
path_to_viirs_aod_resampled = os.path.join(root_digi, 'aod/')
path_to_viirs_aod_flags_resampled = os.path.join(root_digi, 'aod_flags/')
path_to_viirs_orac_resampled = os.path.join(root_digi, 'orac/')
path_to_viirs_orac_cost_resampled = os.path.join(root_digi, 'orac_cost/')

# analysis data paths
root_data = os.path.join(root, 'raw')
path_to_viirs_sdr = os.path.join(root_data, 'viirs', 'sdr')
path_to_viirs_aod = os.path.join(root_data, 'viirs', 'aod')
path_to_viirs_orac = os.path.join(root_data, 'viirs', 'orac')
path_to_himawari_imagery = os.path.join(root_data, 'himawari', 'imagery')
path_to_himawari_frp = os.path.join(root_data, 'himawari', 'frp')

# temporary file paths for data downloads
path_to_viirs_tmp = os.path.join(root, 'tmp', 'viirs')

# frp to tpm models filepath
root_models = os.path.join(root, 'models', 'fre_tpm_features')
path_to_frp_tpm_models = os.path.join(root_models, '{0}_samples.csv'.format(roi))

# visualisation filepaths
root_vis = os.path.join(root, 'visualisations')
pt_vis_path = os.path.join(root_vis, '{0}_plume_tracking/'.format(roi))
path_to_aeronet_visuals = os.path.join(root_vis, 'aeronet/')

# filelist paths

# dataframe paths
path_to_dataframes = os.path.join(root, 'interim', 'dataframes')


# external data paths
path_to_aeronet = os.path.join(root, 'external/aeronet')
