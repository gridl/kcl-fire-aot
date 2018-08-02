import os

data_root = '/group_workspaces/jasmin2/nceo_aerosolfire/data/orac_proc/viirs/'
home_root = '/home/users/dnfisher/data/kcl-fire-aot/'
png_root = os.path.join(home_root, 'resampled_pngs/')

path_to_viirs_sdr = os.path.join(data_root, 'sdr/sumatra_roi/')
path_to_viirs_geo = os.path.join(data_root, 'geo/sumatra_roi/')
path_to_viirs_orac = os.path.join(data_root, 'outputs/sumatra_roi/main/')
path_to_viirs_aod = os.path.join(home_root, 'viirs_aod')


path_to_himawari_frp = os.path.join(home_root, 'frp/')
path_to_peat_maps = os.path.join(home_root, 'peat_maps/')

path_to_viirs_sdr_resampled_peat = os.path.join(png_root, 'peat/')
path_to_viirs_sdr_resampled_no_peat = os.path.join(png_root, 'no_peat/')
path_to_viirs_aod_resampled = os.path.join(png_root, 'aod/')
path_to_viirs_aod_flags_resampled = os.path.join(png_root, 'aod_flags/')
path_to_viirs_orac_resampled = os.path.join(png_root, 'orac/')
path_to_viirs_orac_cost_resampled = os.path.join(png_root, 'orac_cost/')
path_to_resampled_peat_map = os.path.join(png_root, 'peat_maps/')
