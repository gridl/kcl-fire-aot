'''
Going from FRE to TPM for different land cover types.

We generate a linear model that relates FRE to TPM.

Basic logic:
    Load in land cover data
    For each digitised file:
        Resample landcover map to modis scene
        For each plume:
            Get plume AOD
            Get background AOD
            Get landcover type from MODIS fire pixels
            Get FRE from geostationary sensor



'''

import ast
import logging

import pandas as pd
import numpy as np
import pyresample as pr

import src.data.readers as readers
import src.config.filepaths as filepaths


# TODO include subsetting by latitudes in this functiom.
def resample_landcover(lc_data, sub_lats, sub_lons):

    # Dealing with dateline crossings.
    if (np.min(sub_lons) - -180) < 0.01:
        first_line_max_lon = np.max(sub_lons[0,:])
        min_max_lon = np.min(sub_lons[sub_lons > first_line_max_lon]) # the min of the longitudes that cross dateline

        lon_index_west = int((first_line_max_lon - (-180)) / 360 * lc_data['lon'].size)  # lon index
        lon_index_east = int((min_max_lon - (-180)) / 360 * lc_data['lon'].size)  # lon index
        lat_index_north = int((np.max(sub_lats) - 90) * -1 / 180 * lc_data['lat'].size) # lat index
        lat_index_south = int((np.min(sub_lats) - 90) * -1 / 180 * lc_data['lat'].size)  # lat index

        # we can run from 0 up to lon_index
        water_mask_west = np.array(lc_data['wb_class'][lat_index_north:lat_index_south, 0:lon_index_west])
        water_mask_east = np.array(lc_data['wb_class'][lat_index_north:lat_index_south, lon_index_east:])

        # now lets join the masks
        water_mask = np.concatenate((water_mask_east, water_mask_west), axis=1)

        # build the geo grids for the masks
        lons_west = np.tile(lc_data['lon'][0:lon_index_west], (water_mask_west.shape[0], 1))
        lons_east = np.tile(lc_data['lon'][lon_index_east:], (water_mask_east.shape[0], 1))
        lons = np.concatenate((lons_east, lons_west), axis=1)
        lats = np.transpose(np.tile(lc_data['lat'][lat_index_north:lat_index_south],
                                    (water_mask_east.shape[1] + water_mask_west.shape[1], 1)))

    else:
        lon_index_west = int((np.min(sub_lons) - (-180)) / 360 * lc_data['lon'].size)  # lon index
        lon_index_east = int((np.max(sub_lons) - (-180)) / 360 * lc_data['lon'].size)  # lon index
        lat_index_north = int((np.max(sub_lats) - 90) * -1 / 180 * lc_data['lat'].size)  # lat index
        lat_index_south = int((np.min(sub_lats) - 90) * -1 / 180 * lc_data['lat'].size)  # lat index

        water_mask = np.array(lc_data['wb_class'][lat_index_north:lat_index_south, lon_index_west:lon_index_east])
        lons = np.tile(lc_data['lon'][lon_index_west:lon_index_east], (water_mask.shape[0], 1))
        lats = np.transpose(np.tile(lc_data['lat'][lat_index_north:lat_index_south], (water_mask.shape[1], 1)))

    return water_mask, lons, lats


def main():

    # set the filepaths up here
    root_path = '/Users/dnf/Projects/kcl-fire-aot/data/'
    landcover_path = root_path + 'Global/land_cover/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7_900m.nc'
    mask_path = root_path + 'Americas/processed/plume_masks/myd021km_plumes_df.pickle'

    try:
        mask_df = pd.read_pickle(mask_path)
    except:
        mask_df = pd.read_csv(mask_path, quotechar='"', sep=',', converters={'plume_extent': ast.literal_eval})


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
