import logging
import os

import pandas as pd
import numpy as np

import src.features.fre_to_tpm.viirs.ftt_utils as ut
import src.config.filepaths as fp
import src.config.constants as constants



def setup_data(ts):

    dd = dict()
    dd['viirs_aod'] = ut.sat_data_reader(fp.path_to_viirs_aod, 'viirs', 'aod', ts)
    dd['viirs_flags'] = ut.sat_data_reader(fp.path_to_viirs_aod, 'viirs', 'flags', ts)
    dd['orac_aod'] = ut.sat_data_reader(fp.path_to_viirs_orac, 'orac', 'aod', ts)
    dd['orac_flags'] = ut.sat_data_reader(fp.path_to_viirs_orac, 'orac', 'flags', ts)

    lats, lons = ut.sat_data_reader(fp.path_to_viirs_orac, 'orac', 'geo', ts)
    dd['lats'] = lats
    dd['lons'] = lons
    return dd


def main():

    df_list = []

    # set timestamp to check if new data loaded in
    previous_timestamp = ''

    plume_df = ut.read_plume_polygons(fp.path_to_smoke_plume_polygons_viirs_csv)
    for p_number, plume in plume_df.iterrows():

        # get plume time stamp
        current_timestamp = ut.get_timestamp(plume.filename)

        # read in satellite data
        if current_timestamp != previous_timestamp:

            try:
                sat_data = setup_data(current_timestamp)
            except Exception, e:
                logger.info('Could not load all datasets for: ' + plume.filename + '. Failed with error: ' + str(e))
                continue

            # set up resampler
            utm_rs = ut.utm_resampler(sat_data['lats'], sat_data['lons'], constants.utm_grid_size)

            # get the mask for the lats and lons and apply
            null_mask = np.ma.getmask(sat_data['orac_aod'])
            masked_lats = np.ma.masked_array(utm_rs.lats, null_mask)
            masked_lons = np.ma.masked_array(utm_rs.lons, null_mask)

            # resample all the datasets to UTM
            d = {}
            d['viirs_aod_utm'] = utm_rs.resample_image(sat_data['viirs_aod'], masked_lats, masked_lons, fill_value=0)
            d['viirs_flags_utm'] = utm_rs.resample_image(sat_data['viirs_flags'], masked_lats, masked_lons,
                                                         fill_value=0)
            d['orac_aod_utm'] = utm_rs.resample_image(sat_data['orac_aod'], masked_lats, masked_lons, fill_value=0)
            d['orac_flags_utm'] = utm_rs.resample_image(sat_data['orac_flags'], masked_lats, masked_lons, fill_value=0)
            d['lats'] = utm_rs.resample_image(utm_rs.lats, masked_lats, masked_lons, fill_value=0)
            d['lons'] = utm_rs.resample_image(utm_rs.lons, masked_lats, masked_lons, fill_value=0)
            previous_timestamp = current_timestamp

        # construct plume and background coordinate data
        try:
            plume_bounding_box = ut.construct_bounding_box(plume.plume_extent)
            plume_mask = ut.construct_mask(plume.plume_extent, plume_bounding_box)

        except Exception, e:
            logger.error(str(e))
            continue

        # subset the data to the rois
        viirs_aod_utm_plume = ut.subset_data(d['viirs_aod_utm'], plume_bounding_box)
        viirs_flag_utm_plume = ut.subset_data(d['viirs_flag_utm'], plume_bounding_box)
        orac_aod_utm_plume = ut.subset_data(d['orac_aod_utm'], plume_bounding_box)
        orac_cost_utm_plume = ut.subset_data(d['orac_cost_utm'], plume_bounding_box)

        # set up the dataframe for the plume and append to the dataframe list
        df = pd.DataFrame()
        df['viirs_aod'] = viirs_aod_utm_plume[plume_mask]
        df['viirs_flag'] = viirs_flag_utm_plume[plume_mask]
        df['orac_aod'] = orac_aod_utm_plume[plume_mask]
        df['orac_cost'] = orac_cost_utm_plume[plume_mask]

        df_list.append(df)

    plume_df = pd.concat(df_list)
    plume_df.to_csv(os.path.join(fp.path_to_dataframes, 'aod_comparison_dataframe.csv'))


if __name__=="__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()