import logging
import os

import pandas as pd
import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt

import src.features.fre_to_tpm.viirs.ftt_utils as ut

import src.config.filepaths as fp


def main():

    df_list = []

    # set timestamp to check if new data loaded in
    previous_timestamp = ''

    plume_df = ut.read_plume_polygons(fp.path_to_smoke_plume_polygons_viirs_csv)
    for p_number, plume in plume_df.iterrows():

        # get plume time stamp
        current_timestamp = ut.get_timestamp(plume.filename)

        # if working on a new scene. Then set it up
        if current_timestamp != previous_timestamp:

            try:
                viirs_aod_data = ut.load_viirs(fp.path_to_viirs_aod, current_timestamp, plume.filename)
                orac_aod_data = ut.load_orac(fp.path_to_viirs_orac, current_timestamp)
            except Exception, e:
                logger.info('Could not load AOD data with error: ' + str(e))
                continue

            # set up resampler
            utm_image_resampler = ut.utm_resampler(orac_aod_data.variables['lat'][:],
                                                   orac_aod_data.variables['lon'][:],
                                                   750)

            # get the mask for the lats and lons and apply
            orac_aod = ut.orac_aod(orac_aod_data)
            mask = np.ma.getmask(orac_aod)
            masked_lats = np.ma.masked_array(utm_image_resampler.lats, mask)
            masked_lons = np.ma.masked_array(utm_image_resampler.lons, mask)

            viirs_aod_utm = utm_image_resampler.resample_image(ut.viirs_aod(viirs_aod_data),
                                                               masked_lats, masked_lons, fill_value=0)
            viirs_flag_utm = utm_image_resampler.resample_image(ut.viirs_flags(viirs_aod_data),
                                                                masked_lats, masked_lons, fill_value=0)
            orac_aod_utm = utm_image_resampler.resample_image(orac_aod,
                                                              masked_lats, masked_lons, fill_value=0)
            orac_cost_utm = utm_image_resampler.resample_image(ut.orac_cost(orac_aod_data),
                                                               masked_lats, masked_lons, fill_value=0)
            lats = utm_image_resampler.resample_image(utm_image_resampler.lats, masked_lats, masked_lons, fill_value=0)
            lons = utm_image_resampler.resample_image(utm_image_resampler.lons, masked_lats, masked_lons, fill_value=0)

            previous_timestamp = current_timestamp

        # construct plume and background coordinate data
        try:
            plume_bounding_box = ut.construct_bounding_box(plume.plume_extent)
            plume_mask = ut.construct_mask(plume.plume_extent, plume_bounding_box)

        except Exception, e:
            logger.error(str(e))
            continue

        # subset the data to the rois
        viirs_aod_utm_plume = ut.subset_data(viirs_aod_utm, plume_bounding_box)
        viirs_flag_utm_plume = ut.subset_data(viirs_flag_utm, plume_bounding_box)
        orac_aod_utm_plume = ut.subset_data(orac_aod_utm, plume_bounding_box)
        orac_cost_utm_plume = ut.subset_data(orac_cost_utm, plume_bounding_box)

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