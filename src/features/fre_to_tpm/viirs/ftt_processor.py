import logging
import os

import numpy as np
import pandas as pd
import src.features.fre_to_tpm.viirs.ftt_fre as ff
import src.features.fre_to_tpm.viirs.ftt_plume_tracking as pt
import src.features.fre_to_tpm.viirs.ftt_utils as ut

import src.config.filepaths as fp
import src.data.readers.load_hrit as load_hrit
import src.features.fre_to_tpm.viirs.ftt_tpm as tt

import matplotlib.pyplot as plt

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def main():

    # load in static data
    frp_df = ut.read_frp_df(fp.path_to_himawari_frp)
    plume_df = ut.read_plume_polygons(fp.path_to_smoke_plume_polygons_viirs_csv)
    #lc_data = ut.read_nc(fp.path_to_landcover)
    geo_file = fp.root_path + '/processed/himawari/Himawari_lat_lon.img'
    geostationary_lats, geostationary_lons = load_hrit.geo_read(geo_file)

    # setup output path to hold csv
    output_path = os.path.join(fp.path_to_frp_tpm_features, 'model_features_viirs.csv')

    # set up arrays to hold data
    plumes_numbers = []
    fre = []
    tpm = []
    #lc = []

    # set timestamp to check if new data laoded in
    previous_timestamp = ''

    # itereate over the plumes
    for p_number, plume in plume_df.iterrows():

        # make a directory to hold the plume logging information
        plume_logging_path = os.path.join(fp.path_to_plume_tracking_visualisations_viirs, str(p_number))
        if not os.path.isdir(plume_logging_path):
            os.mkdir(plume_logging_path)

        # get plume time stamp
        current_timestamp = ut.get_timestamp(plume.filename)

        # if working on a new scene. Then set it up
        if current_timestamp != previous_timestamp:

            viirs_data = ut.load_viirs(fp.path_to_viirs_aod, current_timestamp, plume.filename)
            orac_data = ut.load_orac(fp.path_to_viirs_orac, current_timestamp)

            # set up resampler
            utm_resampler = ut.utm_resampler(orac_data.variables['lat'][:],
                                             orac_data.variables['lon'][:],
                                             750)

            # get the mask for the lats and lons and apply
            orac_aod = ut.orac_aod(orac_data)
            mask = np.ma.getmask(orac_aod)
            masked_lats = np.ma.masked_array(utm_resampler.lats, mask)
            masked_lons = np.ma.masked_array(utm_resampler.lons, mask)

            viirs_aod_utm = utm_resampler.resample_image(ut.viirs_aod(viirs_data),
                                                         masked_lats, masked_lons, fill_value=0)
            viirs_flag_utm = utm_resampler.resample_image(ut.viirs_flags(viirs_data),
                                                         masked_lats, masked_lons, fill_value=0)
            orac_aod_utm = utm_resampler.resample_image(orac_aod,
                                                         masked_lats, masked_lons, fill_value=0)
            orac_cost_utm = utm_resampler.resample_image(ut.orac_cost(orac_data),
                                                         masked_lats, masked_lons, fill_value=0)
            lats = utm_resampler.resample_image(utm_resampler.lats, masked_lats, masked_lons, fill_value=0)
            lons = utm_resampler.resample_image(utm_resampler.lons,masked_lats, masked_lons, fill_value=0)

            previous_timestamp = current_timestamp


        # construct plume and background coordinate data
        try:
            plume_bounding_box = ut.construct_bounding_box(plume.plume_extent)
            plume_lats = ut.subset_data(lats, plume_bounding_box)
            plume_lons = ut.subset_data(lons, plume_bounding_box)

            plume_vector = ut.construct_vector(plume, plume_bounding_box, plume_lats, plume_lons)
            plume_points = ut.construct_points(plume, plume_bounding_box, plume_lats, plume_lons)
            plume_polygon = ut.construct_polygon(plume, plume_bounding_box, plume_lats, plume_lons)
            plume_mask = ut.construct_mask(plume.plume_extent, plume_bounding_box)

            background_bounding_box = ut.construct_bounding_box(plume.background_extent)
            background_mask = ut.construct_mask(plume.background_extent, background_bounding_box)
            #utm_background_lats = ut.subset_data(lats_utm, utm_background_bounding_box)
            #utm_background_lons = ut.subset_data(lons_utm, utm_background_bounding_box)

        except Exception, e:
            logger.error(str(e))
            continue


        # subset the data to the rois
        viirs_aod_utm_plume = ut.subset_data(viirs_aod_utm, plume_bounding_box)
        viirs_flag_utm_plume = ut.subset_data(viirs_flag_utm, plume_bounding_box)
        orac_aod_utm_plume = ut.subset_data(orac_aod_utm, plume_bounding_box)
        orac_cost_utm_plume = ut.subset_data(orac_cost_utm, plume_bounding_box)

        viirs_aod_utm_background = ut.subset_data(viirs_aod_utm, plume_bounding_box)
        viirs_flag_utm_background = ut.subset_data(viirs_flag_utm, plume_bounding_box)

        # reproject vector data
        utm_resampler_plume = ut.utm_resampler(plume_lats, plume_lons, 750)
        utm_plume_points = ut.reproject_shapely(plume_points, utm_resampler_plume)
        utm_plume_polygon = ut.reproject_shapely(plume_polygon, utm_resampler_plume)
        utm_plume_vector = ut.reproject_shapely(plume_vector, utm_resampler_plume)


        # get FRP integration start and stop times
        try:
            start_time, stop_time, plume_stats = pt.find_integration_start_stop_times(p_number,
                                                                                      plume_logging_path,
                                                                                      utm_plume_points, plume_mask,
                                                                                      plume_lats, plume_lons,
                                                                                      utm_plume_vector,
                                                                                      geostationary_lats,
                                                                                      geostationary_lons,
                                                                                      utm_resampler_plume,
                                                                                      current_timestamp,
                                                                                      frp_df,
                                                                                      utm_plume_polygon,
                                                                                      plot=True)
        except Exception, e:
            logger.error(str(e))
            continue

        # get the variables of interest
        if start_time is not None:
            plumes_numbers.append(p_number)
            fre.append(ff.compute_fre(p_number, plume_logging_path,
                                      utm_plume_polygon, frp_df, start_time, stop_time, utm_resampler))
            tpm.append(tt.compute_tpm(viirs_aod_utm_plume, viirs_flag_utm_plume,
                                      orac_aod_utm_plume, orac_cost_utm_plume,
                                      viirs_aod_utm_background, viirs_flag_utm_background,
                                      utm_plume_polygon, plume_mask, background_mask))
            #lc.append(ut.find_landcover_class(fires_lats, fires_lons, lc_data))

    # dump data to csv via df
    #data = np.array([fre, tpm, lc, plumes_numbers]).T
    #columns = ['fre', 'tpm', 'lc', 'plume_number']
    data = np.array([fre, tpm, plumes_numbers]).T
    columns = ['fre', 'tpm', 'plume_number']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_path)


if __name__=="__main__":
    main()
