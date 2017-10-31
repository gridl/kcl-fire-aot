import logging
import os

import numpy as np

import src.features.fre_to_tpm.ftt_tpm as tt
import src.features.fre_to_tpm.ftt_utils as ut
import src.features.fre_to_tpm.ftt_plume_tracking as pt
import src.features.fre_to_tpm.ftt_fre as ff
import src.config.filepaths as fp
import src.data.readers.load_hrit as load_hrit

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def main():

    # load in static data
    frp_df = ut.read_frp_df(fp.path_to_himawari_frp)
    plume_df = ut.read_plume_polygons(fp.path_to_smoke_plume_polygons_csv)
    #lc_data = ut.read_nc(fp.path_to_landcover)

    geo_file = fp.root_path + '/processed/himawari/Himawari_lat_lon.img'
    geostationary_lats, geostationary_lons = load_hrit.geo_read(geo_file)

    # set up arrays to hold data
    plumes_numbers = []
    fre = []
    tpm = []
    lc = []

    # itereate over the plumes
    for p_number, plume in plume_df.iterrows():

        # make a directory to hold the plume logging information
        plume_logging_path = os.path.join(fp.path_to_plume_tracking_visualisations, str(p_number))
        if not os.path.isdir(plume_logging_path):
            os.mkdir(plume_logging_path)


        # get plume time stamp
        timestamp_mxd = ut.get_timestamp(plume.filename)

        # load plume aod datasets
        try:
            mxd04_fname = ut.get_modis_fname(fp.path_to_modis_aod, timestamp_mxd, plume.filename)
            mxd04 = ut.read_hdf(os.path.join(fp.path_to_modis_aod, mxd04_fname))
            mxd_aod = ut.aod_mxd04(mxd04)

            orac_fname = ut.get_orac_fname(fp.path_to_orac_aod, timestamp_mxd)
            orac_data = ut.read_nc(os.path.join(fp.path_to_orac_aod, orac_fname))
            orac_aod = ut.aod_orac(orac_data)
        except Exception, e:
            logger.error(str(e))
            continue

        # get geo data
        scene_lats, scene_lons = ut.read_modis_geo(fp.path_to_modis_l1b, plume)

        # construct plume and background coordinate data
        try:
            plume_bounding_box = ut.construct_bounding_box(plume.plume_extent)
            plume_lats = ut.subset_data(scene_lats, plume_bounding_box)
            plume_lons = ut.subset_data(scene_lons, plume_bounding_box)
            plume_points = ut.construct_points(plume, plume_bounding_box, plume_lats, plume_lons)
            plume_polygon = ut.construct_polygon(plume, plume_bounding_box, plume_lats, plume_lons)
            plume_mask = ut.construct_mask(plume.plume_extent, plume_bounding_box)

            background_bounding_box = ut.construct_bounding_box(plume.background_extent)
            background_mask = ut.construct_mask(plume.background_extent, background_bounding_box)
            background_lats = ut.subset_data(scene_lats, background_bounding_box)
            background_lons = ut.subset_data(scene_lons, background_bounding_box)

        except Exception, e:
            logger.error(str(e))
            continue

        # subset ORAC and MYD04, MYD14 datasets
        orac_aod_plume = ut.subset_data(orac_aod, plume_bounding_box)
        orac_aod_background = ut.subset_data(orac_aod, background_bounding_box)
        mxd04_aod_background = ut.subset_data(mxd_aod, background_bounding_box)
        mxd04_mask = mxd04_aod_background < 0  # mask out non-aods
        mxd04_aod_background = np.ma.masked_array(mxd04_aod_background, mxd04_mask)

        # get the modis fire information for the plume
        myd14_fname = ut.get_modis_fname(fp.path_to_modis_frp, timestamp_mxd, plume.filename)
        myd14 = ut.read_hdf(os.path.join(fp.path_to_modis_frp, myd14_fname))
        myd14 = ut.fires_myd14(myd14)
        fires = ut.extract_fires(fp.path_to_modis_l1b, plume, myd14)
        fires_lats, fires_lons = ut.fires_in_plume(fires, plume_polygon)

        # reproject all modis datasets to utm (mask, orac_aod, MYD04)
        utm_resampler = ut.utm_resampler(plume_lats, plume_lons, 1000)

        utm_plume_points = ut.reproject_shapely(plume_points, utm_resampler)
        utm_plume_polygon = ut.reproject_shapely(plume_polygon, utm_resampler)
        utm_plume_mask = utm_resampler.resample_image(plume_mask, plume_lats, plume_lons)
        utm_bg_mask = utm_resampler.resample_image(background_mask, background_lats, background_lons)

        utm_orac_aod_plume = utm_resampler.resample_image(orac_aod_plume, plume_lats, plume_lons)
        utm_orac_aod_background = utm_resampler.resample_image(orac_aod_background,
                                                               background_lats, background_lons)
        utm_mxd04_aod_background = utm_resampler.resample_image(mxd04_aod_background,
                                                                background_lats, background_lons)

        utm_fires = utm_resampler.resample_points_to_utm(fires_lats, fires_lons)

        # get FRP integration start and stop times
        try:
            start_time, stop_time, plume_stats = pt.find_integration_start_stop_times(p_number,
                                                                                      plume_logging_path,
                                                                                      plume.filename,
                                                                                      utm_plume_points, utm_plume_mask,
                                                                                      plume_lats, plume_lons,
                                                                                      geostationary_lats, geostationary_lons,
                                                                                      utm_fires,
                                                                                      utm_resampler,
                                                                                      plot=True)
        except Exception, e:
            logger.error(str(e))
            continue

        # get the variables of interest
        if start_time is not None:
            plumes_numbers.append(p_number)
            fre.append(ff.compute_fre(p_number, plume_logging_path,
                                      utm_plume_polygon, frp_df, start_time, stop_time, utm_resampler))
            tpm.append(tt.compute_tpm(utm_orac_aod_plume, utm_orac_aod_background, utm_mxd04_aod_background,
                                      utm_plume_polygon, utm_plume_mask, utm_bg_mask))
            #lc.append(ut.find_landcover_class(fires_lats, fires_lons, lc_data))

    # split data based on lc type

    # compute models


if __name__=="__main__":
    main()
