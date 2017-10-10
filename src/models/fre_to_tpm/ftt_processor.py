import re
import os
import logging

from pyhdf.SD import SD, SDC
import numpy as np

import src.models.fre_to_tpm.ftt_utils as ut
import src.models.fre_to_tpm.ftt_plume_tracking as pt
import src.config.filepaths as fp
import src.data.readers.load_hrit as load_hrit

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt


def get_timestamp(myd021km_fname):
    try:
        return re.search("[0-9]{7}[.][0-9]{4}[.]", myd021km_fname).group()
    except Exception, e:
        logger.warning("Could not extract time stamp from: " + myd021km_fname + " with error: " + str(e))
        return ''


def get_modis_fname(path, timestamp_myd, myd021km_fname):
    fname = [f for f in os.listdir(path) if timestamp_myd in f]
    if len(fname) > 1:
        logger.warning("More that one frp granule matched " + myd021km_fname + "selecting 0th option")
        return fname[0]
    elif len(fname) == 1:
        return fname[0]
    else:
        return ''

def read_hdf(f):
    return SD(f, SDC.READ)

def fires_myd14(myd14_data):
    return np.where(myd14_data.select('fire mask').get() >= 7)


def main():

    # load in static data
    #frp_df = ut.read_frp_df(fp.path_to_himawari_frp)
    plume_df = ut.read_plume_polygons(fp.path_to_smoke_plume_polygons_csv)
    lc_data = []

    geo_file = fp.root_path + '/processed/himawari/Himawari_lat_lon.img'
    geostationary_lats, geostationary_lons = load_hrit.geo_read(geo_file)

    # set up arrays to hold data
    fre = []
    tpm = []
    lc = []

    # itereate over the plumes
    for i, plume in plume_df.iterrows():



        # load plume datasets
        orac_aod = []
        myd04_aod = []

        # construct plume coordinate data
        try:
            plume_bounding_box = ut.construct_bounding_box(plume)
            plume_lats, plume_lons = ut.read_modis_geo_subset(fp.path_to_modis_l1b, plume, plume_bounding_box)
            plume_points = ut.construct_points(plume, plume_bounding_box, plume_lats, plume_lons)
            plume_polygon = ut.construct_polygon(plume, plume_bounding_box, plume_lats, plume_lons)
            plume_mask = ut.construct_plume_mask(plume, plume_bounding_box)



        except Exception, e:
            print e
            continue

        print plume.filename

        # subset ORAC and MYD04, MYD14 datasets
        orac_aod_subset = []
        myd04_aod_subset = []

        timestamp_myd = get_timestamp(plume.filename)
        myd14_fname = get_modis_fname(fp.path_to_modis_frp, timestamp_myd, plume.filename)
        myd14 = read_hdf(os.path.join(fp.path_to_modis_frp, myd14_fname))
        myd14 = fires_myd14(myd14)
        fires = ut.extract_fires(fp.path_to_modis_l1b, plume, myd14)
        fires_lats, fires_lons = ut.fires_in_plume(fires, plume_polygon)

        # set up utm resampler that we use to resample all data to utm

        # reproject all modis datasets to utm (mask, orac_aod, MYD04)
        utm_resampler = ut.utm_resampler(plume_lats, plume_lons, 1000)
        utm_plume_points = ut.reproject_shapely(plume_points, utm_resampler)
        utm_plume_polygon = ut.reproject_shapely(plume_polygon, utm_resampler)
        utm_plume_mask = utm_resampler.resample_image(plume_mask, plume_lats, plume_lons)
        utm_fires = utm_resampler.resample_points_to_utm(fires_lats, fires_lons)
        utm_orac_aod_subset = []
        utm_modis_aod_subset = []

        # get FRP integration start and stop times
        start_time, stop_time = pt.find_integration_start_stop_times(plume.filename,
                                                                     utm_plume_points, utm_plume_mask,
                                                                     plume_lats, plume_lons,
                                                                     geostationary_lats, geostationary_lons,
                                                                     utm_fires,
                                                                     utm_resampler)

        # get the variables of interest
        fre.append(ut.compute_fre(plume_polygon, frp_df, start_time, stop_time))
        tpm.append(ut.compute_aod(orac_aod, myd04_aod, plume_bounding_box, plume_mask, plume_lats, plume_lons))
        lc.append(0)

    # split data based on lc type

    # compute models


if __name__=="__main__":
    main()
