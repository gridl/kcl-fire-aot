import logging
import os

import pandas as pd
from datetime import datetime

import src.features.fre_to_tpm.viirs.ftt_utils as ut

import src.config.filepaths as fp
import src.data.readers.load_hrit as load_hrit
import src.visualization.ftt_visualiser as vis

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

# TODO make some dictionaires to hold all the data and get them out of main

'''
Algorithm description

Load in data
Iterate over each hand digitised plume
    IF current timestamp is different
        Load in all AOD data
        Construct UTM resampler for image at 750m resolution from ORAC AOD
        Build VIIRS no data mask from ORAC AOD
        Mask out all VIIRS deletions (ORAC AOD, VIIRS AOD, lats, lons, flags, costs)
    Build bounding box for hand digitsed plume (rectangular box based on (min,max) pos + padding)
    Get geographic data for plume bounding box
    Get the lat/lon for the head and tail of the plume direction vector
    Get the lats/lons for the points of the plume polygon
    Build binary plume mask for bounding box

    Build a shapely LINESTRING object for plume vector
    Build a shapely MULTIPOINT object based on plume polygon
    Build a shape POLYGON object based on plume polygon

    Build bounding box for hand digisited background extent
    Build binary mask for the background

    Subset all data (AOD, FLAG, COST) to either plume or background bounding boxes

    Construct UTM resampler for plume at 750m resolution from plume lat/lon grid
    Reproject SHAPELY objects to UTM using plume resampler

    Compute optical flow (see below for approach)
    Based on mean flow at each time step segemnt the XXX polygon

'''


def proc_params():
    d = {}

    d['full_plume'] = True
    d['plot'] = True

    d['resampled_pix_size'] = 750  # size of UTM grid in meters
    d['frp_df'] = ut.read_frp_df(fp.path_to_himawari_frp)
    #d['frp_df'] = None
    d['plume_df'] = ut.read_plume_polygons(fp.path_to_smoke_plume_polygons_viirs_csv)

    geo_file = fp.root_path + '/processed/himawari/Himawari_lat_lon.img'
    geostationary_lats, geostationary_lons = load_hrit.geo_read(geo_file)
    d['geostationary_lats'] = geostationary_lats
    d['geostationary_lons'] = geostationary_lons

    d['output_path'] = os.path.join(fp.path_to_frp_tpm_features, 'model_features_viirs_full_plumes.csv')
    d['df_list'] = []

    d['t_start_stop'] = {0: [datetime.strptime('20150706_0600', '%Y%m%d_%H%M'),
                             datetime.strptime('20150706_0510', '%Y%m%d_%H%M')],
                         1: [datetime.strptime('20150706_0600', '%Y%m%d_%H%M'),
                             datetime.strptime('20150706_0520', '%Y%m%d_%H%M')],
                         2: [datetime.strptime('20150706_0600', '%Y%m%d_%H%M'),
                             datetime.strptime('20150706_0530', '%Y%m%d_%H%M')],
                         3: [datetime.strptime('20150807_0600', '%Y%m%d_%H%M'),
                             datetime.strptime('20150807_0510', '%Y%m%d_%H%M')],
                         4: [datetime.strptime('20150807_0600', '%Y%m%d_%H%M'),
                             datetime.strptime('20150807_0510', '%Y%m%d_%H%M')],
                         5: [datetime.strptime('20150807_0600', '%Y%m%d_%H%M'),
                             datetime.strptime('20150807_0530', '%Y%m%d_%H%M')],
                         6: [datetime.strptime('20150814_0340', '%Y%m%d_%H%M'),
                             datetime.strptime('20150814_0300', '%Y%m%d_%H%M')],
                         7: [datetime.strptime('20150814_0340', '%Y%m%d_%H%M'),
                             datetime.strptime('20150814_0240', '%Y%m%d_%H%M')],
                         8: [datetime.strptime('20150902_0610', '%Y%m%d_%H%M'),
                             datetime.strptime('20150902_0520', '%Y%m%d_%H%M')],
                         9: [datetime.strptime('20150903_0410', '%Y%m%d_%H%M'),
                             datetime.strptime('20150903_0330', '%Y%m%d_%H%M')],
                         10: [datetime.strptime('20150908_0600', '%Y%m%d_%H%M'),
                             datetime.strptime('20150908_0520', '%Y%m%d_%H%M')],
                         11: [datetime.strptime('20150908_0600', '%Y%m%d_%H%M'),
                             datetime.strptime('20150908_0500', '%Y%m%d_%H%M')],
                         12: [datetime.strptime('20150909_0540', '%Y%m%d_%H%M'),
                             datetime.strptime('20150909_0500', '%Y%m%d_%H%M')],
                         13: [datetime.strptime('20150909_0540', '%Y%m%d_%H%M'),
                             datetime.strptime('20150909_0450', '%Y%m%d_%H%M')],
                         14: [datetime.strptime('20150909_0540', '%Y%m%d_%H%M'),
                             datetime.strptime('20150909_0450', '%Y%m%d_%H%M')],
                         15: [datetime.strptime('20150909_0540', '%Y%m%d_%H%M'),
                             datetime.strptime('20150909_0500', '%Y%m%d_%H%M')],
                         16: [datetime.strptime('20150909_0540', '%Y%m%d_%H%M'),
                             datetime.strptime('20150909_0500', '%Y%m%d_%H%M')],
                         17: [datetime.strptime('20150909_0540', '%Y%m%d_%H%M'),
                             datetime.strptime('20150909_0520', '%Y%m%d_%H%M')],
                         18: [datetime.strptime('20150909_0540', '%Y%m%d_%H%M'),
                             datetime.strptime('20150909_0510', '%Y%m%d_%H%M')],
                         19: [datetime.strptime('20150918_0610', '%Y%m%d_%H%M'),
                             datetime.strptime('20150918_0540', '%Y%m%d_%H%M')],
                         20: [datetime.strptime('20150918_0610', '%Y%m%d_%H%M'),
                             datetime.strptime('20150918_0540', '%Y%m%d_%H%M')],
                         21: [datetime.strptime('20150918_0610', '%Y%m%d_%H%M'),
                             datetime.strptime('20150918_0520', '%Y%m%d_%H%M')],
                         22: [datetime.strptime('20151002_0650', '%Y%m%d_%H%M'),
                             datetime.strptime('20151002_0630', '%Y%m%d_%H%M')],
                         23: [datetime.strptime('20151002_0650', '%Y%m%d_%H%M'),
                             datetime.strptime('20151002_0610', '%Y%m%d_%H%M')],
                         24: [datetime.strptime('20151003_0630', '%Y%m%d_%H%M'),
                             datetime.strptime('20151003_0540', '%Y%m%d_%H%M')],
                         25: [datetime.strptime('20151003_0630', '%Y%m%d_%H%M'),
                             datetime.strptime('20151003_0550', '%Y%m%d_%H%M')],
                         26: [datetime.strptime('20151004_0610', '%Y%m%d_%H%M'),
                             datetime.strptime('20151004_0530', '%Y%m%d_%H%M')],
                         27: [datetime.strptime('20151004_0610', '%Y%m%d_%H%M'),
                             datetime.strptime('20151004_0520', '%Y%m%d_%H%M')],
                         28: [datetime.strptime('20151004_0610', '%Y%m%d_%H%M'),
                             datetime.strptime('20151004_0550', '%Y%m%d_%H%M')],
                         29: [datetime.strptime('20151017_0340', '%Y%m%d_%H%M'),
                             datetime.strptime('20151017_0250', '%Y%m%d_%H%M')],
                         30: [datetime.strptime('20151017_0340', '%Y%m%d_%H%M'),
                             datetime.strptime('20151017_0300', '%Y%m%d_%H%M')],
                         31: [datetime.strptime('20151019_0630', '%Y%m%d_%H%M'),
                             datetime.strptime('20151019_0530', '%Y%m%d_%H%M')],
                         }

    return d


def main():
    # setup the data dict to hold all data
    pp = proc_params()
    previous_timestamp = ''
    df_list = []

    # itereate over the plumes
    for p_number, plume in pp['plume_df'].iterrows():

        # make a directory to hold the plume logging information
        plume_logging_path = ut.create_logger_path(p_number)

        # get plume time stamp
        current_timestamp = ut.get_timestamp(plume.filename)

        # read in satellite data
        if current_timestamp != previous_timestamp:
            sat_data_utm = ut.resample_satellite_datasets(plume, current_timestamp, pp)
            previous_timestamp = current_timestamp
            if sat_data_utm is None:
                continue

        # construct plume and background coordinate data
        plume_geom_geo = ut.setup_plume_data(plume, sat_data_utm)

        # subset the satellite AOD data to the plume
        plume_data_utm = ut.subset_sat_data_to_plume(sat_data_utm, plume_geom_geo)
        if pp['plot']:
            vis.plot_plume_data(sat_data_utm, plume_data_utm, plume_geom_geo['plume_bounding_box'], plume_logging_path)

        # Reproject plume shapely objects to UTM
        plume_geom_utm = ut.resample_plume_geom_to_utm(plume_geom_geo)

        # load in times for plume number
        t1, t2 = pp['t_start_stop'][p_number]

        ut.process_plume(t1, t2, pp, plume_data_utm, plume_geom_utm, plume_geom_geo, plume_logging_path,
                         p_number, df_list)

    # dump data to csv via df
    df = pd.concat(df_list)
    df.to_csv(pp['output_path'])


if __name__ == "__main__":
    main()
