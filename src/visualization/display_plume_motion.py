import logging
import os

import numpy as np
import pandas as pd
import scipy.misc as misc

import src.features.fre_to_tpm.viirs.ftt_fre as ff
import src.features.fre_to_tpm.viirs.ftt_plume_tracking as pt
import src.features.fre_to_tpm.viirs.ftt_utils as ut

import src.config.filepaths as fp
import src.config.constants as constants
import src.data.readers.load_hrit as load_hrit
import src.visualization.ftt_visualiser as vis


log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def proc_params():
    d = {}

    d['full_plume'] = True
    d['plot'] = True

    d['resampled_pix_size'] = 750  # size of UTM grid in meters
    d['frp_df'] = ut.read_frp_df(fp.path_to_himawari_frp)
    d['plume_df'] = ut.read_plume_polygons(fp.plume_polygon_path_csv)

    geo_file = os.path.join(fp.path_to_himawari_imagery, 'Himawari_lat_lon.img')
    geostationary_lats, geostationary_lons = load_hrit.geo_read(geo_file)
    d['geostationary_lats'] = geostationary_lats
    d['geostationary_lons'] = geostationary_lons

    d['output_path'] = fp.pt_vis_path
    d['df_list'] = []
    return d


def setup_sat_data(ts):

    dd = dict()
    dd['viirs_aod'] = ut.sat_data_reader(fp.path_to_viirs_aod, 'viirs', 'aod', ts)
    dd['viirs_flag'] = ut.sat_data_reader(fp.path_to_viirs_aod, 'viirs', 'flag', ts)
    dd['orac_aod'] = ut.sat_data_reader(fp.path_to_viirs_orac, 'orac', 'aod', ts)
    dd['orac_cost'] = ut.sat_data_reader(fp.path_to_viirs_orac, 'orac', 'cost', ts)

    lats, lons = ut.sat_data_reader(fp.path_to_viirs_orac, 'orac', 'geo', ts)
    dd['lats'] = lats
    dd['lons'] = lons
    return dd


def main():
    # setup the data dict to hold all data
    pp = proc_params()
    previous_timestamp = ''

    # itereate over the plumes
    for p_number, plume in pp['plume_df'].iterrows():

        if p_number != 1:
            continue

        print ''
        print p_number

        # make a directory to hold the plume logging information
        plume_logging_path = ut.create_logger_path(p_number)

        # get plume time stamp
        current_timestamp = ut.get_timestamp(plume.filename, 'viirs')

        # read in satellite data
        if current_timestamp != previous_timestamp:
            try:
                sat_data = setup_sat_data(current_timestamp)
            except Exception, e:
                logger.info('Could not load all datasets for: ' + plume.filename + '. Failed with error: ' + str(e))
                continue

            sat_data_utm = ut.resample_satellite_datasets(sat_data, pp=pp, plume=plume)
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

        # get the plume sub polygons / start stop times based on the wind speed
        try:
            utm_flow_means, geostationary_fnames, t1, t2 = pt.find_flow(p_number, plume_logging_path,
                                                                        plume_geom_utm,
                                                                        plume_geom_geo,
                                                                        pp,
                                                                        current_timestamp)
        except Exception, e:
            logger.error(str(e))
            continue





if __name__ == "__main__":
    main()