import logging
import os
import pandas as pd

import src.features.fre_to_tpm.viirs.ftt_utils as ut
import src.features.fre_to_tpm.viirs.ftt_plume_tracking as pt
import src.config.filepaths as fp
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
    #d['frp_df'] = ut.read_frp_df(fp.path_to_himawari_frp)
    d['plume_df'] = ut.read_plume_polygons(fp.plume_polygon_path_csv)

    geo_file = os.path.join(fp.path_to_himawari_imagery, 'Himawari_lat_lon.img')
    geostationary_lats, geostationary_lons = load_hrit.geo_read(geo_file)
    d['geostationary_lats'] = geostationary_lats
    d['geostationary_lons'] = geostationary_lons

    d['output_path'] = fp.pt_vis_path
    d['df_output_path'] = fp.path_to_frp_tpm_models
    d['df_list'] = []
    return d


def main():
    # setup the data dict to hold all data
    pp = proc_params()
    previous_timestamp = ''
    df_list = []

    # itereate over the plumes
    for p_number, plume in pp['plume_df'].iterrows():

        #
        #if p_number not in [36]:
        #     continue

        # make a directory to hold the plume logging information
        plume_logging_path = ut.create_logger_path(p_number)

        # get plume time stamp
        current_timestamp = ut.get_timestamp(plume.filename, 'viirs')

        # read in satellite data
        if current_timestamp != previous_timestamp:
            try:
                sat_data = ut.setup_sat_data(current_timestamp)
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

        # get start stop times
        try:
            utm_flow_means, geostationary_fnames, t_start, t_stop, time_for_plume = pt.tracker(plume_logging_path,
                                                                                               plume_geom_utm,
                                                                                               plume_geom_geo,
                                                                                               pp,
                                                                                               current_timestamp)
        except Exception, e:
            logger.error(str(e))
            continue
        ut.process_plume(t_start, t_stop, time_for_plume,
                         pp, plume_data_utm, plume_geom_utm, plume_geom_geo, plume_logging_path, p_number, df_list)

    # dump data to csv via df
    df = pd.concat(df_list)
    df.to_csv(pp['df_output_path'])


if __name__ == "__main__":
    main()
