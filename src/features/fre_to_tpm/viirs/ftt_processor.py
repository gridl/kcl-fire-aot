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
import src.features.fre_to_tpm.viirs.ftt_tpm as tt
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
    #d['frp_df'] = ut.read_frp_df(fp.path_to_himawari_frp)
    d['frp_df'] = None
    d['plume_df'] = ut.read_plume_polygons(fp.path_to_smoke_plume_polygons_viirs_csv)

    geo_file = fp.root_path + '/processed/himawari/Himawari_lat_lon.img'
    geostationary_lats, geostationary_lons = load_hrit.geo_read(geo_file)
    d['geostationary_lats'] = geostationary_lats
    d['geostationary_lons'] = geostationary_lons

    d['output_path'] = os.path.join(fp.path_to_frp_tpm_features, 'model_features_viirs_full_plumes.csv')
    d['df_list'] = []
    return d


def create_logger_path(p_number):
    plume_logging_path = os.path.join(fp.path_to_plume_tracking_visualisations_viirs, str(p_number))
    if not os.path.isdir(plume_logging_path):
        os.mkdir(plume_logging_path)
    return plume_logging_path


def resample_satellite_datasets(plume, current_timestamp, pp):

    d = {}

    try:
        viirs_aod_data = ut.load_viirs(fp.path_to_viirs_aod, current_timestamp, plume.filename)
        orac_aod_data = ut.load_orac(fp.path_to_viirs_orac, current_timestamp)
        if pp['plot']:
            d['viirs_png_utm'] = misc.imread(os.path.join(fp.path_to_viirs_sdr_resampled, plume.filename))
    except Exception, e:
        logger.info('Could not load AOD data with error: ' + str(e))
        return None

    # set up resampler
    utm_rs = ut.utm_resampler(orac_aod_data.variables['lat'][:],
                                           orac_aod_data.variables['lon'][:],
                                           constants.utm_grid_size)

    # get the mask for the lats and lons and apply
    orac_aod = ut.orac_aod(orac_aod_data)
    viirs_null_mask = np.ma.getmask(orac_aod)
    masked_lats = np.ma.masked_array(utm_rs.lats, viirs_null_mask)
    masked_lons = np.ma.masked_array(utm_rs.lons, viirs_null_mask)

    # resample all the datasets to UTM
    d['viirs_aod_utm'] = utm_rs.resample_image(ut.viirs_aod(viirs_aod_data), masked_lats, masked_lons, fill_value=0)
    d['viirs_flag_utm'] = utm_rs.resample_image(ut.viirs_flags(viirs_aod_data), masked_lats, masked_lons, fill_value=0)
    d['orac_aod_utm'] = utm_rs.resample_image(orac_aod, masked_lats, masked_lons, fill_value=0)
    d['orac_cost_utm'] = utm_rs.resample_image(ut.orac_cost(orac_aod_data), masked_lats, masked_lons, fill_value=0)
    d['lats'] = utm_rs.resample_image(utm_rs.lats, masked_lats, masked_lons, fill_value=0)
    d['lons'] = utm_rs.resample_image(utm_rs.lons, masked_lats, masked_lons, fill_value=0)
    return d


def setup_plume_data(plume, ds_utm):
    d = {}
    try:
        # get plume extent geographic data (bounding box in in UTM as plume extent is UTM)
        d['plume_bounding_box'] = ut.construct_bounding_box(plume.plume_extent)
        d['plume_lats'] = ut.subset_data(ds_utm['lats'], d['plume_bounding_box'])
        d['plume_lons'] = ut.subset_data(ds_utm['lons'], d['plume_bounding_box'])

        # get plume vector geographic data
        vector_lats, vector_lons = ut.extract_subset_geo_bounds(plume.plume_vector, d['plume_bounding_box'],
                                                                d['plume_lats'], d['plume_lons'])
        # get plume polygon geographic data
        poly_lats, poly_lons = ut.extract_subset_geo_bounds(plume.plume_extent, d['plume_bounding_box'],
                                                            d['plume_lats'], d['plume_lons'])


        # get plume mask
        d['plume_mask'] = ut.construct_mask(plume.plume_extent, d['plume_bounding_box'])

        # setup shapely objects for plume geo data
        d['plume_vector'] = ut.construct_shapely_vector(vector_lats, vector_lons)
        d['plume_points'] = ut.construct_shapely_points(poly_lats, poly_lons)
        d['plume_polygon'] = ut.construct_shapely_polygon(poly_lats, poly_lons)

        d['background_bounding_box'] = ut.construct_bounding_box(plume.background_extent)
        d['background_mask'] = ut.construct_mask(plume.background_extent, d['background_bounding_box'])

        return d
    except Exception, e:
        logger.error(str(e))
        return None


def subset_sat_data_to_plume(sat_data_utm, plume_geom_geo):
    d = {}
    d['viirs_aod_utm_plume'] = ut.subset_data(sat_data_utm['viirs_aod_utm'], plume_geom_geo['plume_bounding_box'])
    d['viirs_flag_utm_plume'] = ut.subset_data(sat_data_utm['viirs_flag_utm'], plume_geom_geo['plume_bounding_box'])
    d['orac_aod_utm_plume'] = ut.subset_data(sat_data_utm['orac_aod_utm'], plume_geom_geo['plume_bounding_box'])
    d['orac_cost_utm_plume'] = ut.subset_data(sat_data_utm['orac_cost_utm'], plume_geom_geo['plume_bounding_box'])

    d['viirs_aod_utm_background'] = ut.subset_data(sat_data_utm['viirs_aod_utm'],
                                                   plume_geom_geo['background_bounding_box'])
    d['viirs_flag_utm_background'] = ut.subset_data(sat_data_utm['viirs_flag_utm'],
                                                    plume_geom_geo['background_bounding_box'])
    d['orac_aod_utm_background'] = ut.subset_data(sat_data_utm['orac_aod_utm'],
                                                  plume_geom_geo['background_bounding_box'])
    d['orac_cost_utm_background'] = ut.subset_data(sat_data_utm['orac_cost_utm'],
                                                   plume_geom_geo['background_bounding_box'])
    return d


def resample_plume_geom_to_utm(plume_geom_geo):
    d = {}
    d['utm_resampler_plume'] = ut.utm_resampler(plume_geom_geo['plume_lats'],
                                                plume_geom_geo['plume_lons'],
                                                constants.utm_grid_size)
    d['utm_plume_points'] = ut.reproject_shapely(plume_geom_geo['plume_points'], d['utm_resampler_plume'])
    d['utm_plume_polygon'] = ut.reproject_shapely(plume_geom_geo['plume_polygon'], d['utm_resampler_plume'])
    d['utm_plume_vector'] = ut.reproject_shapely(plume_geom_geo['plume_vector'], d['utm_resampler_plume'])
    return d


def process_plume_subsets(utm_flow_means, geostationary_fnames, plume_logging_path, plume_geom_geo,
                          plume, plume_geom_utm, pp, plume_data_utm, p_number, current_timestamp,
                          df_list):

    # the flow is computed back in time from the most recent plume extent to the oldest.
    # We need to work out how much of the oldest plume extent is attributable to the
    # most recent part.  To do that, we use the flow speed from the oldest plume extent
    # first, as this gives us the part we are looking for.  Then work back up through time.
    utm_flow_means = utm_flow_means[::-1]
    geostationary_fnames = geostationary_fnames[::-1]

    # now using the flow informatino get the sub polygons on the plume. Each subpolygon
    # contains the pixel positions that correspond to each himawari timestamp.
    plume_sub_polygons = ut.split_plume_polgons(utm_flow_means, plume_logging_path, plume,
                                                plume_geom_geo, plume_geom_utm, pp)


    # get the variables of interest
    if plume_sub_polygons:

        for sub_p_number, sub_polygon in plume_sub_polygons.iteritems():

            sub_plume_logging_path = os.path.join(plume_logging_path, str(sub_p_number))
            if not os.path.isdir(sub_plume_logging_path):
                os.mkdir(sub_plume_logging_path)

            # make mask for sub polygon
            sub_plume_mask = ut.sub_mask(plume_geom_geo['plume_lats'].shape,
                                         sub_polygon,
                                         plume_geom_geo['plume_mask'])

            # make polygon for sub_polygon and intersect with plume polygon
            bounding_lats, bounding_lons = ut.extract_geo_bounds(sub_polygon,
                                                                 plume_geom_geo['plume_lats'],
                                                                 plume_geom_geo['plume_lons'])
            sub_plume_polygon = ut.construct_shapely_polygon(bounding_lats, bounding_lons)
            utm_sub_plume_polygon = ut.reproject_shapely(sub_plume_polygon, plume_geom_utm['utm_resampler_plume'])

            # get intersection of plume and sub_plume polygons.  The reason for this is that
            # the plume polygon has the shape of the plume, whilst the sub plume polygon has
            # the shape of the bounding box (i.e. rectangular).  By taking the intersection
            # we get the segment from the both the plume and the sub part of the boudning box.

            try:
                utm_sub_plume_polygon = utm_sub_plume_polygon.intersection(plume_geom_utm['utm_plume_polygon'])
            except Exception, e:
                logger.error(str(e))
                continue

            # get background aod for sub plume
            bg_aod_dict = tt.extract_bg_aod(plume_data_utm, plume_geom_geo['background_mask'])

            # compute TPM
            out_dict = tt.compute_tpm_subset(plume_data_utm,
                                      utm_sub_plume_polygon, sub_plume_mask, bg_aod_dict,
                                      sub_plume_logging_path, pp)

            out_dict['main_plume_number'] = p_number
            out_dict['sub_plume_number'] = sub_p_number
            out_dict['viirs_time'] = current_timestamp

            # compute FRE
            ff.compute_fre_subset(out_dict, geostationary_fnames[sub_p_number],
                                  plume_geom_utm, pp['frp_df'], sub_plume_logging_path)

            # convert datadict to dataframe and add to list
            df_list.append(pd.DataFrame(out_dict, index=['i', ]))


def process_plume_full(t1, t2, pp, plume_data_utm, plume_geom_utm, plume_geom_geo, plume_logging_path, p_number,
                       df_list):

    # get background aod for sub plume
    bg_aod_dict = tt.extract_bg_aod(plume_data_utm, plume_geom_geo['background_mask'])

    # compute tpm
    out_dict = tt.compute_tpm_full(plume_data_utm, plume_geom_utm, plume_geom_geo, bg_aod_dict, plume_logging_path)
    out_dict['plume_number'] = p_number

    # compute fre
    ff.compute_fre_full_plume(t1, t2, pp['frp_df'], plume_geom_utm, plume_logging_path, out_dict)

    # convert datadict to dataframe and add to list
    df_list.append(pd.DataFrame(out_dict, index=['i', ]))



def main():
    # setup the data dict to hold all data
    pp = proc_params()
    previous_timestamp = ''
    df_list = []

    # itereate over the plumes
    for p_number, plume in pp['plume_df'].iterrows():

        # make a directory to hold the plume logging information
        plume_logging_path = create_logger_path(p_number)

        # get plume time stamp
        current_timestamp = ut.get_timestamp(plume.filename)

        # read in satellite data
        if current_timestamp != previous_timestamp:
            sat_data_utm = resample_satellite_datasets(plume, current_timestamp, pp)
            previous_timestamp = current_timestamp
            if sat_data_utm is None:
                continue

        # construct plume and background coordinate data
        plume_geom_geo = setup_plume_data(plume, sat_data_utm)

        # subset the satellite AOD data to the plume
        plume_data_utm = subset_sat_data_to_plume(sat_data_utm, plume_geom_geo)
        if pp['plot']:
            vis.plot_plume_data(sat_data_utm, plume_data_utm, plume_geom_geo['plume_bounding_box'], plume_logging_path)

        # Reproject plume shapely objects to UTM
        plume_geom_utm = resample_plume_geom_to_utm(plume_geom_geo)

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

        # now one of two processing options full plume or subsets
        if pp['full_plume']:
            process_plume_full(t1, t2, pp, plume_data_utm, plume_geom_utm, plume_geom_geo, plume_logging_path,
                               p_number, df_list)
        else:
            process_plume_subsets(utm_flow_means, geostationary_fnames, plume_logging_path, plume_geom_geo,
                          plume, plume_geom_utm, pp, plume_data_utm, p_number, current_timestamp,
                          df_list)

    # dump data to csv via df
    df = pd.concat(df_list)
    df.to_csv(pp['output_path'])


if __name__ == "__main__":
    main()
