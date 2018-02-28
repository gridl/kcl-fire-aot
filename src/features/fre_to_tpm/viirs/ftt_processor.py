import logging
import os

import numpy as np
import pandas as pd
import scipy.misc as misc

import src.features.fre_to_tpm.viirs.ftt_fre as ff
import src.features.fre_to_tpm.viirs.ftt_plume_tracking as pt
import src.features.fre_to_tpm.viirs.ftt_utils as ut

import src.config.filepaths as fp
import src.data.readers.load_hrit as load_hrit
import src.features.fre_to_tpm.viirs.ftt_tpm as tt

import matplotlib.pyplot as plt
import matplotlib.cm as cm

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

# TODO make some dictionaires to hold all the data and get them out of main

'''
Still Todo:
3. Get all fires near to the head of the plume (Say within some radius i.e. 10km).

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


def main():

    plot = True

    # TODO add to config (it is pixe size that resample to)
    resampled_pix_size = 750

    # load in static data
    frp_df = ut.read_frp_df(fp.path_to_himawari_frp)
    #frp_df = None

    plume_df = ut.read_plume_polygons(fp.path_to_smoke_plume_polygons_viirs_csv)
    geo_file = fp.root_path + '/processed/himawari/Himawari_lat_lon.img'
    geostationary_lats, geostationary_lons = load_hrit.geo_read(geo_file)

    # setup output path to hold csv
    output_path = os.path.join(fp.path_to_frp_tpm_features, 'model_features_viirs_OAODlt1.csv')

    # set timestamp to check if new data loaded in
    previous_timestamp = ''

    # list to hold individual dataframes that will be concatenated at the end
    df_list = []

    # itereate over the plumes
    for p_number, plume in plume_df.iterrows():

        if p_number != 21: continue

        # make a directory to hold the plume logging information
        plume_logging_path = os.path.join(fp.path_to_plume_tracking_visualisations_viirs, str(p_number))
        if not os.path.isdir(plume_logging_path):
            os.mkdir(plume_logging_path)

        # get plume time stamp
        current_timestamp = ut.get_timestamp(plume.filename)

        # if working on a new scene. Then set it up by applying no
        # data masks and resampling everything to UTM grid at 750m
        if current_timestamp != previous_timestamp:

            try:
                viirs_png_utm = misc.imread(os.path.join(fp.path_to_viirs_sdr_resampled, plume.filename))
                viirs_aod_data = ut.load_viirs(fp.path_to_viirs_aod, current_timestamp, plume.filename)
                orac_aod_data = ut.load_orac(fp.path_to_viirs_orac, current_timestamp)
            except Exception, e:
                logger.info('Could not load AOD data with error: ' + str(e))
                continue

            # set up resampler
            utm_image_resampler = ut.utm_resampler(orac_aod_data.variables['lat'][:],
                                             orac_aod_data.variables['lon'][:],
                                             resampled_pix_size)

            # get the mask for the lats and lons and apply
            orac_aod = ut.orac_aod(orac_aod_data)
            mask = np.ma.getmask(orac_aod)
            masked_lats = np.ma.masked_array(utm_image_resampler.lats, mask)
            masked_lons = np.ma.masked_array(utm_image_resampler.lons, mask)

            # resample all the datasets to UTM
            viirs_aod_utm = utm_image_resampler.resample_image(ut.viirs_aod(viirs_aod_data),
                                                         masked_lats, masked_lons, fill_value=0)
            viirs_flag_utm = utm_image_resampler.resample_image(ut.viirs_flags(viirs_aod_data),
                                                          masked_lats, masked_lons, fill_value=0)
            orac_aod_utm = utm_image_resampler.resample_image(orac_aod,
                                                        masked_lats, masked_lons, fill_value=0)
            orac_cost_utm = utm_image_resampler.resample_image(ut.orac_cost(orac_aod_data),
                                                         masked_lats, masked_lons, fill_value=0)
            lats = utm_image_resampler.resample_image(utm_image_resampler.lats, masked_lats, masked_lons, fill_value=0)
            lons = utm_image_resampler.resample_image(utm_image_resampler.lons,masked_lats, masked_lons, fill_value=0)

            previous_timestamp = current_timestamp


        # construct plume and background coordinate data
        try:
            plume_bounding_box = ut.construct_bounding_box(plume.plume_extent)
            plume_lats = ut.subset_data(lats, plume_bounding_box)
            plume_lons = ut.subset_data(lons, plume_bounding_box)

            vector_lats, vector_lons = ut.extract_subset_geo_bounds(plume.plume_vector, plume_bounding_box,
                                                                    plume_lats, plume_lons)
            poly_lats, poly_lons = ut.extract_subset_geo_bounds(plume.plume_extent, plume_bounding_box,
                                                                        plume_lats, plume_lons)
            plume_mask = ut.construct_mask(plume.plume_extent, plume_bounding_box)

            plume_vector = ut.construct_shapely_vector(vector_lats, vector_lons)
            plume_points = ut.construct_shapely_points(poly_lats, poly_lons)
            plume_polygon = ut.construct_shapely_polygon(poly_lats, poly_lons)

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

        viirs_aod_utm_background = ut.subset_data(viirs_aod_utm, background_bounding_box)
        viirs_flag_utm_background = ut.subset_data(viirs_flag_utm, background_bounding_box)
        orac_aod_utm_background = ut.subset_data(orac_aod_utm, background_bounding_box)
        orac_cost_utm_background = ut.subset_data(orac_cost_utm, background_bounding_box)

        # Reproject Shapely objects to UTM using plume geographic info
        utm_resampler_plume = ut.utm_resampler(plume_lats, plume_lons, resampled_pix_size)
        utm_plume_points = ut.reproject_shapely(plume_points, utm_resampler_plume)
        utm_plume_polygon = ut.reproject_shapely(plume_polygon, utm_resampler_plume)
        utm_plume_vector = ut.reproject_shapely(plume_vector, utm_resampler_plume)

        if plot:
            plt.imshow(viirs_png_utm[plume_bounding_box['min_y']:plume_bounding_box['max_y'],
                       plume_bounding_box['min_x']:plume_bounding_box['max_x'], :])
            plt.savefig(os.path.join(plume_logging_path, 'viirs_plume_tcc.png'), bbox_inches='tight')
            plt.close()
            plt.imshow(viirs_aod_utm_plume, vmin=0, vmax=2)
            cb = plt.colorbar()
            cb.set_label('VIIRS IP AOD')
            plt.savefig(os.path.join(plume_logging_path, 'viirs_plume_aod.png'), bbox_inches='tight')
            plt.close()

            ax = plt.imshow(viirs_flag_utm_plume)
            cmap = cm.get_cmap('Set1', 4)
            ax.set_cmap(cmap)
            cb = plt.colorbar()
            cb.set_label('VIIRS IP AOD Flag')
            plt.savefig(os.path.join(plume_logging_path, 'viirs_plume_flag.png'), bbox_inches='tight')
            plt.close()

            plt.imshow(orac_aod_utm_plume, vmin=0, vmax=2)
            cb = plt.colorbar()
            cb.set_label('VIIRS ORAC AOD')
            plt.savefig(os.path.join(plume_logging_path, 'viirs_plume_orac.png'), bbox_inches='tight')
            plt.close()

            plt.imshow(orac_cost_utm_plume, vmax=10, cmap='plasma')
            cb = plt.colorbar()
            cb.set_label('VIIRS ORAC AOD COST')
            plt.savefig(os.path.join(plume_logging_path, 'viirs_plume_orac_cost.png'), bbox_inches='tight')
            plt.close()

        # get the plume sub polygons based on the wind speed
        try:
            utm_flow_means, geostationary_fnames = pt.find_flow(p_number, plume_logging_path, utm_plume_points,
                                                                utm_plume_vector, plume_lats, plume_lons,
                                                                geostationary_lats, geostationary_lons,
                                                                current_timestamp, utm_resampler_plume,
                                                                frp_df, resampled_pix_size, plot=plot)

            # the flow is computed back in time from the most recent plume extent to the oldest.
            # We need to work out how much of the oldest plume extent is attributable to the
            # most recent part.  To do that, we use the flow speed from the oldest plume extent
            # first, as this gives us the part we are looking for.  Then work back up through time.
            utm_flow_means = utm_flow_means[::-1]
            geostationary_fnames = geostationary_fnames[::-1]

            # now using the flow informatino get the sub polygons on the plume. Each subpolygon
            # contains the pixel positions that correspond to each himawari timestamp.
            plume_sub_polygons = ut.split_plume_polgons(plume_logging_path, plume_bounding_box,
                                                        plume.plume_vector, utm_plume_vector,
                                                        utm_flow_means,
                                                        utm_resampler_plume,
                                                        plume_lats, plume_lons,
                                                        plume_mask,
                                                        plot=plot)

        except Exception, e:
            logger.error(str(e))
            continue

        # get the variables of interest
        if plume_sub_polygons:

            for sub_p_number, sub_polygon in plume_sub_polygons.iteritems():

                sub_plume_logging_path = os.path.join(plume_logging_path, str(sub_p_number))
                if not os.path.isdir(sub_plume_logging_path):
                    os.mkdir(sub_plume_logging_path)

                # make mask for sub polygon
                sub_plume_mask = ut.sub_mask(plume_lats.shape, sub_polygon, plume_mask)

                # make polygon for sub_polygon and intersect with plume polygon
                bounding_lats, bounding_lons = ut.extract_geo_bounds(sub_polygon, plume_lats, plume_lons)
                sub_plume_polygon = ut.construct_shapely_polygon(bounding_lats, bounding_lons)
                utm_sub_plume_polygon = ut.reproject_shapely(sub_plume_polygon, utm_resampler_plume)

                # get intersection of plume and sub_plume polygons.  The reason for this is that
                # the plume polygon has the shape of the plume, whilst the sub plume polygon has
                # the shape of the bounding box (i.e. rectangular).  By taking the intersection
                # we get the segment from the both the plume and the sub part of the boudning box.

                try:
                    utm_sub_plume_polygon = utm_sub_plume_polygon.intersection(utm_plume_polygon)
                except Exception, e:
                    logger.error(str(e))
                    continue

                # get background aod for sub plume
                bg_aod_dict = tt.extract_bg_aod(viirs_aod_utm_background, viirs_flag_utm_background,
                                                orac_aod_utm_background, orac_cost_utm_background,
                                                background_mask)

                # compute TPM
                out_dict = tt.compute_tpm(viirs_aod_utm_plume, viirs_flag_utm_plume,
                                          orac_aod_utm_plume, orac_cost_utm_plume,
                                          utm_sub_plume_polygon, sub_plume_mask, bg_aod_dict,
                                          sub_plume_logging_path, plot=plot)

                out_dict['main_plume_number'] = p_number
                out_dict['sub_plume_number'] = sub_p_number
                out_dict['viirs_time'] = current_timestamp

                # compute FRE
                ff.compute_fre(out_dict, geostationary_fnames[sub_p_number],
                               utm_plume_polygon, frp_df, utm_resampler_plume, sub_plume_logging_path)

                # convert datadict to dataframe and add to list
                df_list.append(pd.DataFrame(out_dict, index=['i',]))

    # dump data to csv via df
    df = pd.concat(df_list)
    df.to_csv(output_path)


if __name__=="__main__":
    main()
