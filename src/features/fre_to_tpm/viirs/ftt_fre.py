#!/home/users/dnfisher/soft/virtual_envs/kcl-fire-aot/bin/python2

import logging
import os

import numpy as np
from scipy import integrate
from datetime import datetime


log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def temporal_subset(frp_df, start_time, stop_time):
    """

    :param frp_df: The FRP containing dataframe
    :param start_time: The subset start time
    :param stop_time: The subset stop time
    :return: the subsetted dataframe
    """
    return frp_df.loc[(frp_df['obs_time'] <= stop_time) & (frp_df['obs_time'] >= start_time)]



def temporal_subset_single_time(frp_df, t):
    """

    :param frp_df: The FRP containing dataframe
    :param start_time: The subset time of interest
    :return: the subsetted dataframe
    """
    return frp_df.loc[(frp_df['obs_time'] == t)]



def temporal_subset_single_day(frp_df, t):
    """

    :param frp_df: The FRP containing dataframe
    :param t: The subset time of interest
    :return: the subsetted dataframe
    """
    return frp_df.loc[(frp_df['obs_time'].dt.year == t.year) & (frp_df['obs_time'].dt.day == t.day)]


def spatial_subset(frp_subset, plume_geom_geo):
    """

    :param frp_subset: The temporally subsetted dataframe
    :param plume_polygon: The smoke plume polygon
    :param utm_resampler: The utm resampler object
    :return: The spatially subsetted frp dataframe
    """
    # import src.features.fre_to_tpm.viirs.ftt_utils as ut
    inbounds = []
    for i, (index, frp_pixel) in enumerate(frp_subset.iterrows()):

        # # transform FRP pixel into UTM coordinates
        # projected_fire = ut.reproject_shapely(frp_pixel['point'], plume_geom_utm['utm_resampler_plume'])
        #
        # # get distance between fire head of plume vector
        # fire_coords = np.array(projected_fire.coords[0])
        # head_coords = np.array(plume_geom_utm['utm_plume_vector'].coords[1])
        # dist = np.linalg.norm(fire_coords-head_coords)
        # if dist < max_dist:
        #     inbounds.append(i)

        if frp_pixel['point'].within(plume_geom_geo['plume_polygon']):
            inbounds.append(i)

    return frp_subset.iloc[inbounds]



def group_subset(frp_subset):
    """
    For each time step we need to sum the FRP.  We can do this using
    the groupby functionality of pandas.

    :param frp_subset: the spatially and temporally subsette dataframe
    :return: the grouped dataframe
    """

    frp_subset['FIRE_CONFIDENCE_mean'] = frp_subset['FIRE_CONFIDENCE'].copy()
    frp_subset['FIRE_CONFIDENCE_std'] = frp_subset['FIRE_CONFIDENCE'].copy()
    agg_dict = {'FRP_0': np.sum, 'FIRE_CONFIDENCE_mean': np.mean, 'FIRE_CONFIDENCE_std': np.std}
    grouped = frp_subset.groupby('obs_time').agg(agg_dict)
    return grouped


def integrate_frp(frp_subset):
    """
    Integrate from t0 to final observation.

    :param frp_subset: The spatial and temporal subset of FRPs
    :return: The integrated FRP values, i.e. the FRE
    """
    t0 = frp_subset.index[0]
    sample_times = (frp_subset.index - t0).total_seconds()

    # now integrate
    return integrate.trapz(frp_subset['FRP_0'], sample_times)


def fire_locations_for_plume_roi(plume_geom_geo, frp_df, t):
    try:
        frp_subset = temporal_subset_single_time(frp_df, t)
        frp_subset = spatial_subset(frp_subset, plume_geom_geo)
        # returns a set of geogrpahic coordinate
        return frp_subset.point.values

    except Exception, e:
        logger.error('FRE calculation failed with error' + str(e))
        return None


def fire_locations_for_digitisation(frp_df, t):
    try:
        frp_subset = temporal_subset_single_day(frp_df, t)
        return frp_subset

    except Exception, e:
        logger.error('FRE calculation failed with error' + str(e))
        return None


def compute_fre_subset(out_dict, geostationary_fname, plume_geom_geo, frp_df, sub_plume_logging_path):

    """
    :param plume_polygon: The smoke plume polygon
    :param frp_df: The FRP containing dataframe
    :param start_time: The integration start time
    :param stop_time: The integratinos stop time
    :return:  The FRE
    """

    t = datetime.strptime(geostationary_fname.split('/')[-1][7:20], '%Y%m%d_%H%M')

    try:
        frp_subset = temporal_subset_single_time(frp_df, t)
        frp_subset = spatial_subset(frp_subset, plume_geom_geo)
        frp_subset.to_csv(os.path.join(sub_plume_logging_path, 'fires.csv'))

        grouped_frp_subset = group_subset(frp_subset)
        grouped_frp_subset.to_csv(os.path.join(sub_plume_logging_path, 'fires_grouped.csv'))

        # integrate to get the fre as we are only doing one timestamp
        # assume that the fires is burning the same for the next ten
        # minutes
        fre = grouped_frp_subset['FRP_0'].values[0] * (10 * 60)  #  assume 600 seconds

        out_dict['fre'] = fre
        out_dict['mean_fire_confience'] = grouped_frp_subset['FIRE_CONFIDENCE_mean']
        out_dict['std_fire_confience'] = grouped_frp_subset['FIRE_CONFIDENCE_std']
        out_dict['himawari_file'] = geostationary_fname
        out_dict['himawari_time'] = t

    except Exception, e:
        logger.error('FRE calculation failed with error' + str(e))

        out_dict['fre'] = np.nan
        out_dict['mean_fire_confience'] = np.nan
        out_dict['std_fire_confience'] = np.nan
        out_dict['himawari_file'] = geostationary_fname
        out_dict['himawari_time'] = t


def compute_fre_full_plume(t_start, t_stop, time_for_plume,
                           frp_df, plume_geom_geo, plume_logging_path, out_dict):


    try:
        frp_subset = temporal_subset(frp_df, t_stop, t_start)
        frp_subset = spatial_subset(frp_subset, plume_geom_geo)

        # sort df by fire quality then drop duplicates and keep the first
        print frp_subset.shape
        frp_subset.sort_values('FIRE_CONFIDENCE', ascending=False, inplace=True)
        frp_subset.drop_duplicates(['ABS_line', 'ABS_samp', 'obs_time'], inplace=True, keep='first')
        print frp_subset.shape

        frp_subset.to_csv(os.path.join(plume_logging_path, 'fires.csv'))

        grouped_frp_subset = group_subset(frp_subset)
        grouped_frp_subset.to_csv(os.path.join(plume_logging_path, 'fires_grouped.csv'))

        # integrate to get the fre as we are only doing one timestamp
        # assume that the fires is burning the same for the next ten
        # minutes
        fre = integrate_frp(grouped_frp_subset)

        # lets set up an alternative FRE using mean and multiplying by time in seconds for plume
        # (should be more accurate).
        # alt_fre = grouped_frp_subset['FRP_0'].mean() * (grouped_frp_subset.index[-1] -
        #                                                 grouped_frp_subset.index[0]).total_seconds()
        alt_fre = grouped_frp_subset['FRP_0'].mean() * time_for_plume

        out_dict['fre'] = fre
        out_dict['alt_fre'] = alt_fre
        out_dict['mean_fire_confience'] = np.mean(grouped_frp_subset['FIRE_CONFIDENCE_mean'])
        out_dict['std_fire_confience'] = np.mean(grouped_frp_subset['FIRE_CONFIDENCE_std'])

    except Exception, e:
        logger.error('FRE calculation failed with error' + str(e))

        out_dict['fre'] = np.nan
        out_dict['mean_fire_confience'] = np.nan
        out_dict['std_fire_confience'] = np.nan



