import logging
import os

import numpy as np
from scipy import integrate
from datetime import datetime

import src.features.fre_to_tpm.viirs.ftt_utils as ut

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



def spatial_subset(frp_subset, utm_resampler, utm_plume_vector):
    """

    :param frp_subset: The temporally subsetted dataframe
    :param plume_polygon: The smoke plume polygon
    :param utm_resampler: The utm resampler object
    :return: The spatially subsetted frp dataframe
    """
    inbounds = []
    max_dist = 10000  # TODO ADD TO CONFIG
    for i, (index, frp_pixel) in enumerate(frp_subset.iterrows()):

        # transform FRP pixel into UTM coordinates
        projected_fire = ut.reproject_shapely(frp_pixel['point'], utm_resampler)

        # get distance between fire head of plume vector
        fire_coords = np.array(projected_fire.coords[0])
        head_coords = np.array(utm_plume_vector.coords[1])
        dist = np.linalg.norm(fire_coords-head_coords)
        if dist < max_dist:
            inbounds.append(i)

    return frp_subset.iloc[inbounds]



def group_subset(frp_subset):
    """
    For each time step we need to sum the FRP.  We can do this using
    the groupby functionality of pandas.

    :param frp_subset: the spatially and temporally subsette dataframe
    :return: the grouped dataframe
    """

    frp_subset['FIRE_CONFIDENCE_mean'] = frp_subset['FIRE_CONFIDENCE']
    frp_subset['FIRE_CONFIDENCE_std'] = frp_subset['FIRE_CONFIDENCE']
    frp_subset = frp_subset.groupby('obs_time').agg({'FRP_0': np.sum,
                                                     'FIRE_CONFIDENCE_mean': np.mean,
                                                     'FIRE_CONFIDENCE_std': np.std})[['FRP_0',
                                                                                      'FIRE_CONFIDENCE_mean',
                                                                                      'FIRE_CONFIDENCE_std']]
    return frp_subset


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


def fire_locations_for_plume_roi(plume_polygon, utm_resampler, frp_df, t, utm_plume_vector):
    try:
        frp_subset = temporal_subset_single_time(frp_df, t)
        frp_subset = spatial_subset(frp_subset, utm_resampler, utm_plume_vector)
        return frp_subset.point.values

    except Exception, e:
        logger.error('FRE calculation failed with error' + str(e))
        return None


def fire_locations_for_digitisation(frp_df, t):
    try:
        frp_subset = temporal_subset_single_day(frp_df, t)
        return frp_subset.point.values

    except Exception, e:
        logger.error('FRE calculation failed with error' + str(e))
        return None


def compute_fre(out_dict, geostationary_fname,
                utm_plume_polygon, utm_plume_vector,
                frp_df, utm_resampler_plume, sub_plume_logging_path):
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
        frp_subset = spatial_subset(frp_subset, utm_resampler_plume, utm_plume_vector)
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




