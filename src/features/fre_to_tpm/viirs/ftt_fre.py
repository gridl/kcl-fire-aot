import logging
import os

import numpy as np
from scipy import integrate

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
    try:
        return frp_df.loc[(frp_df['obs_time'] <= stop_time) & (frp_df['obs_time'] >= start_time)]
    except Exception, e:
        logger.error('Could not extract time subset, failed with error: ' + str(e))
        return None


def temporal_subset_single_time(frp_df, t):
    """

    :param frp_df: The FRP containing dataframe
    :param start_time: The subset time of interest
    :return: the subsetted dataframe
    """
    try:
        return frp_df.loc[(frp_df['obs_time'] == t)]
    except Exception, e:
        logger.error('Could not extract time subset, failed with error: ' + str(e))
        return None


def temporal_subset_single_day(frp_df, t):
    """

    :param frp_df: The FRP containing dataframe
    :param t: The subset time of interest
    :return: the subsetted dataframe
    """

    try:
        return frp_df.loc[(frp_df['obs_time'].dt.year == t.year) & (frp_df['obs_time'].dt.day == t.day)]
    except Exception, e:
        logger.error('Could not extract time subset, failed with error: ' + str(e))
        return None


def spatial_subset(frp_subset, plume_polygon, utm_resampler):
    """

    :param frp_subset: The temporally subsetted dataframe
    :param plume_polygon: The smoke plume polygon
    :param utm_resampler: The utm resampler object
    :return: The spatially subsetted frp dataframe
    """
    inbounds = []
    try:
        for i, (index, frp_pixel) in enumerate(frp_subset.iterrows()):

            # transform FRP pixel into UTM coordinates
            projected_fire = ut.reproject_shapely(frp_pixel['point'], utm_resampler)
            if projected_fire.within(plume_polygon):
                inbounds.append(i)

        if inbounds:
            return frp_subset.iloc[inbounds]
    except Exception, e:
        print 'Could not extract spatial subset, failed with error:', str(e)
        return None


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
    try:
        t0 = frp_subset.index[0]
        sample_times = (frp_subset.index - t0).total_seconds()
    except Exception, e:
        logger.error('Could not convert to time since t0. Failed with error: ' + str(e))
        return None

    # now integrate
    return integrate.trapz(frp_subset['FRP_0'], sample_times)


def compute_fre(p_number, plume_logging_path,
                plume_polygon, frp_df, start_time, stop_time, utm_resampler):
    """

    :param plume_polygon: The smoke plume polygon
    :param frp_df: The FRP containing dataframe
    :param start_time: The integration start time
    :param stop_time: The integratinos stop time
    :return:  The FRE
    """
    try:
        frp_subset = temporal_subset(frp_df, start_time, stop_time)
        frp_subset = spatial_subset(frp_subset, plume_polygon, utm_resampler)
        frp_subset.to_csv(os.path.join(plume_logging_path, str(p_number) + '_fires.csv'))

        grouped_frp_subset = group_subset(frp_subset)
        grouped_frp_subset.to_csv(os.path.join(plume_logging_path, str(p_number) + '_fires_grouped.csv'))

    except Exception, e:
        logger.error('FRE calculation failed with error' + str(e))
        return None

    # integrate to get the fre
    fre = integrate_frp(grouped_frp_subset)
    return fre


def fire_locations_for_plume_roi(plume_polygon, utm_resampler, frp_df, t):
    try:
        frp_subset = temporal_subset_single_time(frp_df, t)
        frp_subset = spatial_subset(frp_subset, plume_polygon, utm_resampler)
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

