'''
Contains the various functions and classes that are used in the
ftt (fre-to_tpm) processor.  These can be broken down as follows:
'''


# load in required packages
import ast
import glob
import os
import re
from datetime import datetime, timedelta
import logging

import pandas as pd
import numpy as np
from netCDF4 import Dataset
from pyhdf.SD import SD, SDC
from matplotlib.path import Path
from scipy import stats
from scipy import integrate
from scipy import ndimage
from shapely.geometry import Polygon, Point
from shapely.ops import transform
from mpl_toolkits.basemap import Basemap
import pyresample as pr
import pyproj


log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


#########################    SETUP UTILS    #########################


def read_plume_polygons(path):
    try:
        df = pd.read_pickle(path)
    except Exception, e:
        logger.warning('Could not load pickle with error:' + str(e) + ' ...attempting to load csv')
        df = pd.read_csv(path, quotechar='"', sep=',', converters={'plume_extent': ast.literal_eval})
    return df


#########################    FRE UTILS    #########################


def _build_frp_df(path):
    '''

    :param path: path to the frp csv files and dataframe
    :return: dataframe holding frp
    '''
    frp_csv_files = glob.glob(path + '*.csv')
    df_from_each_file = (pd.read_csv(f) for f in frp_csv_files)
    frp_df = pd.concat(df_from_each_file, ignore_index=True)

    # keep only columns on interest
    frp_df = frp_df[['FIRE_CONFIDENCE', 'FRP_0', 'LATITUDE', 'LONGITUDE', 'year', 'month', 'day', 'time']]

    # make geocoords into shapely points
    points = [Point(p[0], p[1]) for p in zip(frp_df['LONGITUDE'].values, frp_df['LATITUDE'].values)]
    frp_df['point'] = points

    # reindex onto date
    for k in ['year', 'month', 'day', 'time']:
        frp_df[k] = frp_df[k].astype(int).astype(str)
        if k == 'time':
            frp_df[k] = frp_df[k].str.zfill(4)
        if k in ['month', 'day']:
            frp_df[k] = frp_df[k].str.zfill(2)

    format = '%Y%m%d%H%M'
    frp_df['obs_time'] = pd.to_datetime(frp_df['year'] +
                                        frp_df['month'] +
                                        frp_df['day'] +
                                        frp_df['time'], format=format)
    frp_df['obs_date'] = frp_df['obs_time'].dt.date

    # drop columns we dont needs
    frp_df.drop(['LONGITUDE', 'LATITUDE'], axis=1, inplace=True)
    frp_df.drop(['year', 'month', 'day', 'time'], axis=1, inplace=True)

    frp_df.to_pickle(path + 'frp_df.p')

    return frp_df


def load_frp_df(path):
    '''

    :param path: path to frp csv files and dataframe
    :return: the frp holding dataframe
    '''
    try:
        df_path = glob.glob(path + 'frp_df.p')
        frp_df = pd.read_pickle(df_path)
    except Exception, e:
        print('could not load frp dataframe, failed with error ' + str(e) + ' building anew')
        frp_df = _build_frp_df(path)
    return frp_df


def build_polygon(plume, lats, lons):
    '''

    :param plume: plume polygon points
    :param lats:
    :param lons:
    :return: polygon defining the plume
    '''

    # when digitising points are appended (x,y).  However, arrays are accessed
    # in numpy as row, col which is y, x.  So we need to switch
    bounding_lats = [lats[point[1], point[0]] for point in plume.plume_extent]
    bounding_lons = [lons[point[1], point[0]] for point in plume.plume_extent]

    return Polygon(zip(bounding_lons, bounding_lats))


def read_geo(path, plume):
    # TODO replace with ORAC data coords, so not using MYD stuff as not required
    myd = SD(os.path.join(path, plume.filename), SDC.READ)
    lats = ndimage.zoom(myd.select('Latitude').get(), 5)
    lons = ndimage.zoom(myd.select('Longitude').get(), 5)
    return lats, lons


def compute_fre(frp_subset):
    try:
        t0 = frp_subset.index[0]
        sample_times = (frp_subset.index - t0).total_seconds()
    except Exception, e:
        print 'Could not extract spatial subset, failed with error:', str(e)
        return None

    # now integrate
    return integrate.trapz(frp_subset['FRP_0'], sample_times)



def compute_fre(path, plume, frp_df):
    # load in lats and lons for plume polygon
    lats, lons = read_geo(path, plume)

    # calculate integration times
    start_time, stop_time = []  # TODO implement functions to find integration times

    # subset df by time
    try:
        frp_subset = frp_df.loc[(frp_df['obs_date'] == stop_time) |
                                (frp_df['obs_date'] == start_time)]
    except Exception, e:
        print 'Could not extract time subset, failed with error:', str(e)
        return None

    # Subset by space
    #
    # subset spatially finding only those fires within the bounds of the plume
    # note Matplotlib path might be a better option to check with bounds
    # see here: https://goo.gl/Cevi1u.  Also, do we need to project first to
    # determine if the points are inside the polygon?  I think not as everything
    # is in, in effect, geographic projection.  So should be fine.
    plume_polygon = build_polygon(plume, lats, lons)

    inbounds = []
    try:
        for i, (index, frp_pixel) in enumerate(frp_subset.iterrows()):
            if frp_pixel['point'].within(plume_polygon):  # TODO THIS IS WRONG, NEED TO TRANSFORM
                inbounds.append(i)
        if inbounds:
            frp_subset = frp_subset.iloc[inbounds]
    except Exception, e:
        print 'Could not extract spatial subset, failed with error:', str(e)
        return None

    # group by time and aggregate the FRP variables
    frp_subset['FIRE_CONFIDENCE_mean'] = frp_subset['FIRE_CONFIDENCE']
    frp_subset['FIRE_CONFIDENCE_std'] = frp_subset['FIRE_CONFIDENCE']
    frp_subset = frp_subset.groupby('obs_time').agg({'FRP_0': np.sum,
                                                     'FIRE_CONFIDENCE_mean': np.mean,
                                                     'FIRE_CONFIDENCE_std': np.std})[['FRP_0',
                                                                                      'FIRE_CONFIDENCE_mean',
                                                                                      'FIRE_CONFIDENCE_std']]

    # integrate to get the fre
    fre = compute_fre(frp_subset)
    return fre







def find_landcover_class(plume, landcover):
    y = plume.filename[10:14]
    doy = plume.filename[14:17]
    time = plume.filename[18:22]

    myd14 = glob.glob(os.path.join(filepaths.path_to_modis_frp, '*A' + y + doy + '.' + time + '*.hdf'))[0]
    myd14 = SD(myd14, SDC.READ)

    lines = myd14.select('FP_line').get()
    samples = myd14.select('FP_sample').get()
    lats = myd14.select('FP_latitude').get()
    lons = myd14.select('FP_longitude').get()

    poly_verts = plume['plume_extent']
    bb_path = Path(poly_verts)

    # find the geographic coordinates of fires inside the plume mask
    lat_list = []
    lon_list = []
    for l, s, lat, lon in zip(lines, samples, lats, lons):
        if bb_path.contains_point((s, l)):
            lat_list.append(lat)
            lon_list.append(lon)

    # now get the landcover points
    lc_list = []
    for lat, lon in zip(lat_list, lon_list):
        s = int((lon - (-180)) / 360 * landcover['lon'].size)  # lon index
        l = int((lat - 90) * -1 / 180 * landcover['lat'].size)  # lat index

        # image is flipped, so we need to reverse the lat coordinate
        l = -(l + 1)

        lc_list.append(np.array(landcover['Band1'][(l - 1):l, s:s + 1][0])[0])

    # return the most common landcover class for the fire contined in the ROI
    return stats.mode(lc_list).mode[0]


def get_orac_data(plume, orac_file_path):
    y = plume.filename[10:14]
    doy = plume.filename[14:17]
    time = plume.filename[18:22]
    orac_file = glob.glob(os.path.join(orac_file_path, y, doy, 'main', '*' + time + '*.primary.nc'))[0]
    return Dataset(orac_file)



def compute_fre(plume_polygon, frp_data):
    pass





#########################    TPM    #########################