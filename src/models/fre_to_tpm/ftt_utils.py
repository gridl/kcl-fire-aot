'''
Contains the various functions and classes that are used in the
ftt (fre-to-tpm) processor.  These can be broken down as follows:
'''

# load in required packages
import ast
import glob
import os
import logging
from functools import partial

import pandas as pd
import numpy as np
from netCDF4 import Dataset
from pyhdf.SD import SD, SDC
from matplotlib.path import Path
from scipy import stats
from scipy import ndimage
from shapely.geometry import Polygon, Point, MultiPoint
from shapely.ops import transform
import pyresample as pr
import pyproj
import re

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


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

def read_orac_data(plume, orac_file_path):
    y = plume.filename[10:14]
    doy = plume.filename[14:17]
    time = plume.filename[18:22]
    orac_file = glob.glob(os.path.join(orac_file_path, y, doy, 'main', '*' + time + '*.primary.nc'))[0]
    return Dataset(orac_file)


def read_plume_polygons(path):
    try:
        df = pd.read_pickle(path)
    except Exception, e:
        logger.warning('Could not load pickle with error:' + str(e) + ' ...attempting to load csv')
        df = pd.read_csv(path, quotechar='"', sep=',', converters={'plume_extent': ast.literal_eval})
    return df


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


def read_frp_df(path):
    '''

    :param path: path to frp csv files and dataframe
    :return: the frp holding dataframe
    '''
    try:
        df_path = glob.glob(path + 'frp_df.p')[0]
        frp_df = pd.read_pickle(df_path)
    except Exception, e:
        print('could not load frp dataframe, failed with error ' + str(e) + ' building anew')
        frp_df = _build_frp_df(path)
    return frp_df


def find_landcover_class(plume, myd14_path, landcover_ds):
    y = plume.filename[10:14]
    doy = plume.filename[14:17]
    time = plume.filename[18:22]

    myd14 = glob.glob(os.path.join(myd14_path, '*A' + y + doy + '.' + time + '*.hdf'))[0]  # path to modis FRP
    myd14 = SD(myd14, SDC.READ)

    lines = myd14.select('FP_line').get()
    samples = myd14.select('FP_sample').get()
    lats = myd14.select('FP_latitude').get()
    lons = myd14.select('FP_longitude').get()

    poly_verts = plume['plume_extent']
    bb_path = Path(poly_verts)

    # find the geographic coordinates of fires inside the plume mask # TODO need to project here too?
    lat_list = []
    lon_list = []
    for l, s, lat, lon in zip(lines, samples, lats, lons):
        if bb_path.contains_point((s, l)):
            lat_list.append(lat)
            lon_list.append(lon)

    # now get the landcover points
    lc_list = []
    for lat, lon in zip(lat_list, lon_list):
        s = int((lon - (-180)) / 360 * landcover_ds['lon'].size)  # lon index
        l = int((lat - 90) * -1 / 180 * landcover_ds['lat'].size)  # lat index

        # image is flipped, so we need to reverse the lat coordinate
        l = -(l + 1)

        lc_list.append(np.array(landcover_ds['Band1'][(l - 1):l, s:s + 1][0])[0])

    # return the most common landcover class for the fire contined in the ROI
    return stats.mode(lc_list).mode[0]


def construct_bounding_box(plume):
    padding = 10  # pixels
    x, y = zip(*plume.plume_extent)
    min_x, max_x = np.min(x) - padding, np.max(x) + padding
    min_y, max_y = np.min(y) - padding, np.max(y) + padding
    return {'max_x': max_x, 'min_x': min_x, 'max_y': max_y, 'min_y': min_y}


def read_modis_geo_subset(path, plume, bounds):
    myd = SD(os.path.join(path, plume.filename), SDC.READ)
    lats = ndimage.zoom(myd.select('Latitude').get(), 5)[bounds['min_y']:bounds['max_y'],
           bounds['min_x']:bounds['max_x']]
    lons = ndimage.zoom(myd.select('Longitude').get(), 5)[bounds['min_y']:bounds['max_y'],
           bounds['min_x']:bounds['max_x']]
    return lats, lons


def construct_plume_mask(plume, bounds):
    extent = [[x - bounds['min_x'], y - bounds['min_y']] for x, y in plume.plume_extent]

    size_x = bounds['max_x'] - bounds['min_x']
    size_y = bounds['max_y'] - bounds['min_y']
    x, y = np.meshgrid(np.arange(size_x), np.arange(size_y))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T

    # create mask
    path = Path(extent)
    mask = path.contains_points(points)
    mask = mask.reshape((size_y, size_x))
    return mask


def extract_fires(path, plume, myd14):
    myd = SD(os.path.join(path, plume.filename), SDC.READ)
    lats = ndimage.zoom(myd.select('Latitude').get(), 5)
    lons = ndimage.zoom(myd.select('Longitude').get(), 5)
    return lats[myd14[0], myd14[1]], lons[myd14[0], myd14[1]]


def fires_in_plume(fires, plume_polygon):
    inbound_fires_y = []
    inbound_fires_x = []
    for pt in zip(fires[0], fires[1]):
        if plume_polygon.contains(Point(pt[1], pt[0])):
            inbound_fires_y.append(pt[0])
            inbound_fires_x.append(pt[1])
    return inbound_fires_y, inbound_fires_x


def _extract_geo_from_bounds(plume, bounds, lats, lons):
    # adjust plume extent for the subset
    extent = [[x - bounds['min_x'], y - bounds['min_y']] for x, y in plume.plume_extent]

    # when digitising points are appended (x,y).  However, arrays are accessed
    # in numpy as row, col which is y, x.  So we need to switch
    bounding_lats = [lats[point[1], point[0]] for point in extent]
    bounding_lons = [lons[point[1], point[0]] for point in extent]
    return bounding_lats, bounding_lons


def construct_points(plume, bounds, lats, lons):
    bounding_lats, bounding_lons = _extract_geo_from_bounds(plume, bounds, lats, lons)
    return MultiPoint(zip(bounding_lons, bounding_lats))


def construct_polygon(plume, bounds, lats, lons):
    bounding_lats, bounding_lons = _extract_geo_from_bounds(plume, bounds, lats, lons)
    return Polygon(zip(bounding_lons, bounding_lats))


def reproject_shapely(shapely_object, utm_resampler):
    project = partial(
        pyproj.transform,
        pyproj.Proj(init='epsg:4326'),  # source coordinate system (geographic coords)
        utm_resampler.proj)  # destination coordinate system

    return transform(project, shapely_object)  # apply projection


class utm_resampler(object):
    def __init__(self, lats, lons, pixel_size, resolution=0.01):
        self.lats = lats
        self.lons = lons
        self.pixel_size = pixel_size
        self.resolution = resolution
        self.zone = self.__utm_zone()
        self.proj = self.__utm_proj()
        self.extent = self.__utm_extent()
        self.x_size, self.y_size = self.__utm_grid_size()
        self.area_def = self.__construct_area_def()

    def __utm_zone(self):
        '''
        Some of the plumes will cross UTM zones.  This is not problematic
        as the plumes are quite small and so, we can just use the zone
        in which most of the data falls: https://goo.gl/3QY2Re
        see also: http://www.igorexchange.com/node/927 for if we need over Svalbard (highly unlikely)
        '''
        lons = (self.lons + 180) - np.floor((self.lons + 180) / 360) * 360 - 180
        return stats.mode(np.floor((lons + 180) / 6) + 1, axis=None)[0][0]

    def __utm_proj(self):
        return pyproj.Proj(proj='utm', zone=self.zone, ellps='WGS84', datum='WGS84')

    def __utm_extent(self):
        x, y = self.proj(self.lons, self.lats)
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
        return (min_x, min_y, max_x, max_y)

    def __utm_grid_size(self):
        x_size = int(np.ceil((self.extent[2] - self.extent[0]) / self.pixel_size))
        y_size = int(np.ceil((self.extent[3] - self.extent[1]) / self.pixel_size))
        return x_size, y_size

    def __construct_area_def(self):
        area_id = 'utm'
        description = 'utm_grid'
        proj_id = 'utm'
        proj_dict = {'units': 'm', 'proj': 'utm', 'zone': str(self.zone), 'ellps': 'WGS84', 'datum': 'WGS84'}
        return pr.geometry.AreaDefinition(area_id, description, proj_id, proj_dict,
                                          self.x_size, self.y_size, self.extent)

    def resample_image(self, image, image_lats, image_lons):
        swath_def = pr.geometry.SwathDefinition(lons=image_lons, lats=image_lats)
        return pr.kd_tree.resample_nearest(swath_def,
                                           image,
                                           self.area_def,
                                           radius_of_influence=75000,
                                           fill_value=-999)

    def resample_points_to_utm(self, point_lats, point_lons):
        return [self.proj(lon, lat) for lon, lat in zip(point_lons, point_lats)]

    def resample_point_to_geo(self, point_y, point_x):
        return self.proj(point_x, point_y, inverse=True)
