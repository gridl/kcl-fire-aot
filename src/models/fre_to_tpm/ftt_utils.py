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


#########################    GENERAL UTILS    #########################


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


def read_geo(path, plume):
    myd = SD(os.path.join(path, plume.filename), SDC.READ)
    lats = ndimage.zoom(myd.select('Latitude').get(), 5)
    lons = ndimage.zoom(myd.select('Longitude').get(), 5)
    return lats, lons


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
        df_path = glob.glob(path + 'frp_df.p')
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


def construct_polygon(plume, lats, lons):
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


def construct_bounding_box(plume):
    padding = 3  # pixels
    x, y = zip(*plume.plume_extent)
    min_x, max_x = np.min(x) - padding, np.max(x) + padding
    min_y, max_y = np.min(y) - padding, np.max(y) + padding
    return {'max_x': max_x, 'min_x': min_x, 'max_y': max_y, 'min_y': min_y}


def construct_plume_mask(plume, bounds):
    extent = [[x - bounds['min_x'], y - bounds['min_y']] for x, y in plume.plume_extent]

    size_x = bounds['max_x'] - bounds['min_x']
    size_y = bounds['max_y'] - bounds['min_y']
    x, y = np.meshgrid(np.arange(size_x), np.arange(size_y))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    return points


class _utm_resampler(object):
    def __init__(self, lats, lons, pixel_size, resolution=0.01):
        self.lats = lats
        self.lons = lons
        self.pixel_size = pixel_size
        self.resolution = resolution
        self.map_def = self.__utm_map()

    def __utm_zone(self):
        '''
        Some of the plumes will cross UTM zones.  This is not problematic
        as the plumes are quite small and so, we can just use the zone
        in which most of the data falls: https://goo.gl/3QY2Re
        see also: http://www.igorexchange.com/node/927 for if we need over Svalbard (highly unlikely)
        '''
        lons = (self.lons + 180) - np.floor((self.lons + 180) / 360) * 360 - 180;
        return stats.mode(np.floor((lons + 180) / 6) + 1, axis=None)[0][0]

    def __utm_boundaries(self, zone):
        p = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84', datum='WGS84')
        x_ll, y_ll = p(self.lons[-1, 0], self.lats[-1, 0])
        x_ur, y_ur = p(self.lons[0, -1], self.lats[0, -1])
        return {"x_ll": x_ll, "y_ll": y_ll, "x_ur" :x_ur, "y_ur" :y_ur}

    def __utm_grid_size(self, area_extent):
        x_size = (np.abs(area_extent['x_ur'] - area_extent['x_ll']) / self.pixel_size).astype(int)
        y_size = (np.abs(area_extent['y_ur'] - area_extent['y_ll']) / self.pixel_size).astype(int)
        return x_size, y_size

    def __utm_proj(self, zone, area_extent, x_size, y_size):
        area_id = 'utm'
        description = 'utm_grid'
        proj_id = 'utm'
        extent = (area_extent['x_ll'], area_extent['y_ll'], area_extent['x_ur'], area_extent['y_ur'])
        proj_dict = {'units': 'm', 'proj': 'utm', 'zone': str(zone), 'ellps': 'WGS84', 'datum': 'WGS84'}
        return pr.geometry.AreaDefinition(area_id, description, proj_id, proj_dict, x_size, y_size, extent)

    def utm_area_def(self):
        zone = self.__utm_zone()
        area_extent = self.__utm_boundaries(zone)
        x_size, y_size = self.__utm_grid_size(area_extent)
        return self.__utm_proj(zone, area_extent, x_size, y_size)

    def resample_image(self, image, image_lats, image_lons):
        image_def = pr.geometry.SwathDefinition(lons=image_lons, lats=image_lats)
        return pr.kd_tree.resample_nearest(image_def,
                                           image,
                                           self.map_def,
                                           radius_of_influence=75000,
                                           fill_value=-999)


#########################    FRE UTILS    #########################

def integrate_frp(frp_subset):
    try:
        t0 = frp_subset.index[0]
        sample_times = (frp_subset.index - t0).total_seconds()
    except Exception, e:
        print 'Could not extract spatial subset, failed with error:', str(e)
        return None

    # now integrate
    return integrate.trapz(frp_subset['FRP_0'], sample_times)



def compute_fre(plume_polygon, frp_df, start_time, stop_time):

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
    fre = integrate_frp(frp_subset)
    return fre


#########################    TPM    #########################

# TODO still need to make sure area is being calculated correctly
def compute_plume_area(plume_polygon, lons):


    # TODO is sinusoidal proj good enough?  Yes it is: https://goo.gl/KE3tuY
    # get extra accuracy by selecting an appropriate lon_0
    m = Basemap(projection='sinu', lon_0=140, resolution='c')

    lons = (lons + 180) - np.floor((lons + 180) / 360) * 360 - 180;
    zone = stats.mode(np.floor((lons + 180) / 6) + 1, axis=None)[0][0]
    p = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84', datum='WGS84')

    # apply to shapely polygon
    projected_plume_polygon_m = transform(m, plume_polygon)
    projected_plume_polygon_p = transform(p, plume_polygon)

    # compute projected polygon area in m2
    return projected_plume_polygon_m.area, projected_plume_polygon_p.area


def compute_aod(plume_bounding_pixels, plume_mask, lats, lons):
    '''
    Resampling of the ORAC AOD data is required to remove the bowtie effect from the data.  We can then
    sum over the AODs contained with the plume.

    Resampling of the MODIS AOD data is required so that it is in the same projection as the ORAC AOD.
    With it being in the same projection we can replace low quality ORAC AODs with those from MXD04 products.
    We also need to get the background AOD data from MXD04 products as ORAC does not do optically thin
    retrievals well.

    Also, we need to check that the pixel area estimates being computed by compute_plume_area are reasonable.
    That can be done in this function, we just produce another area estimate from the resampled mask by getting
    vertices, getting the lat lons of the vertices, creating a shapely polygon from them and then computing the area.
    '''

    # get the resampler
    lats = lats[plume_bounding_pixels['min_y']:plume_bounding_pixels['max_y'],
                plume_bounding_pixels['min_x']:plume_bounding_pixels['max_x']]
    lons = lons[plume_bounding_pixels['min_y']:plume_bounding_pixels['max_y'],
                plume_bounding_pixels['min_x']:plume_bounding_pixels['max_x']]
    im_resampler = _utm_resampler(lats, lons, 1000)

    # resample the datasets (mask, orac_aod, MYD04)
    resampled_plume_mask = im_resampler.resample_image(plume_mask, lats, lons)

    print np.sum(resampled_plume_mask)

    # create best AOD map from ORAC and MYD04 AOD

    # extract best AOD using plume mask

    # split into background and plume AODs

    # subtract background from plume

    # return plume AOD

