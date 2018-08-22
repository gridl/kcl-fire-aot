#!/home/users/dnfisher/soft/virtual_envs/kcl-fire-aot/bin/python2

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
from datetime import datetime
from dateutil.parser import parse
import re

import pandas as pd
import numpy as np
from netCDF4 import Dataset
import h5py
from matplotlib.path import Path
from scipy import stats
from shapely.geometry import Polygon, Point, MultiPoint, LineString
from shapely.ops import transform
import pyresample as pr
import pyproj
import scipy.misc as misc

import src.features.fre_to_tpm.viirs.ftt_fre as ff
import src.config.filepaths as fp
import src.config.constants as constants
import src.features.fre_to_tpm.viirs.ftt_tpm as tt

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


#### Reader utils ####

def read_h5(f):
    return h5py.File(f,  "r")


def read_nc(f):
    return Dataset(f)


def get_timestamp(fname, sensor):

    if sensor not in ['orac', 'viirs']:
        logger.critical('Sensor Not Implemented!  Need to add to get_timstamp in FTT utils')
        return ''

    if sensor == 'viirs':
        try:
            return re.search("[d][0-9]{8}[_][t][0-9]{6}", fname).group()
        except Exception, e:
            logger.warning("Could not extract time stamp from: " + fname + " with error: " + str(e))
            return ''
    if sensor == 'orac':
        try:
            return re.search("[_][0-9]{12}[_]", fname).group()
        except Exception, e:
            logger.warning("Could not extract time stamp from: " + fname + " with error: " + str(e))
            return ''


def sat_data_reader(p, sensor, var, timestamp):
    '''

    :param p: path to data
    :param sensor: sensor to read data for
    :param var: variable to extract
    :return: the var for sensor
    '''


    if sensor not in ['orac', 'viirs']:
        logger.critical('Sensor Not Implemented!  Need to add to reader in FTT utils')
        return ''

    if sensor == 'orac':

        # check timestamp is in correct form
        if isinstance(timestamp, basestring):
            # strip time and get in the right format
            try:
                timestamp = parse(timestamp, fuzzy=True)
                orac_time_format = datetime.strftime(timestamp, '%Y%m%d%H%M')
            except Exception, e:
                logger.critical(str(e))
        else:
            # get string from time in the right format
            orac_time_format = datetime.strftime(timestamp, '%Y%m%d%H%M')

        orac_fname = [f for f in os.listdir(p) if orac_time_format in f][0]
        ds = read_nc(os.path.join(p, orac_fname))

        if var == 'aod':
            return ds['cot'][:]
        elif var == 'cost':
            return ds['costjm'][:]
        elif var == 'geo':
            return ds['lat'][:], ds['lon'][:]

    if sensor == 'modis':
        pass

    if sensor == 'viirs':
        # check timestamp is in correct form
        if isinstance(timestamp, basestring):
            try:
                timestamp = parse(timestamp, fuzzy=True)
                viirs_time_format = datetime.strftime(timestamp, 'd%Y%m%d_t%H%M%S')
            except Exception, e:
                logger.critical(str(e))
        else:
            # get string from time in the right format
            viirs_time_format = datetime.strftime(timestamp, 'd%Y%m%d_t%H%M%S')


        viirs_fname = [f for f in os.listdir(p) if viirs_time_format in f]
        ds = read_h5(os.path.join(p, viirs_fname[-1]))  # might be multiple matches, takes most recent

        if var == 'aod':
            return ds['All_Data']['VIIRS-Aeros-Opt-Thick-IP_All']['faot550'][:]
        elif var == 'flag':
            return extract_viirs_flags(ds)


def extract_viirs_flags(viirs_data):
    aod_quality = viirs_data['All_Data']['VIIRS-Aeros-Opt-Thick-IP_All']['QF1'][:]
    flags = np.zeros(aod_quality.shape)
    for k, v in zip(['00', '01', '10', '11'], [0, 1, 2, 3]):
        mask = get_mask(aod_quality, 0, 2, k)
        flags[mask] = v
    return flags


def get_mask(arr, bit_pos, bit_len, value):
    '''Generates mask with given bit information.
    Parameters
        bit_pos		-	Position of the specific QA bits in the value string.
        bit_len		-	Length of the specific QA bits.
        value  		-	A value indicating the desired condition.
    '''
    bitlen = int('1' * bit_len, 2)

    if type(value) == str:
        value = int(value, 2)

    pos_value = bitlen << bit_pos
    con_value = value << bit_pos
    mask = (arr & pos_value) == con_value
    return mask


def read_plume_polygons(path):
    try:
        df = pd.read_pickle(path)
    except Exception, e:
        logger.warning('Could not load pickle with error:' + str(e) + ' ...attempting to load csv')
        df = pd.read_csv(path, quotechar='"', sep=',', converters={'plume_extent': ast.literal_eval,
                                                                   'background_extent': ast.literal_eval})
    return df


def read_frp_df(path):
    '''

    :param path: path to frp csv files and dataframe
    :return: the frp holding dataframe
    '''
    try:
        df_path = os.path.join(path, 'frp_df.p')
        frp_df = pd.read_pickle(df_path)
    except Exception, e:
        print('could not load frp dataframe, failed with error ' + str(e) + ' building anew')
        frp_df = build_frp_df(path)
    return frp_df


def build_frp_df(path):
    '''

    :param path: path to the frp csv files and dataframe
    :return: dataframe holding frp
    '''
    frp_csv_files = glob.glob(path + '/*.csv')
    df_from_each_file = (pd.read_csv(f) for f in frp_csv_files)
    frp_df = pd.concat(df_from_each_file, ignore_index=True)

    # keep only columns on interest
    frp_df = frp_df[['FIRE_CONFIDENCE', 'ABS_line', 'ABS_samp', 'BT_MIR', 'BT_TIR',
                     'FRP_0', 'LATITUDE', 'LONGITUDE', 'year', 'month', 'day', 'time']]

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
    #frp_df.drop(['LONGITUDE', 'LATITUDE'], axis=1, inplace=True)
    frp_df.drop(['year', 'month', 'day', 'time'], axis=1, inplace=True)

    frp_df.to_pickle(path + '/frp_df.p')

    return frp_df



#### Spatial Utils
def construct_bounding_box(extent):
    padding = 50  # pixels  TODO Move to config file
    x, y = zip(*extent)
    min_x, max_x = np.min(x) - padding, np.max(x) + padding
    min_y, max_y = np.min(y) - padding, np.max(y) + padding
    return {'max_x': max_x, 'min_x': min_x, 'max_y': max_y, 'min_y': min_y}


def subset_data(data, bounds):
    return data[bounds['min_y']:bounds['max_y'], bounds['min_x']:bounds['max_x']]


def construct_mask(e, bounds):
    extent = [[x - bounds['min_x'], y - bounds['min_y']] for x, y in e]

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


def extract_subset_geo_bounds(ext, bounds, lats, lons):
    # adjust plume extent for the subset
    extent = [[x - bounds['min_x'], y - bounds['min_y']] for x, y in ext]

    # when digitising points are appended (x,y).  However, arrays are accessed
    # in numpy as row, col which is y, x.  So we need to switch
    bounding_lats = [lats[point[1], point[0]] for point in extent]
    bounding_lons = [lons[point[1], point[0]] for point in extent]
    return bounding_lats, bounding_lons


def construct_shapely_points(bounding_lats, bounding_lons):
    return MultiPoint(zip(bounding_lons, bounding_lats))


def construct_shapely_polygon(bounding_lats, bounding_lons):
    return Polygon(zip(bounding_lons, bounding_lats))


def construct_shapely_vector(bounding_lats, bounding_lons):
    return LineString(zip(bounding_lons[0:2], bounding_lats[0:2]))


def reproject_shapely(shapely_object, utm_resampler):
    project = partial(
        pyproj.transform,
        pyproj.Proj(init='epsg:4326'),  # source coordinate system (geographic coords)
        utm_resampler.proj)  # destination coordinate system

    return transform(project, shapely_object)  # apply projection


def create_logger_path(p_number):
    # check logging dir exists
    if not os.path.isdir(fp.pt_vis_path):
        os.mkdir(fp.pt_vis_path)

    # check plume specific dir exists
    plume_logging_path = os.path.join(fp.pt_vis_path, str(p_number))
    if not os.path.isdir(plume_logging_path):
        os.mkdir(plume_logging_path)
    return plume_logging_path


def resample_satellite_datasets(sat_data, pp=None, plume=None, fill_value=0):

    # set up resampler
    utm_rs = utm_resampler(sat_data['lats'], sat_data['lons'], constants.utm_grid_size)

    # get the mask for the lats and lons and apply
    null_mask = np.ma.getmask(sat_data['orac_aod'])
    masked_lats = np.ma.masked_array(utm_rs.lats, null_mask)
    masked_lons = np.ma.masked_array(utm_rs.lons, null_mask)

    # resample all the datasets to UTM
    d = {}
    d['viirs_aod_utm'] = utm_rs.resample_image(sat_data['viirs_aod'], masked_lats, masked_lons, fill_value=fill_value)
    d['viirs_flag_utm'] = utm_rs.resample_image(sat_data['viirs_flag'], masked_lats, masked_lons, fill_value=fill_value)
    d['orac_aod_utm'] = utm_rs.resample_image(sat_data['orac_aod'], masked_lats, masked_lons, fill_value=fill_value)
    d['orac_cost_utm'] = utm_rs.resample_image(sat_data['orac_cost'], masked_lats, masked_lons, fill_value=fill_value)
    d['lats'] = utm_rs.resample_image(utm_rs.lats, masked_lats, masked_lons, fill_value=fill_value)
    d['lons'] = utm_rs.resample_image(utm_rs.lons, masked_lats, masked_lons, fill_value=fill_value)

    if pp:
        if pp['plot']:
            d['viirs_png_utm'] = misc.imread(os.path.join(fp.path_to_viirs_sdr_resampled_no_peat, plume.filename.rstrip()))

    return d


def setup_plume_data(plume, ds_utm):
    d = {}
    try:
        # get plume extent geographic data (bounding box in in UTM as plume extent is UTM)
        d['plume_bounding_box'] = construct_bounding_box(plume.plume_extent)
        d['plume_lats'] = subset_data(ds_utm['lats'], d['plume_bounding_box'])
        d['plume_lons'] = subset_data(ds_utm['lons'], d['plume_bounding_box'])
        d['plume_aod'] = subset_data(ds_utm['viirs_aod_utm'], d['plume_bounding_box'])
        d['plume_flag'] = subset_data(ds_utm['viirs_flag_utm'], d['plume_bounding_box'])

         # get plume polygon geographic data
        poly_lats, poly_lons = extract_subset_geo_bounds(plume.plume_extent, d['plume_bounding_box'],
                                                            d['plume_lats'], d['plume_lons'])

        # get plume mask
        d['plume_mask'] = construct_mask(plume.plume_extent, d['plume_bounding_box'])

        # setup shapely objects for plume geo data
        d['plume_points'] = construct_shapely_points(poly_lats, poly_lons)
        d['plume_polygon'] = construct_shapely_polygon(poly_lats, poly_lons)

        d['background_bounding_box'] = construct_bounding_box(plume.background_extent)
        d['background_mask'] = construct_mask(plume.background_extent, d['background_bounding_box'])
        d['bg_aod'] = subset_data(ds_utm['viirs_aod_utm'], d['background_bounding_box'])
        d['bg_flag'] = subset_data(ds_utm['viirs_flag_utm'], d['background_bounding_box'])

        return d
    except Exception, e:
        logger.error(str(e))
        return None


def subset_sat_data_to_plume(sat_data_utm, plume_geom_geo):
    d = {}
    d['viirs_aod_utm_plume'] = subset_data(sat_data_utm['viirs_aod_utm'], plume_geom_geo['plume_bounding_box'])
    d['viirs_flag_utm_plume'] = subset_data(sat_data_utm['viirs_flag_utm'], plume_geom_geo['plume_bounding_box'])
    d['orac_aod_utm_plume'] = subset_data(sat_data_utm['orac_aod_utm'], plume_geom_geo['plume_bounding_box'])
    d['orac_cost_utm_plume'] = subset_data(sat_data_utm['orac_cost_utm'], plume_geom_geo['plume_bounding_box'])

    d['viirs_aod_utm_background'] = subset_data(sat_data_utm['viirs_aod_utm'],
                                                   plume_geom_geo['background_bounding_box'])
    d['viirs_flag_utm_background'] = subset_data(sat_data_utm['viirs_flag_utm'],
                                                    plume_geom_geo['background_bounding_box'])
    d['orac_aod_utm_background'] = subset_data(sat_data_utm['orac_aod_utm'],
                                                  plume_geom_geo['background_bounding_box'])
    d['orac_cost_utm_background'] = subset_data(sat_data_utm['orac_cost_utm'],
                                                   plume_geom_geo['background_bounding_box'])
    return d


def resample_plume_geom_to_utm(plume_geom_geo):
    d = {}
    d['utm_resampler_plume'] = utm_resampler(plume_geom_geo['plume_lats'],
                                                plume_geom_geo['plume_lons'],
                                                constants.utm_grid_size)
    d['utm_plume_points'] = reproject_shapely(plume_geom_geo['plume_points'], d['utm_resampler_plume'])
    d['utm_plume_polygon'] = reproject_shapely(plume_geom_geo['plume_polygon'], d['utm_resampler_plume'])
    return d


def process_plume(t1, t2, pp, plume_data_utm, plume_geom_utm, plume_geom_geo, plume_logging_path, p_number,
                  df_list):
    # get background aod for sub plume
    bg_aod_dict = tt.extract_bg_aod(plume_data_utm, plume_geom_geo['background_mask'])

    # compute tpm
    out_dict = tt.compute_tpm_full(plume_data_utm, plume_geom_utm, plume_geom_geo, bg_aod_dict, plume_logging_path, pp)
    out_dict['plume_number'] = p_number

    # compute fre
    ff.compute_fre_full_plume(t1, t2, pp['frp_df'], plume_geom_geo, plume_logging_path, out_dict)

    # convert datadict to dataframe and add to list
    df_list.append(pd.DataFrame(out_dict, index=['i', ]))


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
        lats = self.lats
        lons = self.lons
        mask = (lats < 90) & (lats > -90) & (lons < 180) & (lons > -180)
        x, y = self.proj(lons[mask], lats[mask])
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
        return (min_x, min_y, max_x, max_y)

    def __utm_grid_size(self):
        x_size = int(np.round((self.extent[2] - self.extent[0]) / self.pixel_size))
        y_size = int(np.round((self.extent[3] - self.extent[1]) / self.pixel_size))
        return x_size, y_size

    def __construct_area_def(self):
        area_id = 'utm'
        description = 'utm_grid'
        proj_id = 'utm'
        proj_dict = {'units': 'm', 'proj': 'utm', 'zone': str(self.zone), 'ellps': 'WGS84', 'datum': 'WGS84'}
        return pr.geometry.AreaDefinition(area_id, description, proj_id, proj_dict,
                                          self.x_size, self.y_size, self.extent)

    def resample_image(self, image, image_lats, image_lons, fill_value=-999):
        swath_def = pr.geometry.SwathDefinition(lons=image_lons, lats=image_lats)
        return pr.kd_tree.resample_nearest(swath_def,
                                           image,
                                           self.area_def,
                                           radius_of_influence=1500,
                                           fill_value=fill_value)

    def resample_points_to_utm(self, point_lats, point_lons):
        return [self.proj(lon, lat) for lon, lat in zip(point_lons, point_lats)]

    def resample_point_to_geo(self, point_y, point_x):
        return self.proj(point_x, point_y, inverse=True)
