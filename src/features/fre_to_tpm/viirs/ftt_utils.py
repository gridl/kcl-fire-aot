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
from skimage.measure import grid_points_in_poly
from sklearn.neighbors import BallTree
import scipy.misc as misc
import matplotlib.pyplot as plt

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


def open_orac_ds(path, timestamp):
    orac_fname = get_orac_fname(path, timestamp)
    return read_nc(os.path.join(path, orac_fname))


def extract_orac_aod(orac_data):
    return orac_data.variables['cot'][:]


def extract_orac_cost(orac_data):
    return orac_data['costjm'][:]


def extract_orac_geo(orac_data):
    lats = orac_data.variables['lat'][:]
    lons = orac_data.variables['lon'][:]
    return lats, lons


def open_viirs_ds(path, timestamp, filename):
    viirs_fname = get_viirs_fname(path, timestamp, filename)
    viirs_data = read_h5(os.path.join(path, viirs_fname))
    return viirs_data


def extract_viirs_flags(viirs_data):
    aod_quality = viirs_data['All_Data']['VIIRS-Aeros-Opt-Thick-IP_All']['QF1'][:]
    flags = np.zeros(aod_quality.shape)
    for k, v in zip(['00', '01', '10', '11'], [0, 1, 2, 3]):
        mask = get_mask(aod_quality, 0, 2, k)
        flags[mask] = v
    return flags


def extract_viirs_aod(viirs_data):
    return viirs_data['All_Data']['VIIRS-Aeros-Opt-Thick-IP_All']['faot550'][:]


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
                                                                   'background_extent': ast.literal_eval,
                                                                   'plume_vector': ast.literal_eval})
    return df


def read_frp_df(path):
    '''

    :param path: path to frp csv files and dataframe
    :return: the frp holding dataframe
    '''
    try:
        print path
        df_path = glob.glob(path + 'frp_df.p')[0]
        print df_path
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
    frp_csv_files = glob.glob(path + '*.csv')
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

    frp_df.to_pickle(path + 'frp_df.p')

    return frp_df


#### File name matching utils ####
def get_timestamp(viirs_sdr_fname):
    try:
        return re.search("[d][0-9]{8}[_][t][0-9]{6}", viirs_sdr_fname).group()
    except Exception, e:
        logger.warning("Could not extract time stamp from: " + viirs_sdr_fname + " with error: " + str(e))
        return ''


def get_viirs_fname(path, timestamp_viirs, viirs_sdr_fname):
    fname = [f for f in os.listdir(path) if timestamp_viirs in f]
    if len(fname) > 1:
        logger.warning("More that one frp granule matched " + viirs_sdr_fname + "selecting 0th option")
        return fname[0]
    elif len(fname) == 1:
        return fname[0]
    else:
        return ''


def get_orac_fname(path, timestamp_viirs):
    t = datetime.strptime(timestamp_viirs, 'd%Y%m%d_t%H%M%S')
    t = datetime.strftime(t, '%Y%m%d%H%M')
    fname = [f for f in os.listdir(path) if t in f]
    return fname[0]


#### Spatial Utils
def make_balltree(array_lats, array_lons):
    array_lat_lon = np.dstack([np.deg2rad(array_lats.flatten()), np.deg2rad(array_lons.flatten())])[0]
    return BallTree(array_lat_lon, metric='haversine')


def make_balltree_subset(array_lats_flat, array_lons_flat):
    array_lat_lon = np.dstack([np.deg2rad(array_lats_flat), np.deg2rad(array_lons_flat)])[0]
    return BallTree(array_lat_lon, metric='haversine')


def spatial_intersection(array_balltree, x_shape, point_lat, point_lon):

    # get the unique flare lats and lons for assessment in kdtree
    point_location = np.array([np.deg2rad(point_lat), np.deg2rad(point_lon)]).reshape(1,-1)

    # compare the flare locations to the potential locations in the orbit
    distances, indexes = array_balltree.query(point_location, k=1)

    # set up the dataframe to hold the distances
    x = indexes[0][0] % x_shape
    y = indexes[0][0] / x_shape
    d = distances[0][0]
    return x, y, np.rad2deg(d)


def spatial_intersection_subset(array_balltree, point_lat, point_lon, x_positions_flat, y_positions_flat):
    # get the unique flare lats and lons for assessment in kdtree
    point_location = np.array([np.deg2rad(point_lat), np.deg2rad(point_lon)]).reshape(1, -1)

    # compare the flare locations to the potential locations in the orbit
    distance, index = array_balltree.query(point_location, k=1)

    # get indexes
    x = x_positions_flat[index][0][0]
    y = y_positions_flat[index][0][0]
    d = distance[0][0]

    return x, y, np.rad2deg(d)


def construct_bounding_box(extent):
    padding = 10  # pixels  TODO Move to config file
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


def fires_in_plume(fires, plume_polygon):
    inbound_fires_y = []
    inbound_fires_x = []
    for pt in zip(fires[0], fires[1]):
        if plume_polygon.contains(Point(pt[1], pt[0])):
            inbound_fires_y.append(pt[0])
            inbound_fires_x.append(pt[1])
    return inbound_fires_y, inbound_fires_x


def extract_subset_geo_bounds(ext, bounds, lats, lons):
    # adjust plume extent for the subset
    extent = [[x - bounds['min_x'], y - bounds['min_y']] for x, y in ext]

    # when digitising points are appended (x,y).  However, arrays are accessed
    # in numpy as row, col which is y, x.  So we need to switch
    bounding_lats = [lats[point[1], point[0]] for point in extent]
    bounding_lons = [lons[point[1], point[0]] for point in extent]
    return bounding_lats, bounding_lons


def extract_geo_bounds(extent, lats, lons):

    # the extent can be outside of the lat/lon grid, so we need to adjust any
    # indexes that are outside of this extent.
    shape_y, shape_x = lats.shape
    for point in extent:
        if point[0] >= shape_y:
            point[0] = shape_y - 1
        if point[0] < 0:
            point[0] = 0
        if point[1] >= shape_y:
            point[1] = shape_x - 1
        if point[1] < 0:
            point[1] = 0

    # these points are generated as y, x
    bounding_lats = [lats[int(point[0]), int(point[1])] for point in extent]
    bounding_lons = [lons[int(point[0]), int(point[1])] for point in extent]
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


def _extract_geo_from_bounds(ext, bounds, lats, lons):
    # adjust plume extent for the subset
    extent = [[x - bounds['min_x'], y - bounds['min_y']] for x, y in ext]

    # when digitising points are appended (x,y).  However, arrays are accessed
    # in numpy as row, col which is y, x.  So we need to switch
    bounding_lats = [lats[point[1], point[0]] for point in extent]
    bounding_lons = [lons[point[1], point[0]] for point in extent]
    return bounding_lats, bounding_lons


def construct_points(plume, bounds, lats, lons):
    bounding_lats, bounding_lons = _extract_geo_from_bounds(plume.plume_extent, bounds, lats, lons)
    return MultiPoint(zip(bounding_lons, bounding_lats))


def construct_polygon(plume, bounds, lats, lons):
    bounding_lats, bounding_lons = _extract_geo_from_bounds(plume.plume_extent, bounds, lats, lons)
    return Polygon(zip(bounding_lons, bounding_lats))


def construct_vector(plume, bounds, lats, lons):
    bounding_lats, bounding_lons = _extract_geo_from_bounds(plume.plume_vector, bounds, lats, lons)
    return LineString(zip(bounding_lons[0:2], bounding_lats[0:2]))


def reproject_shapely(shapely_object, utm_resampler):
    project = partial(
        pyproj.transform,
        pyproj.Proj(init='epsg:4326'),  # source coordinate system (geographic coords)
        utm_resampler.proj)  # destination coordinate system

    return transform(project, shapely_object)  # apply projection


def compute_perpendicular_slope(vector):
    head = np.array(vector[1])
    tail = np.array(vector[0])
    deltas = head - tail
    slope = float(deltas[1]) / deltas[0]  # dy/ dx
    return -1 / slope  # perpendicular slope


def split_plume_polgons(utm_flow_means, plume_logging_path, plume,
                        plume_geom_geo, plume_geom_utm, pp):

    # compute orthogonal slope of plume vector in pixels
    m = compute_perpendicular_slope(plume.plume_vector)

    y_shape, x_shape = plume_geom_geo['plume_lats'].shape

    # set up iterator variables
    a = [0, 0]  # ll
    b = [0, 0]  # ul
    c = [0, 0]
    d = [0, 0]
    tail_position = np.array(plume_geom_utm['utm_plume_vector'].coords[0])

    # set up list to hold polygon corners
    polygon_corner_dict = {}

    if pp['plot']:
        display = np.zeros(plume_geom_geo['plume_mask'].shape)
        sub_x_positions = []
        sub_y_positions = []

    # iterate over UTM points
    for i, position in enumerate(utm_flow_means):

        # add current position onto tail
        tail_position += position

        # convert UTM point to lat lon
        flow_lon, flow_lat = plume_geom_utm['utm_resampler_plume'].resample_point_to_geo(tail_position[1],
                                                                                      tail_position[0])

        # convert lat lon to pixel index, to give us a point on the line
        dists = np.abs(flow_lat - plume_geom_geo['plume_lats']) + np.abs(flow_lon - plume_geom_geo['plume_lons'])
        sub_y, sub_x = divmod(dists.argmin(), x_shape)

        # using slope, and point on the point, get b so we know have full linear equation
        slope = sub_y - m*sub_x   # b = y - mx, and when x=0, y=b

        # using linear equation determine the intersections of the line with the x and y
        # axes.  If the point of intersection does not exceed the extent of the ROI in the given axis then it is
        # the direction colinear with the plume, and the segmentation must be performed in
        # this axis.
        y0 = m*0 + slope
        y1 = m*x_shape + slope
        x0 = (0-slope) / m
        x1 = (y_shape-slope) / m

        # check if we are updatin in the y direction, else assume x
        update_y = (y0 > 0) & (y0 < y_shape) & (y1 > 0) & (y1 < y_shape)

        if update_y:
            # do y parts
            a[0] = y0
            d[0] = y1
            # do x parts
            c[1] = x_shape-1  # ur
            d[1] = x_shape-1  # lr
        else:
            # do x parts
            c[1] = x0
            d[1] = x1
            # do y parts
            a[0] = y_shape - 1  # ur
            d[0] = y_shape - 1  # lr


        # define mask
        polygon_corner_dict[i] = [a[:], b[:], c[:], d[:]]
        if pp['plot']:
            mask = grid_points_in_poly([y_shape, x_shape], [a, b, c, d])
            display[mask] = i
            sub_x_positions.append(sub_x)
            sub_y_positions.append(sub_y)

        # update polygon corner arrays
        if update_y:
            b = a[:]
            c = d[:]
        else:
            b = c[:]
            a = d[:]

    # now get the final part of the plume
    if update_y:
        a[0] = y_shape-1
        d[0] = y_shape-1
    else:
        c[1] = x_shape-1
        d[1] = x_shape-1

    polygon_corner_dict[i] = [a[:], b[:], c[:], d[:]]

    if pp['plot']:
        mask = grid_points_in_poly([y_shape, x_shape], [a, b, c, d])
        display[mask] = i
        display[~plume_geom_geo['plume_mask']] = np.nan
        plt.imshow(display)
        plt.colorbar()
        for x, y in zip(sub_x_positions, sub_y_positions):
            plt.plot(x,y, 'r.')

        extent = [[x - plume_geom_geo['plume_bounding_box']['min_x'],
                   y - plume_geom_geo['plume_bounding_box']['min_y']] for x, y in plume.plume_vector]
        t = extent[0]
        h = extent[1]
        plt.plot((t[0], h[0]), (t[1], h[1]), 'k-')
        plt.savefig(os.path.join(plume_logging_path, 'plumes_subsets.png'), bbox_inches='tight', dpi=300)
        plt.close()

    return polygon_corner_dict


def sub_mask(shape, poly, plume_mask):
    return grid_points_in_poly(shape, poly) * plume_mask


def create_logger_path(p_number):
    plume_logging_path = os.path.join(fp.path_to_plume_tracking_visualisations_viirs, str(p_number))
    if not os.path.isdir(plume_logging_path):
        os.mkdir(plume_logging_path)
    return plume_logging_path


def resample_satellite_datasets(plume, current_timestamp, pp):
    d = {}

    try:
        viirs_aod_data = open_viirs_ds(fp.path_to_viirs_aod, current_timestamp, plume.filename)
        orac_aod_data = open_orac_ds(fp.path_to_viirs_orac, current_timestamp)
        if pp['plot']:
            d['viirs_png_utm'] = misc.imread(os.path.join(fp.path_to_viirs_sdr_resampled_peat, plume.filename))
    except Exception, e:
        logger.info('Could not load AOD data with error: ' + str(e))
        return None

    # set up resampler
    utm_rs = utm_resampler(orac_aod_data.variables['lat'][:],
                              orac_aod_data.variables['lon'][:],
                              constants.utm_grid_size)

    # get the mask for the lats and lons and apply
    orac_aod = extract_orac_aod(orac_aod_data)
    viirs_null_mask = np.ma.getmask(orac_aod)
    masked_lats = np.ma.masked_array(utm_rs.lats, viirs_null_mask)
    masked_lons = np.ma.masked_array(utm_rs.lons, viirs_null_mask)

    # resample all the datasets to UTM
    d['viirs_aod_utm'] = utm_rs.resample_image(extract_viirs_aod(viirs_aod_data), masked_lats, masked_lons, fill_value=0)
    d['viirs_flag_utm'] = utm_rs.resample_image(extract_viirs_flags(viirs_aod_data), masked_lats, masked_lons, fill_value=0)
    d['orac_aod_utm'] = utm_rs.resample_image(orac_aod, masked_lats, masked_lons, fill_value=0)
    d['orac_cost_utm'] = utm_rs.resample_image(extract_orac_cost(orac_aod_data), masked_lats, masked_lons, fill_value=0)
    d['lats'] = utm_rs.resample_image(utm_rs.lats, masked_lats, masked_lons, fill_value=0)
    d['lons'] = utm_rs.resample_image(utm_rs.lons, masked_lats, masked_lons, fill_value=0)
    return d


def setup_plume_data(plume, ds_utm):
    d = {}
    try:
        # get plume extent geographic data (bounding box in in UTM as plume extent is UTM)
        d['plume_bounding_box'] = construct_bounding_box(plume.plume_extent)
        d['plume_lats'] = subset_data(ds_utm['lats'], d['plume_bounding_box'])
        d['plume_lons'] = subset_data(ds_utm['lons'], d['plume_bounding_box'])

        # get plume vector geographic data
        vector_lats, vector_lons = extract_subset_geo_bounds(plume.plume_vector, d['plume_bounding_box'],
                                                                d['plume_lats'], d['plume_lons'])
        # get plume polygon geographic data
        poly_lats, poly_lons = extract_subset_geo_bounds(plume.plume_extent, d['plume_bounding_box'],
                                                            d['plume_lats'], d['plume_lons'])

        # get plume mask
        d['plume_mask'] = construct_mask(plume.plume_extent, d['plume_bounding_box'])

        # setup shapely objects for plume geo data
        d['plume_vector'] = construct_shapely_vector(vector_lats, vector_lons)
        d['plume_points'] = construct_shapely_points(poly_lats, poly_lons)
        d['plume_polygon'] = construct_shapely_polygon(poly_lats, poly_lons)

        d['background_bounding_box'] = construct_bounding_box(plume.background_extent)
        d['background_mask'] = construct_mask(plume.background_extent, d['background_bounding_box'])

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
    d['utm_plume_vector'] = reproject_shapely(plume_geom_geo['plume_vector'], d['utm_resampler_plume'])
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
    plume_sub_polygons = split_plume_polgons(utm_flow_means, plume_logging_path, plume,
                                                plume_geom_geo, plume_geom_utm, pp)

    # get the variables of interest
    if plume_sub_polygons:

        for sub_p_number, sub_polygon in plume_sub_polygons.iteritems():

            sub_plume_logging_path = os.path.join(plume_logging_path, str(sub_p_number))
            if not os.path.isdir(sub_plume_logging_path):
                os.mkdir(sub_plume_logging_path)

            # make mask for sub polygon
            sub_plume_mask = sub_mask(plume_geom_geo['plume_lats'].shape,
                                         sub_polygon,
                                         plume_geom_geo['plume_mask'])

            # make polygon for sub_polygon and intersect with plume polygon
            bounding_lats, bounding_lons = extract_geo_bounds(sub_polygon,
                                                                 plume_geom_geo['plume_lats'],
                                                                 plume_geom_geo['plume_lons'])
            sub_plume_polygon = construct_shapely_polygon(bounding_lats, bounding_lons)
            utm_sub_plume_polygon = reproject_shapely(sub_plume_polygon, plume_geom_utm['utm_resampler_plume'])

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
    out_dict = tt.compute_tpm_full(plume_data_utm, plume_geom_utm, plume_geom_geo, bg_aod_dict, plume_logging_path, pp)
    out_dict['plume_number'] = p_number

    # compute fre
    ff.compute_fre_full_plume(t1, t2, pp['frp_df'], plume_geom_utm, plume_logging_path, out_dict)

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






def find_landcover_class(lat_list, lon_list, landcover_ds):

    # now get the landcover points
    lc_list = []
    for lat, lon in zip(lat_list, lon_list):
        s = int((lon - (-180)) / 360 * landcover_ds['lon'].size)  # lon index
        l = int((lat - 90) * -1 / 180 * landcover_ds['lat'].size)  # lat index

        # image is flipped, so we need to reverse the lat coordinate
        l = -(l + 1)

        lc_list.append(np.array(landcover_ds['lccs_class'][(l - 1):l, s:s + 1][0])[0])

    # return the most common landcover class for the fire contined in the ROI
    return stats.mode(lc_list).mode[0]