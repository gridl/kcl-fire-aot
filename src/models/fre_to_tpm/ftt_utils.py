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
from functools import partial
import math

import pandas as pd
import numpy as np
from netCDF4 import Dataset
from pyhdf.SD import SD, SDC
from matplotlib.path import Path
from scipy import stats
from scipy import integrate
from scipy import ndimage
from shapely.geometry import Polygon, Point, MultiPoint
from shapely.ops import transform
import pyresample as pr
import pyproj
import cv2

import src.data.readers.load_hrit as load_hrit
import src.config.filepaths as fp


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


def reproject_shapely(plume_polygon, utm_resampler):

    project = partial(
        pyproj.transform,
        pyproj.Proj(init='epsg:4326'),  # source coordinate system (geographic coords)
        utm_resampler.utm_transform)  # destination coordinate system

    return transform(project, plume_polygon)  # apply projection


def hist_eq(im,nbr_bins=256):

    #get image histogram
    imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize

    #use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(),bins[:-1],cdf)

    return im2.reshape(im.shape).astype('uint8')


class utm_resampler(object):
    def __init__(self, lats, lons, pixel_size, resolution=0.01):
        self.lats = lats
        self.lons = lons
        self.pixel_size = pixel_size
        self.resolution = resolution
        self.utm_transform = self.__utm_transform()
        self.area_def = self.__utm_area_def()

    def __utm_zone(self):
        '''
        Some of the plumes will cross UTM zones.  This is not problematic
        as the plumes are quite small and so, we can just use the zone
        in which most of the data falls: https://goo.gl/3QY2Re
        see also: http://www.igorexchange.com/node/927 for if we need over Svalbard (highly unlikely)
        '''
        lons = (self.lons + 180) - np.floor((self.lons + 180) / 360) * 360 - 180
        return stats.mode(np.floor((lons + 180) / 6) + 1, axis=None)[0][0]

    def __utm_proj(self, zone):
        return pyproj.Proj(proj='utm', zone=zone, ellps='WGS84', datum='WGS84')

    def __utm_area_extent(self, zone):
        p = self.__utm_proj(zone)
        x, y = p(self.lons, self.lats)
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
        return {'max_x': max_x, 'min_x': min_x, 'max_y': max_y, 'min_y': min_y}

    def __utm_grid_size(self, utm_boundaries):
        x_size = int(np.ceil((utm_boundaries['max_x'] - utm_boundaries['min_x']) / self.pixel_size))
        y_size = int(np.ceil((utm_boundaries['max_y'] - utm_boundaries['min_y']) / self.pixel_size))
        return x_size, y_size

    def __construct_area_def(self, zone, utm_boundaries, x_size, y_size):
        area_id = 'utm'
        description = 'utm_grid'
        proj_id = 'utm'
        area_extent = (utm_boundaries['min_x'], utm_boundaries['min_y'],
                       utm_boundaries['max_x'], utm_boundaries['max_y'])
        proj_dict = {'units': 'm', 'proj': 'utm', 'zone': str(zone), 'ellps': 'WGS84', 'datum': 'WGS84'}
        return pr.geometry.AreaDefinition(area_id, description, proj_id, proj_dict, x_size, y_size, area_extent)

    def __utm_area_def(self):
        zone = self.__utm_zone()
        area_extent = self.__utm_area_extent(zone)
        x_size, y_size = self.__utm_grid_size(area_extent)
        return self.__construct_area_def(zone, area_extent, x_size, y_size)

    def __utm_transform(self):
        zone = self.__utm_zone()
        return self.__utm_proj(zone)

    def resample(self, image, image_lats, image_lons):
        swath_def = pr.geometry.SwathDefinition(lons=image_lons, lats=image_lats)
        return pr.kd_tree.resample_nearest(swath_def,
                                           image,
                                           self.area_def,
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


#########################    INTEGRATION TIME   #########################


def compute_plume_length(plume_points):

    '''
    From the minimum rotated rectangle that encloses the digitised points
    we can compute the plumes length from its vertices.  This simply
    involves computing the euclidean distance between the verticies, which
    is possible as we are working with projected data.

    Also, this means that we need to digitise plumes that are longer than wide

    :param plume_points: the set of points that comprise the plume polygon
    :return: length of the longest side
    '''

    x, y = plume_points.minimum_rotated_rectangle.exterior.xy
    verts = zip(x, y)

    side_a = np.linalg.norm(np.array(verts[0])-np.array(verts[1]))
    side_b = np.linalg.norm(np.array(verts[1])-np.array(verts[2]))

    return max([side_a, side_b])  # return max UTM distance in (m)


def geo_spatial_subset(lats_1, lons_1, lats_2, lons_2):
    '''

    :param lats_1: target lat region
    :param lons_1: target lon region
    :param lats_2: extended lat region
    :param lons_2: extended lon region
    :return bounds: bounding box locating l1 in l2
    '''

    padding = 25  # pixels

    min_lat = np.min(lats_1)
    max_lat = np.max(lats_1)
    min_lon = np.min(lons_1)
    max_lon = np.max(lons_1)

    coords = np.where((lats_2 >= min_lat) & (lats_2 <= max_lat) &
                      (lons_2 >= min_lon) & (lons_2 <= max_lon))

    min_x = np.min(coords[1])
    max_x = np.max(coords[1])
    min_y = np.min(coords[0])
    max_y = np.max(coords[0])

    return {'max_x': max_x+padding,
            'min_x': min_x-padding,
            'max_y': max_y+padding,
            'min_y': min_y-padding}


def find_image_segment(bb):

    # there are ten 2200 pixel segments in himawari
    seg_size = 2200
    segment_min = bb['min_y'] / seg_size
    segment_max = bb['max_y'] / seg_size

    if segment_min == segment_max:
        return segment_min
    else:
        logger.critical('Plumes crossing multiple himawari image segments not yet implemented')
        return None


def adjust_bb_for_segment(bb, segment):
    seg_size = 2200
    bb['min_y'] -= (segment * seg_size)
    bb['max_y'] -= (segment * seg_size)


def get_plume_time(plume_fname):
    return datetime.strptime(plume_fname[10:21], '%Y%j.%H%M')


def get_geostationary_fnames(plume_time, image_segment):

    ym = str(plume_time.year) + str(plume_time.month).zfill(2)
    day = str(plume_time.day).zfill(2)

    # get all files in the directory using glob with band 3 for main segment
    p = os.path.join(fp.path_to_himawari_l1b, ym, day)
    fp_1 = glob.glob(p + '/*/*/B03/*S' + str(image_segment).zfill(2) + '*')

    # get the day before also
    day = str(plume_time.day - 1).zfill(2)
    p = os.path.join(fp.path_to_himawari_l1b, ym, day)
    fp_2 = glob.glob(p + '/*/*/B03/*S' + str(image_segment).zfill(2) + '*')

    files = fp_1 + fp_2

    return files


def restrict_geostationary_times(plume_time, geostationary_fnames):
    return [f for f in geostationary_fnames if datetime.strptime(f.split('/')[-1][7:20], '%Y%m%d_%H%M') < plume_time]


def find_integration_start_stop_times(plume_fname,
                                      plume_points, plume_mask,
                                      plume_lats, plume_lons,
                                      geostationary_lats, geostationary_lons,
                                      utm_resampler_modis):

    # get plume observation time
    plume_time = get_plume_time(plume_fname)

    # find distance in plume polygon from fire head to tail
    total_plume_length = compute_plume_length(plume_points)

    # find geostationary bounding box from plume lats and lons
    bb = geo_spatial_subset(plume_lats, plume_lons, geostationary_lats, geostationary_lons)

    # extract lats and lons using bounding box
    geostationary_lats_subset = geostationary_lats[bb['min_y']:bb['max_y'], bb['min_x']:bb['max_x']]
    geostationary_lons_subset = geostationary_lons[bb['min_y']:bb['max_y'], bb['min_x']:bb['max_x']]

    # adjust pixel resolution of bouding box so that lats/lons, bb have the same size as the image for channel
    zoom = 4
    geostationary_lats_subset = ndimage.zoom(geostationary_lats_subset, zoom)
    geostationary_lons_subset = ndimage.zoom(geostationary_lons_subset, zoom)
    bb.update((x, y * zoom) for x, y in bb.items())  # enlarge bounding box by factor of zoom also

    # find the image segment related to the bb
    image_segment = find_image_segment(bb)

    # adjust the bb for the image segment
    adjust_bb_for_segment(bb, image_segment)

    # set up image reprojection object for geostationary imager using bounded lats and lons
    utm_resampler_geos = utm_resampler(geostationary_lats_subset,
                                       geostationary_lons_subset,
                                       pixel_size=500)

    # get the geographic coordinates for the utm geostationry grid
    utm_lons, utm_lats = utm_resampler_geos.area_def.get_lonlats()

    # get the geostationary filenames for the given plume time and image segment
    geostationary_fnames = get_geostationary_fnames(plume_time, image_segment)

    # reduce geostationary filenames to only those prior to plume observation
    geostationary_fnames = restrict_geostationary_times(plume_time, geostationary_fnames)

    # reverse the filename list
    geostationary_fnames.reverse()

    # set up stopping condition which is the current estimate of the plume length
    current_plume_length = 0

    # iterate over geostationary files
    for f1, f2 in zip(geostationary_fnames[:-1], geostationary_fnames[1:]):

        # load geostationary files
        f1_radiances, _ = load_hrit.H8_file_read(os.path.join(fp.path_to_himawari_l1b, f1))
        f2_radiances, _ = load_hrit.H8_file_read(os.path.join(fp.path_to_himawari_l1b, f2))

        # extract geostationary image subset using adjusted bb
        f1_radiances_subset = f1_radiances[bb['min_y']:bb['max_y'], bb['min_x']:bb['max_x']]
        f2_radiances_subset = f2_radiances[bb['min_y']:bb['max_y'], bb['min_x']:bb['max_x']]

        # equalise the image
        f1_radiances_subset = hist_eq(f1_radiances_subset)
        f2_radiances_subset = hist_eq(f2_radiances_subset)

        # reproject image subset to UTM using resampler
        f1_radiances_subset_reproj = utm_resampler_geos.resample(f1_radiances_subset,
                                                                 geostationary_lats_subset,
                                                                 geostationary_lons_subset)
        f2_radiances_subset_reproj = utm_resampler_geos.resample(f2_radiances_subset,
                                                                 geostationary_lats_subset,
                                                                 geostationary_lons_subset)

        # compute optical flow between two images
        flow_image = np.zeros(f1_radiances_subset_reproj.shape).astype('uint8')
        flow = cv2.calcOpticalFlowFarneback(f1_radiances_subset_reproj,
                                            f2_radiances_subset_reproj,
                                            flow_image,
                                            0.5, 4, 50, 10, 7, 1.5,
                                            cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        # compute distances using magnitude
        flow_x = flow[:, :, 0]
        flow_y = flow[:, :, 1]
        pixel_distances = np.sqrt(flow_x ** 2 + flow_y ** 2)
        utm_distances = pixel_distances * 500  # 500m imager resolution

        # now resample distances onto plume
        distances_in_modis_proj = utm_resampler_modis.resample(utm_distances, utm_lats, utm_lons)

        # smoke mask needs to be applied to distances, else we might get non plume features contributing

        # extract median distance travelled for plume using plume mask
        median_distance = np.median(distances_in_modis_proj[plume_mask])

        # sum distance with total distance
        current_plume_length += median_distance
        if current_plume_length > total_plume_length:
            return datetime.strptime(f2.split('/')[-1][7:20], '%Y%m%d_%H%M')  # return time of the second file

    return None


#########################    TPM    #########################

# TODO still need to make sure area is being calculated correctly
def compute_plume_area(utm_plume_polygon):


    # # TODO is sinusoidal proj good enough?  Yes it is: https://goo.gl/KE3tuY
    # # get extra accuracy by selecting an appropriate lon_0
    # m = Basemap(projection='sinu', lon_0=140, resolution='c')
    #
    # lons = (lons + 180) - np.floor((lons + 180) / 360) * 360 - 180;
    # zone = stats.mode(np.floor((lons + 180) / 6) + 1, axis=None)[0][0]
    # p = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84', datum='WGS84')
    #
    # # apply to shapely polygon
    # projected_plume_polygon_m = transform(m, plume_polygon)
    # projected_plume_polygon_p = transform(p, plume_polygon)

    # compute projected polygon area in m2
    # return projected_plume_polygon_m.area, projected_plume_polygon_p.area

    # we already have the plume polygon area
    return utm_plume_polygon.area


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

    # create best AOD map from ORAC and MYD04 AOD

    # extract best AOD using plume mask

    # split into background and plume AODs

    # subtract background from plume

    # return plume AOD

