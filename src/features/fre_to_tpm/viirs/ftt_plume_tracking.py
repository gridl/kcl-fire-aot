# load in required packages
import glob
import os
from datetime import datetime
import logging
import re

import numpy as np
from scipy import ndimage
import pandas as pd
import cv2
from shapely.geometry import Polygon, Point, LineString

import src.data.readers.load_hrit as load_hrit
import src.config.filepaths as fp
import src.visualization.ftt_visualiser as vis
import src.features.fre_to_tpm.viirs.ftt_fre as ff
import src.features.fre_to_tpm.viirs.ftt_utils as ut
import src.config.constants as constants

import matplotlib.pyplot as plt

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def get_plume_time(timestamp):
    """
    :param plume_fname: the MYD filename for the plume
    :return: the plume time stamp
    """
    return datetime.strptime(timestamp, 'd%Y%m%d_t%H%M%S')


def find_plume_head(plume_geom_geo, plume_geom_utm, pp, t):

    fire_locations_in_plume = ff.fire_locations_for_plume_roi(plume_geom_geo, pp['frp_df'], t)

    mean_fire_lon = np.mean([i.x for i in fire_locations_in_plume])
    mean_fire_lat = np.mean([i.y for i in fire_locations_in_plume])

    # project to utm
    mean_fire_utm = ut.reproject_shapely(Point(mean_fire_lon, mean_fire_lat),
                                         plume_geom_utm['utm_resampler_plume'])

    return {'head_lon': mean_fire_lon,
            'head_lat': mean_fire_lat,
            'head': mean_fire_utm}


def find_tail_edge(head, plume_geom_utm):

    # convex hull of plume
    x, y = plume_geom_utm['utm_plume_points'].minimum_rotated_rectangle.exterior.xy

    # get parallel edges of convex hull
    edge_pair_a = [LineString([(x[0], y[0]), (x[1], y[1])]),
                   LineString([(x[2], y[2]), (x[3], y[3])])]
    edge_pair_b = [LineString([(x[1], y[1]), (x[2], y[2])]),
                   LineString([(x[3], y[3]), (x[4], y[4])])]

    # distance between edges and head of fire
    distances_pair_a = [head.distance(i) for i in edge_pair_a]
    distances_pair_b = [head.distance(i) for i in edge_pair_b]

    # compute the ratios between the distances.  A larger ratio will
    # indicate the longest axis (perhaps change this to most extreme ratio)
    ratios_pair_a = np.divide(distances_pair_a, distances_pair_a[::-1])
    ratios_pair_b = np.divide(distances_pair_b, distances_pair_b[::-1])

    # find the largest ratio
    argmax_a = np.argmax(ratios_pair_a)
    argmax_b = np.argmax(ratios_pair_b)

    # select side most distant from plume for side pair with largest ratio
    if ratios_pair_a[argmax_a] > ratios_pair_b[argmax_b]:
        return edge_pair_a[np.argmax(distances_pair_a)]
    else:
        return edge_pair_b[np.argmax(distances_pair_a)]


def find_plume_tail(head, plume_geom_utm, plume_geom_geo):

    # find tail edge
    tail_edge = find_tail_edge(head, plume_geom_utm)

    # check all pixels on tail edge.  If all background, then
    # plume finishes before edge.  Iterate over lat/lon in
    # plume, project and compute distance to line
    aods = []
    flags = []
    flat_lats = plume_geom_geo['plume_lats'].flatten()
    flat_lons = plume_geom_geo['plume_lons'].flatten()
    flat_aods = plume_geom_geo['plume_aod'].flatten()
    flat_flags = plume_geom_geo['plume_flag'].flatten()
    for i, (lon, lat) in enumerate(zip(flat_lons, flat_lats)):
        utm_point = ut.reproject_shapely(Point(lon, lat), plume_geom_utm['utm_resampler_plume'])
        dist_tail = utm_point.distance(tail_edge)
        if dist_tail < 325: # half a viirs pixel
            # get the aods and flags that intersect the line
            aods.append(flat_aods[i])
            flags.append(flat_flags[i])

    # select appropriate processing to determine if plume
    # finishes on edge or not.
    if np.min(flags) <= 1:
        min_test = np.min(flags) <= 1
        bg_mask = plume_geom_geo['bg_flag'] <= 1
        bg_aod = plume_geom_geo['bg_aod'][bg_mask]
        aod_test = np.mean(aods[flags <= 1]) <= (np.mean(bg_aod) + 2*np.std(bg_aod))
    else:
        min_test = False
        aod_test = False

    if min_test and aod_test:
        # in this instance the plume does not intersect with the end of
        # bounding box.  So we need to find the pixel furthest from the
        # head of the plume that is a bg pixel, and assume that that is tail
        bg_mask = flat_aods < np.mean(bg_aod) + 2*np.std(bg_aod)
        max_dist = 0
        tail_lat = 0
        tail_lon = 0
        tail = Point(0, 0)
        for i, (lon, lat) in enumerate(zip(flat_lons, flat_lats)):
            if bg_mask[i]:
                continue
            utm_point = ut.reproject_shapely(Point(lon, lat), plume_geom_utm['utm_resampler_plume'])
            dist_head = utm_point.distance(head)
            if dist_head > max_dist:
                max_dist = dist_head
                tail = utm_point
                tail_lat = lat
                tail_lon = lon
    else:
        # if the plume does intersect, the tail in then the midpoint of the
        # edge
        tail = Point(np.sum(tail_edge.xy[0]) /2, np.sum(tail_edge.xy[1]) /2)
        tail_lon, tail_lat = plume_geom_utm['utm_resampler_plume'].resample_point_to_geo(tail.y, tail.x)

    return {'tail_lon': tail_lon,
            'tail_lat': tail_lat,
            'tail': tail}


def compute_plume_vector(plume_geom_geo, plume_geom_utm, pp, t):
    # first set up the two alternative head and tail combintations
    # second cehck if one of the heads is outside of the bounding polygon
    # if both inside find the orientation of the rectangle

    # tail = np.array(pv.coords[0])
    # head = np.array(pv.coords[1])

    head_dict = find_plume_head(plume_geom_geo, plume_geom_utm, pp, t)
    tail_dict = find_plume_tail(head_dict['head'],
                                     plume_geom_utm, plume_geom_geo)
    vect = np.array(head_dict['head'].coords[0]) - np.array(tail_dict['tail'].coords)

    return head_dict, tail_dict, vect


def spatial_subset(lats_1, lons_1, lats_2, lons_2):
    """
    :param lats_1: target lat region
    :param lons_1: target lon region
    :param lats_2: extended lat region
    :param lons_2: extended lon region
    :return bounds: bounding box locating l1 in l2
    """

    padding = 100  # pixels  TODO add to config

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

    return {'max_x': max_x + padding,
            'min_x': min_x - padding,
            'max_y': max_y + padding,
            'min_y': min_y - padding}


def subset_geograpic_data(geostationary_lats, geostationary_lons, bb):
    """

    :param geostationary_lats: the lat image
    :param geostationary_lons: the lon image
    :param bb: the plume bounding box
    :return: the lats and lons for the bounding box
    """
    geostationary_lats_subset = geostationary_lats[bb['min_y']:bb['max_y'], bb['min_x']:bb['max_x']]
    geostationary_lons_subset = geostationary_lons[bb['min_y']:bb['max_y'], bb['min_x']:bb['max_x']]

    zoom = 4  # zoom if using 0.5km himawara data (B03) for tracking
    geostationary_lats_subset = ndimage.zoom(geostationary_lats_subset, zoom)
    geostationary_lons_subset = ndimage.zoom(geostationary_lons_subset, zoom)
    bb.update((x, y * zoom) for x, y in bb.items())  # enlarge bounding box by factor of zoom also

    return geostationary_lats_subset, geostationary_lons_subset


def find_min_himawari_image_segment(bb):
    """
    :param bb: bounding box
    :return: the himawari image segment for the given bounding box
    """
    # there are ten 2200 pixel segments in himawari 0.5 km data
    seg_size = 2200
    min_segment = bb['min_y'] / seg_size + 1
    return min_segment


def adjust_bb_for_segment(bb, segment):
    """
    :param bb: plume bounding box
    :param segment: the image segment that contains the bounding box
    :return: Nothing, the boudning box coordinates are adjusted inplace
    """
    seg_size = 2200
    bb['min_y'] -= (segment * seg_size)
    bb['max_y'] -= (segment * seg_size)


def get_geostationary_fnames(plume_time, image_segment):
    """

    :param plume_time: the time of the MYD observation of the plume
    :param image_segment: the Himawari image segment
    :return: the geostationary files for the day of and the day before the fire
    """
    ym = str(plume_time.year) + str(plume_time.month).zfill(2)
    day = str(plume_time.day).zfill(2)

    # get all files in the directory using glob with band 3 for main segment
    p = os.path.join(fp.path_to_himawari_imagery, ym, day)
    fp_1 = glob.glob(p + '/*/*/B03/*S' + str(image_segment).zfill(2) + '*')

    # get the day before also
    day = str(plume_time.day - 1).zfill(2)
    p = os.path.join(fp.path_to_himawari_imagery, ym, day)
    fp_2 = glob.glob(p + '/*/*/B03/*S' + str(image_segment).zfill(2) + '*')

    files = fp_1 + fp_2

    return files


def restrict_geostationary_times(plume_time, geostationary_fnames):
    """

    :param plume_time: the plume time
    :param geostationary_fnames: the list of geostationary file names
    :return: only those goestationary files that were obtained prior to the myd overpass
    """
    return [f for f in geostationary_fnames if
            datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", f).group(),
                              '%Y%m%d_%H%M') <= plume_time]


def sort_geostationary_by_time(geostationary_fnames):
    """

    :param geostationary_fnames goestationary filenames
    :return: the geostationary filenames in time order
    """
    times = [datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", f).group()
                               , '%Y%m%d_%H%M') for f in geostationary_fnames]
    return [f for _, f in sorted(zip(times, geostationary_fnames))]


def setup_geostationary_files(plume_time, image_segment):
    geostationary_fnames = get_geostationary_fnames(plume_time, image_segment)
    geostationary_fnames = restrict_geostationary_times(plume_time, geostationary_fnames)
    geostationary_fnames = sort_geostationary_by_time(geostationary_fnames)
    geostationary_fnames.reverse()
    return geostationary_fnames


def rescale_image(image, display_min, display_max):
    '''

    :param image: image to rescale
    :param display_min: image min
    :param display_max: image max
    :return: the image scaled to 8bit
    '''
    image -= display_min
    np.floor_divide(image, (display_max - display_min + 1) / 256,
                    out=image, casting='unsafe')
    return image.astype(np.uint8)


def normalise_image(im):
    eps = 0.001  # to prevent div by zero
    sig = 1  # standard deviation of gaussian  TODO add to config file
    mean_im = ndimage.filters.gaussian_filter(im, sig)
    sd_im = np.sqrt(ndimage.filters.gaussian_filter((im - mean_im) ** 2, sig))
    return (im - mean_im) / (sd_im + eps)


def extract_observation(f, bb, segment):
    # load geostationary files for the segment
    rad_segment_1, _ = load_hrit.H8_file_read(os.path.join(fp.path_to_himawari_imagery, f))

    # load for the next segment
    f_new = f.replace('S' + str(segment).zfill(2), 'S' + str(segment + 1).zfill(2))
    rad_segment_2, _ = load_hrit.H8_file_read(os.path.join(fp.path_to_himawari_imagery, f_new))

    # concat the himawari files
    rad = np.vstack((rad_segment_1, rad_segment_2))

    # extract geostationary image subset using adjusted bb and rescale to 8bit
    rad_bb = rad[bb['min_y']:bb['max_y'], bb['min_x']:bb['max_x']]

    # normalise and rescale
    rad_bb_norm = normalise_image(rad_bb)
    rad_bb_norm = rescale_image(rad_bb_norm, rad_bb_norm.min(), rad_bb_norm.max())
    return rad_bb_norm, rad_bb


def feature_detector(fast, im, tracks):
    points = fast.detect(im, None)
    if points is not None:
        for pt in points:
            tracks.append([pt.pt])


def find_good_tracks(p0, p1, p0r):
    # lets check the points and consistent in both match directions and keep only those
    # points with a pixel shift greater than the minimum limit
    min_pix_shift = 1  # in pixels  TODO move to config
    bad_thresh = 1  # in pixels  TODO move to config
    d = abs(p0 - p0r).reshape(-1, 2).max(-1)
    d1 = abs(p0 - p1).reshape(-1, 2).max(-1)  # gets the max across all
    good = (d < bad_thresh) & (d1 >= min_pix_shift)
    return good


def update_tracks(tracks, p1, good):
    new_tracks = []
    for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
        if not good_flag:
            continue
        tr.append((x, y))
        new_tracks.append(tr)
    return new_tracks


def adjust_image_map_coordinates(flow):
    """
    The image flow is returned in image coordinate space.  The derived vectors
    that describe the flow cannot therefore be used as is for calculating the
    flow on the map.  They need to be adjusted to the map space.  As everything
    is projected onto a UTM grid this is relatively straightforward, we just need
    to invert the y axis (as the image and map coordinate systems are in effect
    inverted).

    :param flow: the image flow
    :return: the adjusted image flow
    """
    flow[:, 1] *= -1
    return flow


def compute_flow(tracks, im1, im2):
    """

    :param tracks: the pionts that are being tracked across the images
    :param im1: the for which the feature have been generated
    :param im2: the image to be matched to
    :return: the flow in the map projection
    """

    # TODO move this to a config file
    # set up feature detection and tracking parameters
    lk_params = {'winSize': (10, 10), 'maxLevel': 3,
                 'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)}

    # track the features
    p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)  # the feature points to be tracked
    p1, _st, _err = cv2.calcOpticalFlowPyrLK(im1, im2,
                                             p0, None, **lk_params)  # the shifted points
    # back match the featrues
    try:
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(im2, im1,
                                                  p1, None, **lk_params)  # matching in other direction
        good = find_good_tracks(p0, p1, p0r)
        tracks = update_tracks(tracks, p1, good)
    except Exception, e:
        print 'Could not get back match tracks with error:', str(e)
        return tracks, []

    if tracks:
        # subtract the feature locations in the second image from the first image
        # in effect this is im2 - im1.  Could perhaps make the code clearer here
        flow = (p0 - p1).reshape(-1, 2)[good]
        flow = adjust_image_map_coordinates(flow)
        return tracks, flow
    else:
        return tracks, []


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def assess_flow(flow, flow_means, flow_sds, flow_nobs, flow_update, tracks, i, current_vector, resampled_pix_size):
    """

    :param current_vector: the current flow vector for checking angle of flow vectors
    :param flow: the current flowfc vectors from the feature tracking
    :param flow_means: the vector containing the flow means
    :param flow_sds: the vector containing the flow standard deviations
    :param flow_nobs: the vector containing the number of flow observations
    :param flow_update: vector containing mean used to to update in case of tracks <= min tracks
    :param tracks: the track points
    :param i: the current index
    :return: None
    """

    # only keep flow within certain angular threshold of current flow vector
    flow_mask = np.zeros(len(flow)).astype('bool')
    new_tracks = []
    for f, (flow_vector, track) in enumerate(zip(flow, tracks)):
        angle = angle_between(flow_vector, current_vector)
        if (angle <= constants.angular_limit) | (angle >= (2 * np.pi) - constants.angular_limit):
            flow_mask[f] = 1
            new_tracks.append(track)

    tracks = new_tracks

    if len(tracks) <= constants.min_number_tracks:
        if i != 0:
            flow_means[i] = flow_means[i - 1]
            flow_sds[i] = flow_sds[i - 1]
            flow_update[i] = i - 1
    else:
        # here is where the flow update occurs
        flow = flow[flow_mask]

        flow_means[i, :] = np.mean(flow, axis=0) * resampled_pix_size
        flow_sds[i, :] = np.std(flow, axis=0) * resampled_pix_size
        flow_nobs[i] = len(tracks)

    # check if any flows of zero and update with most recent flow estimate
    # even if most recent estimate is zero, they will still get updated
    # at some point
    if (flow_means[:i] == 0).any():
        mask = [flow_means[:i, 0] == 0]  # can take either  x or y, as both will have zero entries

        # update x values
        flow_means[:i, 0][mask] = flow_means[i, 0]
        flow_sds[:i, 0][mask] = flow_sds[i, 0]

        # update y values
        flow_means[:i, 1][mask] = flow_means[i, 1]
        flow_sds[:i, 1][mask] = flow_sds[i, 1]

        flow_update[:i][mask] = i


def project_flow(plume_vector, flow_means, projected_flow_means, projected_flow_magnitudes, i):
    """

    :param plume_vector: the vector of the plume
    :param flow_means: the mean flows computed for the vector
    :param projected_flow_magnitude: the vector holding the projected mean flows
    :param i: the number of current observations we have
    :return: nothing, projected flows updated in place
    """
    for obs in np.arange(i + 1):
        projected_flow_vector = np.dot(plume_vector, flow_means[obs]) / \
                                np.dot(plume_vector, plume_vector) * plume_vector
        projected_flow_means[obs] = projected_flow_vector
        projected_flow_magnitudes[obs] = np.linalg.norm(projected_flow_vector)


# functions used in plotting

def geographic_extent(geostationary_lats, geostationary_lons, bb):
    extent = [(bb['min_y'], bb['min_x']),
              (bb['min_y'], bb['max_x']),
              (bb['max_y'], bb['max_x']),
              (bb['max_y'], bb['min_x'])]

    bounding_lats = [geostationary_lats[point[0], point[1]] for point in extent]
    bounding_lons = [geostationary_lons[point[0], point[1]] for point in extent]
    return bounding_lats, bounding_lons


def dump_tracking_data(i, flow_means, projected_flow_means, flow_sds,
                       projected_flow_magnitude, flow_nobs, flow_update,
                       geostationary_fnames, plume_logging_path, p_number):
    data = np.array([flow_means[:i + 1, 0], flow_means[:i + 1, 1],
                     projected_flow_means[:i + 1, 0], projected_flow_means[:i + 1, 1],
                     flow_sds[:i + 1, 0], flow_sds[:i + 1, 1],
                     projected_flow_magnitude[:i + 1],
                     flow_nobs[:i + 1], flow_update[:i + 1]]).T
    columns = ['flow_means_x', 'flow_means_y', 'proj_flow_means_x', 'proj_flow_means_y',
               'flow_sds_x', 'flow_sds_y', 'proj_flow_mag', 'flow_nobs', 'flow_update']
    df = pd.DataFrame(data, index=[f.split('/')[-1] for f in geostationary_fnames[:i + 1]], columns=columns)
    df.to_csv(os.path.join(plume_logging_path, str(p_number) + '_tracks.csv'))


def find_flow(p_number, plume_logging_path, plume_geom_utm, plume_geom_geo, pp, timestamp):

    # get bounding box around smoke plume in geostationary imager coordinates
    # and extract the geographic coordinates for the roi, also set up plot stuff
    bbox = spatial_subset(plume_geom_geo['plume_lats'], plume_geom_geo['plume_lons'],
                          pp['geostationary_lats'], pp['geostationary_lons'])

    geostationary_lats_subset, geostationary_lons_subset = subset_geograpic_data(pp['geostationary_lats'],
                                                                                 pp['geostationary_lons'],
                                                                                 bbox)

    min_geo_segment = find_min_himawari_image_segment(bbox)
    adjust_bb_for_segment(bbox, min_geo_segment - 1)

    plume_time = get_plume_time(timestamp)
    geostationary_fnames = setup_geostationary_files(plume_time, min_geo_segment)

    # establish plume vector, and importantly the total plume length
    t0 = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[0]).group(),
                           '%Y%m%d_%H%M')
    plume_head, plume_tail, vector = compute_plume_vector(plume_geom_geo, plume_geom_utm,
                                                          pp, t0)
    length = np.linalg.norm(vector)  # plume length in metres

    # set up feature detector
    fast = cv2.FastFeatureDetector_create(threshold=constants.fast_threshold)

    # plotting stuff
    if pp['plot']:
        # set up this polygon so we can see all fires near to the plume
        bounding_lats, bounding_lons = geographic_extent(pp['geostationary_lats'], pp['geostationary_lons'], bbox)
        geo_polygon = Polygon(zip(bounding_lons, bounding_lats))
        fires = []

    # set up iteration
    flow_means, flow_sds, projected_flow_means = np.zeros([72, 2]), np.zeros([72, 2]), np.zeros([72, 2])
    flow_nobs, flow_update, projected_flow_magnitude = np.zeros([72]), np.zeros([72]), np.zeros(72)
    tracks = []
    stopping_thresh = constants.utm_grid_size  # stopping when within one pix
    current_vector = vector.copy()
    for i in xrange(len(geostationary_fnames) - 1):

        # setup imagery for tracking, only reproject both images if we are on the
        # first iteration, else just reassign and reproject most recent.  They are
        # reprojected to the plume so will have the same resolution as the plume resample
        if i == 0:
            f1_subset, f1_display_subset = extract_observation(geostationary_fnames[i], bbox, min_geo_segment)
            f2_subset, f2_display_subset = extract_observation(geostationary_fnames[i + 1], bbox, min_geo_segment)

            # subset to the plume
            f1_subset_reproj = plume_geom_utm['utm_resampler_plume'].resample_image(f1_subset,
                                                                                 geostationary_lats_subset,
                                                                                 geostationary_lons_subset)
            f2_subset_reproj = plume_geom_utm['utm_resampler_plume'].resample_image(f2_subset,
                                                                                 geostationary_lats_subset,
                                                                                 geostationary_lons_subset)
        else:
            f1_subset_reproj, f1_display_subset = f2_subset_reproj, f2_display_subset
            f2_subset, f2_display_subset = extract_observation(geostationary_fnames[i + 1], bbox, min_geo_segment)

            # subset to the plume
            f2_subset_reproj = plume_geom_utm['utm_resampler_plume'].resample_image(f2_subset,
                                                                                 geostationary_lats_subset,
                                                                                 geostationary_lons_subset)

        # if we are plotting do this stuff
        if pp['plot'] & (i == 0):
            plot_images = [plume_geom_utm['utm_resampler_plume'].resample_image(f1_display_subset,
                                                                             geostationary_lats_subset,
                                                                             geostationary_lons_subset)]
            fnames = [geostationary_fnames[i]]
        if pp['plot']:
            t = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[0]).group(),
                                  '%Y%m%d_%H%M')
            fires.append(ff.fire_locations_for_plume_roi(plume_geom_geo, pp['frp_df'], t))
            plot_images.append(plume_geom_utm['utm_resampler_plume'].resample_image(f2_display_subset,
                                                                                 geostationary_lats_subset,
                                                                                 geostationary_lons_subset))
            fnames.append(geostationary_fnames[i + 1])

        # detect features and compute flow
        feature_detector(fast, f2_subset_reproj, tracks)  # tracks updated inplace
        tracks, flow = compute_flow(tracks, f2_subset_reproj, f1_subset_reproj)

        # compute robust mean flow for plume and update current flow vector
        assess_flow(flow, flow_means, flow_sds, flow_nobs, flow_update, tracks, i, current_vector,
                    constants.utm_grid_size)
        if (flow_means[i] != 0).any():
            current_vector = flow_means[i, :]

        # now project mean flow vector onto plume vector to get flow projected along plume direction
        project_flow(vector, flow_means, projected_flow_means, projected_flow_magnitude, i)

        # sum current plume length and compare with total plume length, if length reached, break out
        summed_length = projected_flow_magnitude.sum()
        if ((length - summed_length) < stopping_thresh) | (summed_length > length):

            # if plotting do this stuff
            if pp['plot']:
                # also need to get the fires for the last scene
                t = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[i+1]).group(),
                                      '%Y%m%d_%H%M')
                fires.append(ff.fire_locations_for_plume_roi(plume_geom_geo, pp['frp_df'], t))
            break

    # save tracking information
    dump_tracking_data(i, flow_means, projected_flow_means, flow_sds,
                       projected_flow_magnitude, flow_nobs, flow_update,
                       geostationary_fnames, plume_logging_path, p_number)

    # plot plume
    if pp['plot']:
        vis.run_plot(plot_images, fires, flow_means, projected_flow_means,
                     plume_head, plume_tail, plume_geom_utm['utm_plume_points'], plume_geom_utm['utm_resampler_plume'],
                     plume_logging_path, fnames, i)

    # get the plume start and stop times
    t1 = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[0]).group(),
                           '%Y%m%d_%H%M')
    t2 = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[i+1]).group(),
                           '%Y%m%d_%H%M')

    # return the projected flow means in UTM coords, and the list of himawari filenames asspocated with the flows
    return projected_flow_means[:i + 1], geostationary_fnames[:i + 1], t1, t2