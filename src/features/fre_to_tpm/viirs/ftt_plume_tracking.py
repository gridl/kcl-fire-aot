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
from matplotlib.lines import Line2D

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
    # if np.min(flags) <= 1:
    #     min_test = np.min(flags) <= 1
    #     bg_mask = plume_geom_geo['bg_flag'] <= 1
    #     bg_aod = plume_geom_geo['bg_aod'][bg_mask]
    #     aod_test = np.mean(aods[flags <= 1]) <= (np.mean(bg_aod) + 2*np.std(bg_aod))
    # else:
    #     min_test = False
    #     aod_test = False

    # if no
    if np.max(flags) == 3:
        # if the plume does intersect, the tail is point with the
        # least distance from the edge
        tail = tail_edge.interpolate(tail_edge.project(head))
        tail_lon, tail_lat = plume_geom_utm['utm_resampler_plume'].resample_point_to_geo(tail.y, tail.x)
        return {'tail_lon': tail_lon,
                'tail_lat': tail_lat,
                'tail': tail}
    else:
        # in this instance the plume does not intersect with the end of
        # bounding box.  So we can assume that the frp that produced the
        # observed plume all occurred since the last minimum.
        return None


def compute_plume_vector(plume_geom_geo, plume_geom_utm, pp, t):
    # first set up the two alternative head and tail combintations
    # second cehck if one of the heads is outside of the bounding polygon
    # if both inside find the orientation of the rectangle

    # tail = np.array(pv.coords[0])
    # head = np.array(pv.coords[1])

    head_dict = find_plume_head(plume_geom_geo, plume_geom_utm, pp, t)
    tail_dict = find_plume_tail(head_dict['head'],
                                     plume_geom_utm, plume_geom_geo)

    if tail_dict is None:
        return None, None, None

    vect = np.array(head_dict['head'].coords[0]) - np.array(tail_dict['tail'].coords)
    return head_dict, tail_dict, vect[0]


def spatial_subset(lats_1, lons_1, lats_2, lons_2):
    """
    :param lats_1: target lat region
    :param lons_1: target lon region
    :param lats_2: extended lat region
    :param lons_2: extended lon region
    :return bounds: bounding box locating l1 in l2
    """

    padding = 50  # pixels  TODO add to config

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
    lk_params = {'winSize': (20, 20), 'maxLevel': 4,
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
    # first norm the vectors
    eps = 0.0001
    v1 /= np.sqrt(((v1 + eps) ** 2).sum(-1))[..., np.newaxis]
    v2 /= np.sqrt(((v2 + eps) ** 2).sum(-1))[..., np.newaxis]

    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


def draw_flow(img, flow, step=1):
    plt.close('all')
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    # TODO figure out why I am having to invert the x coords
    #fx *= -1

    ax = plt.axes()
    ax.imshow(img, cmap='gray')
    # for l in lines:
    #     x1 = l[0][0]
    #     y1 = l[0][1]
    #     x2 = l[1][0]
    #     y2 = l[1][1]
        # vect = Line2D([x1, x2], [y1, y2], lw=1, color='black')
        # ax.add_line(vect)
    ax.quiver(x, y, fx, fy, scale=100, color='red')

    plt.show()
    # vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # cv2.polylines(vis, lines, 0, (0, 255, 0))
    # for (x1, y1), (_x2, _y2) in lines:
    #     cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)


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

    # establish plume vector, and the total plume length.  If the plume does not
    # intersect with the end of the polygon then we do not need to worry about
    # limiting the FRP integration times.  So just return the min and max geo times
    t0 = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[0]).group(),
                           '%Y%m%d_%H%M')
    #plume_head, plume_tail, plume_vector = compute_plume_vector(plume_geom_geo, plume_geom_utm, pp, t0)

    # debugging
    plume_head = {'head_lon': 104.1464,
                  'head_lat': -1.79205,
                  'head': Point(405058.3078391715, -198098.1352896034)}
    plume_tail = {'tail_lon': 104.080129898,
                  'tail_lat': -1.71434879993,
                  'tail': Point(397682.580420428, -189512.1929388661)}
    plume_vector = (np.array(plume_head['head'].coords[0]) - np.array(plume_tail['tail'].coords))[0]

    if plume_head is None:
        t1 = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[0]).group(),
                               '%Y%m%d_%H%M')
        t2 = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[-1]).group(),
                               '%Y%m%d_%H%M')
        return None, geostationary_fnames[:], t1, t2

    plume_length = np.linalg.norm(plume_vector)  # plume length in metres


    # no iteration over the Himawari imagery, just assume that the closest
    # image pair to the VIIRS overpass is representative of the boundary
    # layer atmopsheric motion for the duration from the fire to the edge
    # of the bounding box.  This has a key benefit in that motion from
    # confounding features (e.g. clouds passing over the plume) can be
    # avoided.
    _, f1_subset = extract_observation(geostationary_fnames[1], bbox, min_geo_segment)
    _, f2_subset = extract_observation(geostationary_fnames[2], bbox, min_geo_segment)

    # subset to the plume
    f1_subset_reproj = plume_geom_utm['utm_resampler_plume'].resample_image(f1_subset,
                                                                            geostationary_lats_subset,
                                                                            geostationary_lons_subset)
    f2_subset_reproj = plume_geom_utm['utm_resampler_plume'].resample_image(f2_subset,
                                                                            geostationary_lats_subset,
                                                                            geostationary_lons_subset)


    # prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
    flow = cv2.calcOpticalFlowFarneback(f2_subset_reproj, f1_subset_reproj, None, 0.5, 3, 11, 5, 7, 1.5,
                                        cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    # plt.imshow(f1_subset, cmap='gray')
    # plt.savefig('/Users/danielfisher/Desktop/im1.png', bbox_inches='tight')
    # plt.close()
    #
    # plt.imshow(f2_subset, cmap='gray')
    # plt.savefig('/Users/danielfisher/Desktop/im2.png', bbox_inches='tight')
    # plt.close()

    plume_mask = plume_geom_geo['plume_mask']
    # if plume mask not same shape as himawari subset them
    # adjust it to match.  Exact overlay doesn't matter as
    # we are looking for average plume motion
    if plume_mask.shape != f1_subset_reproj.shape:
        if plume_mask.size < f1_subset_reproj.size:
            ax0_diff = f1_subset_reproj.shape[0] - plume_mask.shape[0]
            ax0_pad = (ax0_diff,0)  # padding n_before, n_after.  n_after always zero
            ax1_diff = f1_subset_reproj.shape[1] - plume_mask.shape[1]
            ax1_pad = (ax1_diff,0)
            plume_mask = np.pad(plume_mask, (ax0_pad, ax1_pad), 'edge')
        else:
            plume_mask = plume_mask[:f1_subset_reproj.shape[0], :f1_subset_reproj.shape[1]]

    # mask flow to plume extent
    flow *= plume_mask[..., np.newaxis]
    #adjust_image_map_coordinates(flow)
    flow[:,:,0] *= -1

    angles = angle_between(flow.copy(), plume_vector)
    angular_mask = (angles <= constants.angular_limit) | (angles >= (2 * np.pi) - constants.angular_limit)
    flow *= angular_mask[..., np.newaxis]

    draw_flow(f1_subset_reproj, flow)











    # detect features and compute flow
    tracks = []
    feature_detector(fast, f2_subset_reproj, tracks)  # tracks updated inplace
    tracks, flow = compute_flow(tracks, f2_subset_reproj, f1_subset_reproj)

    good_flow = []
    for flow_vector in flow:
        angle = angle_between(flow_vector, plume_vector)
        if (angle <= constants.angular_limit) | (angle >= (2 * np.pi) - constants.angular_limit):
            good_flow.append(flow_vector)

    # compute  mean flow for plume
    flow_mean = np.mean(good_flow, axis=0) * constants.utm_grid_size
    flow_sd = np.std(good_flow, axis=0) * constants.utm_grid_size
    flow_nobs = len(good_flow)

    # project flow onto principle axis
    projected_flow = np.dot(plume_vector, flow_mean) / \
                     np.dot(plume_vector, plume_vector) * plume_vector

    # get the number of himawari time steps
    n_steps = int(plume_length / np.linalg.norm(projected_flow))

    # get the plume start and stop times
    t1 = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[0]).group(),
                           '%Y%m%d_%H%M')
    t2 = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[n_steps]).group(),
                           '%Y%m%d_%H%M')

    # plotting
    if pp['plot']:
        for i in xrange(n_steps+2):
            if i == 0:
                # for first iteration need to get the first two images and associated information
                _, f1_display_subset = extract_observation(geostationary_fnames[i], bbox, min_geo_segment)
                _, f2_display_subset = extract_observation(geostationary_fnames[i+1], bbox, min_geo_segment)
                plot_images = [plume_geom_utm['utm_resampler_plume'].resample_image(f1_display_subset,
                                                                                    geostationary_lats_subset,
                                                                                    geostationary_lons_subset),
                               plume_geom_utm['utm_resampler_plume'].resample_image(f2_display_subset,
                                                                                    geostationary_lats_subset,
                                                                                    geostationary_lons_subset)]
                # get files names
                fnames = [geostationary_fnames[i], geostationary_fnames[i + 1]]

                # get fires
                t = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[i]).group(),
                                      '%Y%m%d_%H%M')
                fires = [ff.fire_locations_for_plume_roi(plume_geom_geo, pp['frp_df'], t)]
                t = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[i+1]).group(),
                                      '%Y%m%d_%H%M')
                fires.append(ff.fire_locations_for_plume_roi(plume_geom_geo, pp['frp_df'], t))

            else:
                # for following iterations just need the new image i.e
                # f1_display_subset = f2_display_subset for posterity
                _, f2_display_subset = extract_observation(geostationary_fnames[i+1], bbox, min_geo_segment)
                plot_images.append(plume_geom_utm['utm_resampler_plume'].resample_image(f2_display_subset,
                                                                                        geostationary_lats_subset,
                                                                                        geostationary_lons_subset))
                fnames.append(geostationary_fnames[i+1])

                t = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[i+1]).group(),
                                  '%Y%m%d_%H%M')
                fires.append(ff.fire_locations_for_plume_roi(plume_geom_geo, pp['frp_df'], t))

        vis.run_plot(plot_images, fires, flow_mean, projected_flow,
                     plume_head, plume_tail, plume_geom_utm['utm_plume_points'], plume_geom_utm['utm_resampler_plume'],
                     plume_logging_path, fnames, i)

    # return the projected flow means in UTM coords, and the list of himawari filenames asspocated with the flows
    return projected_flow, geostationary_fnames[:n_steps], t1, t2