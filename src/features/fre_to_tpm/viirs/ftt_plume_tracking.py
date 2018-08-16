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
        return edge_pair_b[np.argmax(distances_pair_b)]


def find_plume_tail(head, plume_geom_utm, plume_geom_geo):

    # find tail edge
    tail_edge = find_tail_edge(head, plume_geom_utm)

    # check all pixels on tail edge.  If all background, then
    # plume finishes before edge.  Iterate over lat/lon in
    # plume, project and compute distance to line
    # aods = []
    # flags = []
    # flat_lats = plume_geom_geo['plume_lats'].flatten()
    # flat_lons = plume_geom_geo['plume_lons'].flatten()
    # flat_aods = plume_geom_geo['plume_aod'].flatten()
    # flat_flags = plume_geom_geo['plume_flag'].flatten()
    # for i, (lon, lat) in enumerate(zip(flat_lons, flat_lats)):
    #     utm_point = ut.reproject_shapely(Point(lon, lat), plume_geom_utm['utm_resampler_plume'])
    #     dist_tail = utm_point.distance(tail_edge)
    #     if dist_tail < 325: # half a viirs pixel
    #         # get the aods and flags that intersect the line
    #         aods.append(flat_aods[i])
    #         flags.append(flat_flags[i])

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
    # if np.max(flags) == 3:
    #     # if the plume does intersect, the tail is point with the
    #     # least distance from the edge
    #     tail = tail_edge.interpolate(tail_edge.project(head))
    #     tail_lon, tail_lat = plume_geom_utm['utm_resampler_plume'].resample_point_to_geo(tail.y, tail.x)
    #     return {'tail_lon': tail_lon,
    #             'tail_lat': tail_lat,
    #             'tail': tail}
    # else:
    #     # in this instance the plume does not intersect with the end of
    #     # bounding box.  So we can assume that the frp that produced the
    #     # observed plume all occurred since the last minimum.
    #     return None

    tail = tail_edge.interpolate(tail_edge.project(head))
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

    return rad_bb


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


def extract_plume_flow(plume_geom_geo, f1_subset_reproj, flow, plume_vector, plume_logging_path, fname, plot=True):
    plume_mask = plume_geom_geo['plume_mask']
    # if plume mask not same shape as himawari subset them
    # adjust it to match.  Exact overlay doesn't matter as
    # we are looking for average plume motion
    if plume_mask.shape != f1_subset_reproj.shape:
        if plume_mask.size < f1_subset_reproj.size:
            ax0_diff = f1_subset_reproj.shape[0] - plume_mask.shape[0]
            ax0_pad = (ax0_diff, 0)  # padding n_before, n_after.  n_after always zero
            ax1_diff = f1_subset_reproj.shape[1] - plume_mask.shape[1]
            ax1_pad = (ax1_diff, 0)
            plume_mask = np.pad(plume_mask, (ax0_pad, ax1_pad), 'edge')
        else:
            plume_mask = plume_mask[:f1_subset_reproj.shape[0], :f1_subset_reproj.shape[1]]

    # mask flow to plume extent and invert
    # the y displacements.
    flow *= plume_mask[..., np.newaxis]
    flow[:, :, 0] *= -1

    # mask flow to only valid angles
    angles = angle_between(flow.copy(), plume_vector)
    angular_mask = (angles <= constants.angular_limit) | (angles >= (2 * np.pi) - constants.angular_limit)
    flow *= angular_mask[..., np.newaxis]

    if plot:
        vis.draw_flow(f1_subset_reproj, flow, plume_logging_path, fname)

    # mask flow to moving points
    x, y = flow.T
    mask = (x != 0) & (y != 0)
    y = y[mask]
    x = x[mask]

    # take the most most extreme quartile data
    if np.abs(y.min()) > y.max():
        y_pc = np.percentile(y, 25)
        y = y[y < y_pc]
    else:
        y_pc = np.percentile(y, 75)
        y = y[y > y_pc]
    if np.abs(x.min()) > x.max():
        x_pc = np.percentile(x, 25)
        x = x[x < x_pc]
    else:
        x_pc = np.percentile(x, 75)
        x = x[x > x_pc]

    # determine plume flow in metres
    y = np.mean(y) * constants.utm_grid_size
    x = np.mean(x) * constants.utm_grid_size
    plume_flow = (x,y)

    return plume_flow


def find_flow(plume_logging_path, plume_geom_utm, plume_geom_geo, pp, timestamp):

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
    plume_head, plume_tail, plume_vector = compute_plume_vector(plume_geom_geo, plume_geom_utm, pp, t0)

    # debugging
    # plume_head = {'head_lon': 104.1464,
    #               'head_lat': -1.79205,
    #               'head': Point(405058.3078391715, -198098.1352896034)}
    # plume_tail = {'tail_lon': 104.080129898,
    #               'tail_lat': -1.71434879993,
    #               'tail': Point(397682.580420428, -189512.1929388661)}
    # plume_vector = (np.array(plume_head['head'].coords[0]) - np.array(plume_tail['tail'].coords))[0]

    if plume_head is None:
        t1 = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[0]).group(),
                               '%Y%m%d_%H%M')
        t2 = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[-1]).group(),
                               '%Y%m%d_%H%M')
        return None, geostationary_fnames[:], t1, t2

    plume_length = np.linalg.norm(plume_vector)  # plume length in metres

    # begin iteration here
    current_tracked_plume_distance = 0
    for i in xrange(len(geostationary_fnames) - 1):
        if i == 0:
            f1_subset = extract_observation(geostationary_fnames[i], bbox, min_geo_segment)
            f1_subset_reproj = plume_geom_utm['utm_resampler_plume'].resample_image(f1_subset,
                                                                                    geostationary_lats_subset,
                                                                                    geostationary_lons_subset)
            f2_subset = extract_observation(geostationary_fnames[i + 1], bbox, min_geo_segment)
            f2_subset_reproj = plume_geom_utm['utm_resampler_plume'].resample_image(f2_subset,
                                                                                    geostationary_lats_subset,
                                                                                    geostationary_lons_subset)
        else:
            f1_subset_reproj = f2_subset_reproj
            f2_subset = extract_observation(geostationary_fnames[i + 1], bbox, min_geo_segment)
            f2_subset_reproj = plume_geom_utm['utm_resampler_plume'].resample_image(f2_subset,
                                                                                    geostationary_lats_subset,
                                                                                    geostationary_lons_subset)

        # prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
        scene_flow = cv2.calcOpticalFlowFarneback(f2_subset_reproj, f1_subset_reproj, None, 0.5, 3, 11, 5, 7, 1.5,
                                            cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        plume_flow = extract_plume_flow(plume_geom_geo, f2_subset_reproj, scene_flow, plume_vector,
                                        plume_logging_path, geostationary_fnames[i + 1], plot=pp['plot'])

        # project flow onto principle axis
        projected_flow = np.dot(plume_vector, plume_flow) / \
                         np.dot(plume_vector, plume_vector) * plume_vector

        current_tracked_plume_distance += np.linalg.norm(projected_flow)

        if pp['plot']:
            if i == 0:
                plot_images = [f1_subset_reproj, f2_subset_reproj]
                plot_names = [geostationary_fnames[i], geostationary_fnames[i + 1]]
                t = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[i]).group(),
                                      '%Y%m%d_%H%M')
                plot_fires = [ff.fire_locations_for_plume_roi(plume_geom_geo, pp['frp_df'], t)]
                t = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[i+1]).group(),
                                      '%Y%m%d_%H%M')
                plot_fires.append(ff.fire_locations_for_plume_roi(plume_geom_geo, pp['frp_df'], t))
                plume_flows = [plume_flow]
                projected_flows = [projected_flow]
            else:
                plot_images.append(f2_subset_reproj)
                plot_names.append(geostationary_fnames[i+1])
                t = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[i+1]).group(),
                                  '%Y%m%d_%H%M')
                plot_fires.append(ff.fire_locations_for_plume_roi(plume_geom_geo, pp['frp_df'], t))
                plume_flows.append(plume_flow)
                projected_flows.append(projected_flow)

        if (((plume_length - current_tracked_plume_distance) < constants.utm_grid_size) |
            (current_tracked_plume_distance > plume_length)):
            break

    if pp['plot']:
        vis.run_plot(plot_images, plot_fires, plume_flows, projected_flows,
                     plume_head, plume_tail, plume_geom_utm['utm_plume_points'], plume_geom_utm['utm_resampler_plume'],
                     plume_logging_path, plot_names, i + 1)

    # get the plume start and stop times
    t1 = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[0]).group(), '%Y%m%d_%H%M')
    t2 = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[i+1]).group(), '%Y%m%d_%H%M')

    # return the projected flow means in UTM coords, and the list of himawari filenames asspocated with the flows
    return projected_flows[:i + 1], geostationary_fnames[:i + 1], t1, t2