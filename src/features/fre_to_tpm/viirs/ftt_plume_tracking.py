# load in required packages
import glob
import os
from datetime import datetime, timedelta
import logging
import re

import numpy as np
from scipy import ndimage
import cv2
from shapely.geometry import Point, LineString

import src.data.readers.load_hrit as load_hrit
import src.config.filepaths as fp
import src.visualization.ftt_visualiser as vis
import src.features.fre_to_tpm.viirs.ftt_fre as ff
import src.features.fre_to_tpm.viirs.ftt_utils as ut
import src.config.constants as constants

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


def find_tail_edge(plume_geom_utm):

    # convex hull of plume
    x, y = plume_geom_utm['utm_plume_points'].minimum_rotated_rectangle.exterior.xy

    # get parallel edges of convex hull
    edge_a = LineString([(x[0], y[0]), (x[1], y[1])])
    edge_b = LineString([(x[2], y[2]), (x[3], y[3])])
    edge_c = LineString([(x[1], y[1]), (x[2], y[2])])
    edge_d = LineString([(x[3], y[3]), (x[4], y[4])])

    edges = [edge_a, edge_b, edge_c, edge_d]
    distances = [plume_geom_utm['utm_plume_tail'].distance(i) for i in edges]

    return edges[np.argmin(distances)]


def find_plume_tail(head, plume_geom_utm):

    # find tail edge
    tail_edge = find_tail_edge(plume_geom_utm)

    # using head, project it on to the tail edge to find plume tail for purposes of the code
    tail = tail_edge.interpolate(tail_edge.project(head))
    tail_lon, tail_lat = plume_geom_utm['utm_resampler_plume'].resample_point_to_geo(tail.y, tail.x)
    return {'tail_lon': tail_lon,
            'tail_lat': tail_lat,
            'tail': tail}


def compute_plume_vector(plume_geom_geo, plume_geom_utm, pp, t):
    # first set up the two alternative head and tail combintations
    # second cehck if one of the heads is outside of the bounding polygon
    # if both inside find the orientation of the rectangle

    head_dict = find_plume_head(plume_geom_geo, plume_geom_utm, pp, t)
    tail_dict = find_plume_tail(head_dict['head'], plume_geom_utm)
    vect = np.array(head_dict['head'].coords) - np.array(tail_dict['tail'].coords)
    return head_dict, tail_dict, vect[0]


def compute_flow_window_size(plume_geom_utm):
    # convex hull of plume
    x, y = plume_geom_utm['utm_plume_points'].minimum_rotated_rectangle.exterior.xy

    d1 = np.linalg.norm(np.array([x[1], y[1]]) - np.array([x[0], y[0]]))
    d2 = np.linalg.norm(np.array([x[2], y[2]]) - np.array([x[1], y[1]]))

    smallest_edge_len = np.min([d1, d2])
    return int((smallest_edge_len / constants.utm_grid_size) / 4.0)  # 4 determined from experimentation


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

    return {'geostationary_lats_subset':geostationary_lats_subset,
            'geostationary_lons_subset':geostationary_lons_subset}


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


def load_image(geostationary_fname, bbox, min_geo_segment):
    return extract_observation(geostationary_fname, bbox, min_geo_segment)


def reproject_image(im, geo_dict, plume_geom_utm):
        return plume_geom_utm['utm_resampler_plume'].resample_image(im, geo_dict['geostationary_lats_subset'],
                                                                        geo_dict['geostationary_lons_subset'])

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


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
    flow[:,:,1] *= -1
    return flow


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """

    v1 = adjust_image_map_coordinates(v1)

    # first norm the vectors
    eps = 0.0001
    v1 /= np.sqrt(((v1 + eps) ** 2).sum(-1))[..., np.newaxis]
    v2 /= np.sqrt(((v2 + eps) ** 2).sum(-1))[..., np.newaxis]

    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


def extract_plume_flow(plume_geom_geo, plume_geom_utm, f1_subset_reproj, flow,
                       plume_vector, plume_head, plume_tail,
                       plume_logging_path, fname, stage_name, plot=True):
    # plume_mask = plume_geom_geo['plume_mask']
    #
    # # if plume mask not same shape as himawari subset them
    # # adjust it to match.  Exact overlay doesn't matter as
    # # we are looking for average plume motion
    # if plume_mask.shape != f1_subset_reproj.shape:
    #     if plume_mask.size < f1_subset_reproj.size:
    #         ax0_diff = f1_subset_reproj.shape[0] - plume_mask.shape[0]
    #         ax0_pad = (ax0_diff, 0)  # padding n_before, n_after.  n_after always zero
    #         ax1_diff = f1_subset_reproj.shape[1] - plume_mask.shape[1]
    #         ax1_pad = (ax1_diff, 0)
    #         plume_mask = np.pad(plume_mask, (ax0_pad, ax1_pad), 'edge')
    #     else:
    #         plume_mask = plume_mask[:f1_subset_reproj.shape[0], :f1_subset_reproj.shape[1]]
    #
    # # mask flow to plume extent and invert
    # # the x displacements.
    # flow *= plume_mask[..., np.newaxis]


    angles = angle_between(flow.copy(), plume_vector)
    angular_mask = angles <= constants.angular_limit
    flow *= angular_mask[..., np.newaxis]

    if plot:
        vis.draw_flow(f1_subset_reproj, flow, plume_logging_path, fname, 'unmapped_flow_' + stage_name)
        vis.draw_flow_map(f1_subset_reproj, plume_geom_utm['utm_resampler_plume'], plume_geom_utm['utm_plume_points'],
                          plume_head, plume_tail, flow, plume_logging_path, fname, 'mapped_flow_' + stage_name, step=2)  # mask flow to moving points
    x, y = flow.T
    mask = (x != 0) & (y != 0)
    y = y[mask]
    x = x[mask]

    # take the most most extreme quartile data
    if np.abs(y.min()) > y.max():
        y_pc = np.percentile(y, 25)
        y_mask = y < y_pc

        # y_pc_upper = np.percentile(y, 30)
        # y_pc_lower = np.percentile(y, 5)
        # y = y[(y < y_pc_upper) & (y > y_pc_lower)]
    else:
        y_pc = np.percentile(y, 75)
        y_mask = y > y_pc

        # y_pc_upper = np.percentile(y, 95)
        # y_pc_lower = np.percentile(y, 70)
        # y = y[(y < y_pc_upper) & (y > y_pc_lower)]
    if np.abs(x.min()) > x.max():
        x_pc = np.percentile(x, 25)
        x_mask = x < x_pc

        # x_pc_upper = np.percentile(x, 30)
        # x_pc_lower = np.percentile(x, 5)
        # x = x[(x < x_pc_upper) & (x > x_pc_lower)]
    else:
        x_pc = np.percentile(x, 75)
        x_mask = x > x_pc

    # TODO check this masking
    y = y[y_mask | x_mask]
    x = x[y_mask | x_mask]

    # determine plume flow in metres
    y = np.mean(y) * constants.utm_grid_size
    x = np.mean(x) * constants.utm_grid_size
    plume_flow = (x,y)

    return plume_flow


def tracker(plume_logging_path, plume_geom_utm, plume_geom_geo, pp, timestamp):

    # get bounding box around smoke plume in geostationary imager coordinates
    # and extract the geographic coordinates for the roi, also set up plot stuff
    bbox = spatial_subset(plume_geom_geo['plume_lats'], plume_geom_geo['plume_lons'],
                          pp['geostationary_lats'], pp['geostationary_lons'])

    him_geo_dict = subset_geograpic_data(pp['geostationary_lats'], pp['geostationary_lons'], bbox)

    him_segment = find_min_himawari_image_segment(bbox)
    adjust_bb_for_segment(bbox, him_segment - 1)

    plume_time = get_plume_time(timestamp)
    geostationary_fnames = setup_geostationary_files(plume_time, him_segment)

    # establish plume vector, and the total plume length.  If the plume does not
    # intersect with the end of the polygon then we do not need to worry about
    # limiting the FRP integration times.  So just return the min and max geo times
    t0 = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[0]).group(),
                          '%Y%m%d_%H%M')
    plume_head, plume_tail, plume_vector = compute_plume_vector(plume_geom_geo, plume_geom_utm, pp, t0)

    # determine window size as a function of the smallest axis of the plume polygon exterior
    #flow_win_size = compute_flow_window_size(plume_geom_utm)

    # plume length in metres
    plume_length = np.linalg.norm(plume_vector)

    # a priori flow determination
    flow_images = []
    flows = []
    current_tracked_plume_distance = 0
    for i in xrange(6):

        im_subset = load_image(geostationary_fnames[i], bbox, him_segment)
        im_subset_reproj = reproject_image(im_subset, him_geo_dict, plume_geom_utm)
        flow_images.append(im_subset_reproj)

        # if on the first image, continue to load the second
        if i == 0:
            continue

        # As the tracking is from t0 back to source (i.e. bak through time to t-n), we want
        # to calulate the flow in reverse, with the previous image being the most recent
        # and the next image being the observation prior to the most recent.
        flow_win_size = 5
        scene_flow = cv2.calcOpticalFlowFarneback(flow_images[i-1], flow_images[i],
                                                  flow=None,
                                                  pyr_scale=0.5, levels=1,
                                                  winsize=flow_win_size, iterations=7,
                                                  poly_n=7, poly_sigma=1.5,
                                                  flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)


        # lets do an additional median smoothing here and store flows
        #scene_flow = ndimage.filters.median_filter(scene_flow, 2)
        flows.append(scene_flow)

        plume_flow_x, plume_flow_y = extract_plume_flow(plume_geom_geo, plume_geom_utm, flow_images[i-1], scene_flow,
                                                        plume_vector, plume_head, plume_tail,
                                                        plume_logging_path, geostationary_fnames[i-1],
                                                        'prior_flow_', plot=pp['plot'])


        # adust flow for utm
        plume_flow_y *= -1

        # projected_flow = np.dot(plume_vector, (plume_flow_x, plume_flow_y)) / \
        #                  np.dot(plume_vector, plume_vector) * plume_vector

        current_tracked_plume_distance += np.linalg.norm((plume_flow_x, plume_flow_y))
        if (((plume_length - current_tracked_plume_distance) < constants.utm_grid_size) |
                (current_tracked_plume_distance > plume_length)):
            break

    # repeat first flow as best estimate
    flows.insert(0, flows[0])

    # a posteriori flow determination
    if pp['plot']:
        plot_fires = []
        plume_flows = []
        projected_flows = []

    current_tracked_plume_distance = 0
    velocity = []
    for i in xrange(6):  # look at the last hour of data

        # again skip first image
        if i == 0:
            continue

        # prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
        scene_flow = cv2.calcOpticalFlowFarneback(flow_images[i-1], flow_images[i],
                                                  flow=flows[i-1],
                                                  pyr_scale=0.5, levels=1,
                                                  winsize=flow_win_size, iterations=3,
                                                  poly_n=7, poly_sigma=1.4,
                                                  flags=cv2.OPTFLOW_USE_INITIAL_FLOW + cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        # we should not do this for the second round, as have already applied it in the first.  Instead
        # just mask to plume and take the mean and sd as the values
        plume_flow_x, plume_flow_y = extract_plume_flow(plume_geom_geo, plume_geom_utm, flow_images[i-1], scene_flow,
                                                        plume_vector, plume_head, plume_tail,
                                                        plume_logging_path, geostationary_fnames[i-1],
                                                        'post_flow_', plot=pp['plot'])
        # adust flow for utm
        plume_flow_y *= -1

        # project flow onto principle axis
        # projected_flow = np.dot(plume_vector, (plume_flow_x, plume_flow_y)) / \
        #                  np.dot(plume_vector, plume_vector) * plume_vector
        # distance_travelled = np.linalg.norm(projected_flow)
        distance_travelled = np.linalg.norm((plume_flow_x, plume_flow_y))

        current_tracked_plume_distance += distance_travelled

        # record the the velocity in the plume direction
        velocity.append(distance_travelled / 600)  # gives velocity in m/s (600 seconds between images)


        if pp['plot']:
            t = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[i-1]).group(), '%Y%m%d_%H%M')
            plot_fires.append(ff.fire_locations_for_plume_roi(plume_geom_geo, pp['frp_df'], t))
            plume_flows.append((plume_flow_x, plume_flow_y))
            projected_flows.append((plume_flow_x, plume_flow_y))

        print current_tracked_plume_distance
        print plume_length
        print

        if (((plume_length - current_tracked_plume_distance) < constants.utm_grid_size) |
                (current_tracked_plume_distance > plume_length)):
            break

    # get the plume start and stop times
    t_start = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[0]).group(), '%Y%m%d_%H%M')

    #mean_velocity = np.mean(velocity)
    #time_for_plume = plume_length / mean_velocity
    max_velocity = np.max(velocity)
    time_for_plume = plume_length / max_velocity  # in seconds

    t_stop = t_start - datetime.timedelta(seconds=time_for_plume)

    # round to nearest 10 minutes
    t_stop += timedelta(minutes=5)
    t_stop -= timedelta(minutes=t_stop.minute % 10,
                                 seconds=t_stop.second,
                                 microseconds=t_stop.microsecond)

    #print 'plume velocity m/s', mean_velocity
    print 'plume velocity m/s', max_velocity
    print 'time for plume s', time_for_plume
    print t_start
    print t_stop
    print

    if pp['plot']:
        n = np.round(time_for_plume / 600)
        plot_images = flow_images[:n+1]
        plot_names = geostationary_fnames[:n+1]
        t = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[n]).group(), '%Y%m%d_%H%M')
        plot_fires.append(ff.fire_locations_for_plume_roi(plume_geom_geo, pp['frp_df'], t))

        vis.run_plot(plot_images, plot_fires, plume_flows, projected_flows,
                     plume_head, plume_tail, plume_geom_utm['utm_plume_points'], plume_geom_utm['utm_resampler_plume'],
                     plume_logging_path, plot_names, n+1)


    # return the projected flow means in UTM coords, and the list of himawari filenames asspocated with the flows
    return projected_flows[:i], geostationary_fnames[:i], t_start, t_stop, time_for_plume