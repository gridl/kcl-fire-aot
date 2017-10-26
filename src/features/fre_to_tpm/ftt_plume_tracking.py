# load in required packages
import glob
import os
from datetime import datetime
import logging

import numpy as np
from scipy import ndimage
import pandas as pd
import cv2

import matplotlib.pyplot as plt

import src.data.readers.load_hrit as load_hrit
import src.config.filepaths as fp
import src.config.features as fc
import src.visualization.ftt_visualiser as vis

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def get_plume_time(plume_fname):
    """
    :param plume_fname: the MYD filename for the plume
    :return: the plume time stamp
    """
    return datetime.strptime(plume_fname[10:22], '%Y%j.%H%M')


def compute_plume_vector(plume_points, fire_positions):
    """

    :param plume_points: The verticies of the plume in the form of Shapely points
    :param fire_positions: The location of the fire contained within the polygon
    :return: The loingest vector that points from the distal end of the plume to the fire end
    """

    # ratio to determine if the head of the vector is too far from the fires
    ratio = 0.25  # TODO add this to config file

    # compute median fire position
    fire_positions = np.array(fire_positions)
    # x, y
    fire_pos = np.array([np.median(fire_positions[:, 0]), np.median(fire_positions[:, 1])])

    # first find the vertices of the plume polygon
    x, y = plume_points.minimum_rotated_rectangle.exterior.xy
    verts = zip(x, y)

    smallest_dist_from_side = 999999
    for i, (v1, v2) in enumerate(zip(verts[:-1], verts[1:])):
        side = np.array(v2) - np.array(v1)  # get the vector of the side of the rectangle
        hyp = fire_pos - np.array(v1)  # get the vector between plume and vertex
        proj_hyp = np.dot(side, hyp) / np.dot(side, side) * side  # project that vector onto side
        dist_from_side = np.linalg.norm(np.array(hyp - proj_hyp))  # compute distance between them
        if dist_from_side < smallest_dist_from_side:
            smallest_dist_from_side = dist_from_side

            # head and tail of the vector are the side midpoint near the fires (i.e. the current side)
            # and the opposite side to the side near the fires
            head = (np.array(v1) + np.array(v2)) / 2
            tail = (np.array(verts[(i + 2) % 4]) + np.array(verts[(i + 3) % 4])) / 2

    # compute distance between fires and head of the midpoint vector, and head and tail of the midpoint vector
    fire_to_head_distance = np.linalg.norm(np.array(fire_pos - head))
    head_to_tail_distance = np.linalg.norm(np.array(tail - head))

    # if the ratio of the above less than the set threshold, then we have an appropriate plume vector
    # that runs from the end of the plume, to the fires.
    if fire_to_head_distance / head_to_tail_distance < ratio:
        return head, tail, head - tail

    # if not then we assume that the plume is much longer than wide and use the following approach
    else:

        # next find the midpoints of the shortest sides
        side_a = np.linalg.norm(np.array(verts[0]) - np.array(verts[1]))
        side_b = np.linalg.norm(np.array(verts[1]) - np.array(verts[2]))
        if side_a > side_b:
            mid_point_a = (np.array(verts[1]) + np.array(verts[2])) / 2.
            mid_point_b = (np.array(verts[3]) + np.array(verts[4])) / 2.
        else:
            mid_point_a = (np.array(verts[0]) + np.array(verts[1])) / 2.
            mid_point_b = (np.array(verts[2]) + np.array(verts[3])) / 2.

        # determine which mid point is closest to the fire and create vector
        dist_a = np.linalg.norm(fire_pos - mid_point_a)
        dist_b = np.linalg.norm(fire_pos - mid_point_b)

        # we want the head of the vector at the fire, and the tail at the origin
        # if mid_point_a is closest to the fire then we need to subtract b from a (by vector subtraction)
        # and vice versa if the fire is closest to midpoint b.
        if dist_a < dist_b:
            # head location, tail location, relative shift
            return mid_point_a, mid_point_b, mid_point_a - mid_point_b  # fires closest to a
        else:
            # head location, tail location, relative shift
            return mid_point_b, mid_point_a, mid_point_b - mid_point_a  # fire closest to b


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

    zoom = 2  # zoom is fixed and function of imagery we are using
    geostationary_lats_subset = ndimage.zoom(geostationary_lats_subset, zoom)
    geostationary_lons_subset = ndimage.zoom(geostationary_lons_subset, zoom)

    bb.update((x, y * zoom) for x, y in bb.items())  # enlarge bounding box by factor of zoom also
    return geostationary_lats_subset, geostationary_lons_subset


def find_min_himawari_image_segment(bb):
    """
    :param bb: bounding box
    :return: the himawari image segment for the given bounding box
    """
    # there are ten 1100 pixel segments in himawari 1 km data
    seg_size = 1100
    min_segment = bb['min_y'] / seg_size + 1
    return min_segment


def adjust_bb_for_segment(bb, segment):
    """
    :param bb: plume bounding box
    :param segment: the image segment that contains the bounding box
    :return: Nothing, the boudning box coordinates are adjusted inplace
    """
    seg_size = 1100
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
    p = os.path.join(fp.path_to_himawari_l1b, ym, day)
    fp_1 = glob.glob(p + '/*/*/B01/*S' + str(image_segment).zfill(2) + '*')

    # get the day before also
    day = str(plume_time.day - 1).zfill(2)
    p = os.path.join(fp.path_to_himawari_l1b, ym, day)
    fp_2 = glob.glob(p + '/*/*/B01/*S' + str(image_segment).zfill(2) + '*')

    files = fp_1 + fp_2

    return files


def restrict_geostationary_times(plume_time, geostationary_fnames):
    """

    :param plume_time: the plume time
    :param geostationary_fnames: the list of geostationary file names
    :return: only those goestationary files that were obtained prior to the myd overpass
    """
    return [f for f in geostationary_fnames if datetime.strptime(f.split('/')[-1][7:20], '%Y%m%d_%H%M') <= plume_time]


def sort_geostationary_by_time(geostationary_fnames):
    """

    :param geostationary_fnames goestationary filenames
    :return: the geostationary filenames in time order
    """
    times = [datetime.strptime(f.split('/')[-1][7:20], '%Y%m%d_%H%M') for f in geostationary_fnames]
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


def extract_observations(f1, f2, bb, segment):
    # load geostationary files for the segment
    f1_rad_s1, _ = load_hrit.H8_file_read(os.path.join(fp.path_to_himawari_l1b, f1))
    f2_rad_s1, _ = load_hrit.H8_file_read(os.path.join(fp.path_to_himawari_l1b, f2))

    # load for the next segment
    f1_rad_s2, _ = load_hrit.H8_file_read(os.path.join(fp.path_to_himawari_l1b, f1.replace('S' + str(segment).zfill(2),
                                                                                           'S' + str(segment + 1).zfill(
                                                                                               2))))
    f2_rad_s2, _ = load_hrit.H8_file_read(os.path.join(fp.path_to_himawari_l1b, f2.replace('S' + str(segment).zfill(2),
                                                                                           'S' + str(segment + 1).zfill(
                                                                                               2))))

    # concat the himawari files
    f1_rad = np.vstack((f1_rad_s1, f1_rad_s2))
    f2_rad = np.vstack((f2_rad_s1, f2_rad_s2))

    # extract geostationary image subset using adjusted bb and rescale to 8bit
    f1_rad_bb = f1_rad[bb['min_y']:bb['max_y'], bb['min_x']:bb['max_x']]
    f2_rad_bb = f2_rad[bb['min_y']:bb['max_y'], bb['min_x']:bb['max_x']]

    # normalise
    f1_rad_bb_norm = normalise_image(f1_rad_bb)
    f2_rad_bb_norm = normalise_image(f2_rad_bb)

    f1_rad_bb_norm = rescale_image(f1_rad_bb_norm, f1_rad_bb_norm.min(), f1_rad_bb_norm.max())
    f2_rad_bb_norm = rescale_image(f2_rad_bb_norm, f2_rad_bb_norm.min(), f2_rad_bb_norm.max())
    return f1_rad_bb_norm, f2_rad_bb_norm, f1_rad_bb, f2_rad_bb


def feature_detector(fast, im, mask, tracks):
    points = fast.detect(im, None)
    if points is not None:
        for pt in points:
            # check if points is in plume
            if mask[int(pt.pt[1]), int(pt.pt[0])]:
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


def assess_flow(flow, flow_means, flow_sds, flow_nobs, flow_update, tracks, i, pix_size=1000):
    """

    :param flow: the current flow vectors from the feature tracking
    :param flow_means: the vector containing the flow means
    :param flow_sds: the vector containing the flow standard deviations
    :param flow_nobs: the vector containing the number of flow observations
    :param flow_update: vector containing mean used to to update in case of tracks <= min tracks
    :param tracks: the track points
    :param i: the current index
    :return: None
    """
    if len(tracks) <= fc.min_number_tracks:
        if i != 0:
            flow_means[i] = flow_means[i - 1]
            flow_sds[i] = flow_sds[i - 1]
            flow_update[i] = i - 1
    else:
        flow_means[i, :] = np.mean(flow, axis=0) * pix_size
        flow_sds[i, :] = np.std(flow, axis=0) * pix_size
        flow_nobs[i] = len(tracks)

    # check if any points zero and update with most recent estimate
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


def projected_flow(plume_vector, flow_means, projected_flow_means, projected_flow_magnitudes, i):
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


def find_integration_start_stop_times(p_number,
                                      plume_logging_path,
                                      plume_fname,
                                      plume_points, plume_mask,
                                      plume_lats, plume_lons,
                                      geostationary_lats, geostationary_lons,
                                      fires,
                                      utm_resampler,
                                      plot=True):
    """
    Main function to compute the time stamps over which we need to integrate the
    FRP observations.  This is done by tracking the motion of the plume, and determining
    how long it takes for the plume to travel from its distal end to its source
    :param plume_fname:  The filename of the plume
    :param plume_points: The points that make up the plume polygon
    :param plume_mask: The plume mask
    :param plume_lats: The latitudes associated with the plume
    :param plume_lons: The longitudes associated with the plume
    :param geostationary_lats: The geostationary lat grid
    :param geostationary_lons: The geostationary lon grid
    :param fires:  The fires for the plume
    :param utm_resampler:  The resampler used to reproject everything into a common UTM grid
    :param plot:  Plotting flag
    :return:  TBD, but likely the himawari observation time associate with the plume crossover
    """
    plume_time = get_plume_time(plume_fname)

    plume_head, plume_tail, plume_vector = compute_plume_vector(plume_points, fires)

    bb = spatial_subset(plume_lats, plume_lons, geostationary_lats, geostationary_lons)

    subset_lats, subset_lons = subset_geograpic_data(geostationary_lats, geostationary_lons, bb)

    min_image_segment = find_min_himawari_image_segment(bb)

    adjust_bb_for_segment(bb, min_image_segment - 1)

    geostationary_fnames = setup_geostationary_files(plume_time, min_image_segment)

    fast = cv2.FastFeatureDetector_create(threshold=25)  # feature detector TODO move thresh to config file

    # iterator stuff
    plume_length = np.linalg.norm(plume_vector)
    flow_means = np.zeros([72, 2])
    flow_sds = np.zeros([72, 2])
    flow_nobs = np.zeros([72])
    flow_update = np.zeros([72])
    projected_flow_means = np.zeros([72, 2])
    projected_flow_magnitude = np.zeros(72)
    tracks = []

    stopping_thresh = 1000  # stopping condition in metres TODO move to config

    # set time variables that will be returned
    t1 = None
    t2 = None

    # iterate over geostationary files
    for i, (f1, f2) in enumerate(zip(geostationary_fnames[:-1], geostationary_fnames[1:])):

        # set up observations
        f1_subset, f2_subset, f1_display_subset, f2_display_subset = extract_observations(f1, f2, bb, min_image_segment)

        # reproject subsets to UTM grid
        f1_subset_reproj = utm_resampler.resample_image(f1_subset, subset_lats, subset_lons)
        f2_subset_reproj = utm_resampler.resample_image(f2_subset, subset_lats, subset_lons)

        if plot & (i == 0):
            plot_images = [utm_resampler.resample_image(f1_display_subset, subset_lats, subset_lons)]
            fnames = [f1]
        if plot:
            plot_images.append(utm_resampler.resample_image(f2_display_subset, subset_lats, subset_lons))
            fnames.append(f2)

        # FEATURE DETECTION - detect good points to track in the image using FAST
        feature_detector(fast, f2_subset_reproj, plume_mask, tracks)  # tracks updated inplace

        # FLOW COMPUTATION - compute the flow between the images
        tracks, flow = compute_flow(tracks, f2_subset_reproj, f1_subset_reproj)

        # compute mean flow for plume
        assess_flow(flow, flow_means, flow_sds, flow_nobs, flow_update, tracks, i)

        # now project flow vector onto plume vector
        projected_flow(plume_vector, flow_means, projected_flow_means, projected_flow_magnitude, i)

        # sum current plume length and compare with total plume length
        summed_length = projected_flow_magnitude.sum()
        if ((plume_length - summed_length) < stopping_thresh) | (summed_length > plume_length):
            t1 = datetime.strptime(geostationary_fnames[0].split('/')[-1][7:20], '%Y%m%d_%H%M')
            t2 = datetime.strptime(f2.split('/')[-1][7:20], '%Y%m%d_%H%M')
            break

    # save tracking information
    data = np.array([flow_means[:i+1, 0], flow_means[:i+1, 1],
                     projected_flow_means[:i+1, 0], projected_flow_means[:i+1, 1],
                     flow_sds[:i+1, 0], flow_sds[:i+1, 1],
                     projected_flow_magnitude[:i+1],
                     flow_nobs[:i+1], flow_update[:i+1]]).T
    columns = ['flow_means_x', 'flow_means_y', 'proj_flow_means_x', 'proj_flow_means_y',
               'flow_sds_x', 'flow_sds_y', 'proj_flow_mag', 'flow_nobs', 'flow_update']
    df = pd.DataFrame(data, index=[f.split('/')[-1] for f in geostationary_fnames[:i+1]], columns=columns)
    df.to_csv(os.path.join(plume_logging_path, str(p_number) + '_tracks.csv'))

    # plot plume
    if plot:
        vis.run_plot(plot_images, flow_means, projected_flow_means,
                     plume_head, plume_tail, plume_points, fires, utm_resampler,
                     plume_logging_path, fnames, i)

    return t1, t2
