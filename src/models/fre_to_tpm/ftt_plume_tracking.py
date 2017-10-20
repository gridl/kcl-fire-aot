# load in required packages
import glob
import os
from datetime import datetime
import logging

import numpy as np
from scipy import ndimage
import cv2

import matplotlib.pyplot as plt

import src.data.readers.load_hrit as load_hrit
import src.config.filepaths as fp
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
            mid_point_a = (np.array(v1) + np.array(v2)) / 2
            # get the opposite side of the rectangle
            mid_point_b = (np.array(verts[(i + 2) % 4]) + np.array(verts[(i + 3) % 4])) / 2

    # we want the head of the vector at the fire, and the tail at the origin
    # as mid_point_a is closest to the fire then we need to subtract b from a (by vector subtraction)
    # head location, tail location, relative shift
    return mid_point_a, mid_point_b, mid_point_a - mid_point_b  # fires closest to a


def spatial_subset(lats_1, lons_1, lats_2, lons_2):
    """
    :param lats_1: target lat region
    :param lons_1: target lon region
    :param lats_2: extended lat region
    :param lons_2: extended lon region
    :return bounds: bounding box locating l1 in l2
    """

    padding = 100  # pixels

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
    sig = 1  # standard deviation of gaussian
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
    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(im2, im1,
                                              p1, None, **lk_params)  # matching in other direction
    good = find_good_tracks(p0, p1, p0r)
    tracks = update_tracks(tracks, p1, good)

    if tracks:
        # subtract the feature locations in the second image from the first image
        # in effect this is im2 - im1.  Could perhaps make the code clearer here
        flow = (p0 - p1).reshape(-1, 2)[good]
        flow = adjust_image_map_coordinates(flow)
        return tracks, flow
    else:
        return tracks, []


def assess_flow(flow, flow_means, flow_sds, flow_nobs, tracks, i, pix_size=1000):
    """

    :param flow: the current flow vectors from the feature tracking
    :param flow_means: the vector containing the flow means
    :param flow_sds: the vector containing the flow standard deviations
    :param tracks: the track points
    :param i: the current index
    :return: None
    """
    if len(tracks) <= 0:
        if i != 0:
            flow_means[i] = flow_means[i - 1]
            flow_sds[i] = flow_sds[i - 1]
            flow_nobs[i] = 0
    else:
        flow_means[i, :] = np.mean(flow, axis=0) * pix_size
        flow_sds[i, :] = np.std(flow, axis=0) * pix_size
        flow_nobs[i] = len(tracks)

    if (flow_means[:i] == 0).any():
        mask = [flow_means[:i] == 0]
        flow_means[:i][mask] = flow_means[i]
        flow_sds[:i][mask] = flow_sds[i]

    if flow_nobs[i] < 5:
        flow_means[i] = flow_means[i - 1]
        flow_sds[i] = flow_sds[i - 1]


def find_integration_start_stop_times(plume_fname,
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

    fast = cv2.FastFeatureDetector_create(threshold=15)  # feature detector

    # plot stuff
    if plot:
        utm_flow_vectors = []
        utm_plume_projected_flow_vectors = [plume_tail.copy()]

    # iterator stuff
    plume_length = np.linalg.norm(plume_vector)
    flow_means = np.zeros([72, 2])
    flow_sds = np.zeros([72, 2])
    flow_nobs = np.zeros([72])
    projected_flow_magnitude = np.zeros(72)
    tracks = []
    thresh = 1000  # stopping condition in metres (if distance between plumes is less than this)

    # iterate over geostationary files
    for i, (f1, f2) in enumerate(zip(geostationary_fnames[:-1], geostationary_fnames[1:])):

        # set up observations
        f1_subset, f2_subset, f1_display_subset, f2_display_subset = extract_observations(f1, f2, bb, min_image_segment)

        # reproject subsets to UTM grid
        f1_subset_reproj = utm_resampler.resample_image(f1_subset, subset_lats, subset_lons)
        f2_subset_reproj = utm_resampler.resample_image(f2_subset, subset_lats, subset_lons)
        f2_display_subset_reproj = utm_resampler.resample_image(f2_display_subset, subset_lats, subset_lons)

        # if plotting and on first iteration plot the first image
        if plot & (i == 0):
            f1_display_subset_reproj = utm_resampler.resample_image(f1_display_subset, subset_lats, subset_lons)
            vis.display_masked_map_first(f1_display_subset_reproj,
                                         plume_points,
                                         utm_resampler,
                                         plume_head,
                                         plume_tail,
                                         f1.split('/')[-1].split('.')[0] + '_subset.jpg')

        # FEATURE DETECTION - detect good points to track in the image using FAST
        feature_detector(fast, f2_subset_reproj, plume_mask, tracks)  # tracks updated inplace

        # FLOW COMPUTATION - compute the flow between the images, but only if we have features
        print 'n points:', len(tracks)
        if len(tracks) > 0:
            tracks, flow = compute_flow(tracks, f2_subset_reproj, f1_subset_reproj)

        # compute mean flow for plume
        assess_flow(flow, flow_means, flow_sds, flow_nobs, tracks, i)

        # now project flow vector onto plume vector
        projected_flow_vector = np.dot(plume_vector, flow_means[i]) / \
                                np.dot(plume_vector, plume_vector) * plume_vector
        projected_flow_magnitude[i] = np.linalg.norm(projected_flow_vector)

        # plot masked plume
        if plot:
            utm_flow_vectors += [utm_plume_projected_flow_vectors[-1] + flow_means[i]]
            utm_plume_projected_flow_vectors += [utm_plume_projected_flow_vectors[-1] + projected_flow_vector]
            vis.display_masked_map(f2_display_subset_reproj,
                                   plume_points,
                                   utm_resampler,
                                   plume_head,
                                   plume_tail,
                                   utm_flow_vectors,
                                   utm_plume_projected_flow_vectors,
                                   f2.split('/')[-1].split('.')[0] + '_subset.jpg')

        # sum current plume length and compare with total plume length
        summed_length = projected_flow_magnitude.sum()
        if ((summed_length - plume_length) < thresh) | (summed_length > plume_length):
            t1 = datetime.strptime(geostationary_fnames[0].split('/')[-1][7:20], '%Y%m%d_%H%M')
            t2 = datetime.strptime(f2.split('/')[-1][7:20], '%Y%m%d_%H%M')
            return t1, t2  # return time of the second file

    return None, None
