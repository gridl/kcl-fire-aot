import os
import glob
import re
from datetime import datetime

import pandas as pd
import scipy.ndimage as ndimage
import numpy as np
import matplotlib.pyplot as plt
import cv2

import src.data.readers.load_hrit as load_hrit
import src.config.filepaths as fp

# CONSTANTS
MAX_REFLEC = 0.05


def get_geostationary_fnames(ym, day, image_segment):
    """

    :param plume_time: the time of the MYD observation of the plume
    :param image_segment: the Himawari image segment
    :return: the geostationary files for the day of and the day before the fire
    """

    # get all files in the directory using glob with band 3 for main segment
    p = os.path.join(fp.path_to_himawari_imagery, ym, day)
    return glob.glob(p + '/*/*/B03/*S' + str(image_segment).zfill(2) + '*')


def sort_geostationary_by_time(geostationary_fnames):
    """

    :param geostationary_fnames goestationary filenames
    :return: the geostationary filenames in time order
    """
    times = [datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", f).group()
                               , '%Y%m%d_%H%M') for f in geostationary_fnames]
    return [f for _, f in sorted(zip(times, geostationary_fnames))]


def setup_geostationary_files(ym, day, image_segment):
    geostationary_fnames = get_geostationary_fnames(ym, day, image_segment)
    geostationary_fnames = sort_geostationary_by_time(geostationary_fnames)
    geostationary_fnames.reverse()
    return geostationary_fnames


def extract_observation(f, bb):
    # load geostationary files for the segment
    # rad_segment, _ = load_hrit.H8_file_read(os.path.join(fp.path_to_himawari_imagery, f))
    rad_segment, ref_segment = load_hrit.H8_file_read(os.path.join(fp.path_to_himawari_imagery, f))

    # extract geostationary image subset using adjusted bb
    rad_subset = rad_segment[bb['min_y']:bb['max_y'], bb['min_x']:bb['max_x']]
    ref_subset = ref_segment[bb['min_y']:bb['max_y'], bb['min_x']:bb['max_x']]
    return rad_subset, ref_subset


def assess_coregistration(im_1, mask_1, im_2, mask_2, sift, flann, f, plot=True):
    # convert datatypes to the used in opencv
    mask_1 = mask_1.astype('uint8')
    im_1 = (im_1 * 255).astype('uint8')
    mask_2 = mask_2.astype('uint8')
    im_2 = (im_2 * 255).astype('uint8')

    # # find the keypoints and descriptors with SIFT
    kp_1, des_1 = sift.detectAndCompute(im_1, mask_1)
    kp_2, des_2 = sift.detectAndCompute(im_2, mask_2)

    # BFMatcher with default params
    matches = flann.knnMatch(des_1, des_2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    src_pts = np.float32([kp_1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    if plot:
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matches_mask,  # draw only inliers
                           flags=2)
        img = cv2.drawMatches(im_1, kp_1, im_2, kp_2, good, None, **draw_params)
        plt.imshow(img, 'gray')
        plt.savefig(os.path.join(fp.path_to_opt_flow_visuals, f.replace('.DAT.bz2', '_surf.png')), bbox_inches='tight')

    src_pts = src_pts[np.where(matches_mask), :].squeeze()
    dst_pts = dst_pts[np.where(matches_mask), :].squeeze()

    #
    d = (src_pts - dst_pts).squeeze()
    x, y = d[:, 0], d[:, 1]

    # remove outliers
    mean_x = np.mean(x)
    std_x = np.std(x)
    mean_y = np.mean(y)
    std_y = np.std(y)
    mask = (np.abs(x) < mean_x + 3 * std_x) & (np.abs(y) < mean_y + 3 * std_y)

    x = x[mask]
    y = y[mask]

    out_dict = {'fname': np.repeat(f, y.size),
                'x_shift': x,
                'y_shift': y}

    return pd.DataFrame.from_dict(out_dict)


def assess_dense_flow(im_1, mask_1, im_2, mask_2, f, plot=True):
    flow_win_size = 5
    flow = cv2.calcOpticalFlowFarneback(im_1, im_2,
                                        flow=None,
                                        pyr_scale=0.5, levels=1,
                                        winsize=flow_win_size, iterations=7,
                                        poly_n=7, poly_sigma=1.5,
                                        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    if plot:
        step = 30

        plt.close('all')
        h, w = im_1.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T

        x_shift = x + fx
        y_shift = y + fy

        mask = (x_shift > 0) & (x_shift < w) & (y_shift > 0) & (y_shift < h)
        x = x[mask]
        y = y[mask]
        x_shift = x_shift[mask]
        y_shift = y_shift[mask]

        ax = plt.axes()
        ax.imshow(im_1, cmap='gray')
        ax.plot((x, x_shift), (y, y_shift), 'r-', linewidth=0.25)
        ax.plot(x, y, 'r.', markersize=0.25)

        plt.savefig(os.path.join(fp.path_to_opt_flow_visuals, f.replace('.DAT.bz2', '_flow.png')), bbox_inches='tight')
        plt.close()

    # get masked flows
    y = flow[:,:,1][mask_1 * mask_2]
    x = flow[:,:,0][mask_1 * mask_2]

    mean_x = np.mean(x)
    std_x = np.std(x)
    mean_y = np.mean(y)
    std_y = np.std(y)
    mask = (np.abs(x) < mean_x + 3 * std_x) & (np.abs(y) < mean_y + 3 * std_y)

    x = x[mask]
    y = y[mask]

    # return outputs
    out_dict = {'fname': np.repeat(f, y.size),
                'x_shift': x,
                'y_shift': y}

    return pd.DataFrame.from_dict(out_dict)


def main():
    bb = {'min_x': 4500,
          'min_y': 750,
          'max_x': 6500,
          'max_y': -1}

    # get vis filenames
    geostationary_file_paths = setup_geostationary_files('201507', '06', 5)

    # init sift feature detector and brute force matcher
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.001, edgeThreshold=30)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # set up some lists to store the outputs
    sparse_flow_list = []
    dense_flow_list = []

    # iterate over vis files
    for f1, f2 in zip(geostationary_file_paths[0:-1], geostationary_file_paths[1:]):

        fname = f1.split('/')[-1]
        print fname

        # read in the data for roi
        rad_1, ref_1 = extract_observation(f1, bb)
        rad_2, ref_2 = extract_observation(f2, bb)

        # generate cloud mask and erode
        cloudfree_1 = ref_1 < MAX_REFLEC
        cloudfree_2 = ref_2 < MAX_REFLEC
        cloudfree_1 = ndimage.binary_erosion(cloudfree_1)
        cloudfree_2 = ndimage.binary_erosion(cloudfree_2)

        # check image to image coregistation
        sparse_df = assess_coregistration(ref_1, cloudfree_1, ref_2, cloudfree_2, sift, flann, fname)

        # do dense tracking (looking for plume motion)
        dense_df = assess_dense_flow(rad_1, cloudfree_1, rad_2, cloudfree_2, fname)

        # record
        sparse_flow_list.append(sparse_df)
        dense_flow_list.append(dense_df)

    # create and save the dataframes
    sparse_flow_df = pd.concat(sparse_flow_list)
    dense_flow_df = pd.concat(dense_flow_list)

    sparse_flow_df.to_csv(os.path.join(fp.path_to_dataframes, 'sparse_flow_assessment_data.csv'))
    dense_flow_df.to_csv(os.path.join(fp.path_to_dataframes, 'dense_flow_assessment_data.csv'))

if __name__ == "__main__":
    main()
