# -*- coding: utf-8 -*-
import os
import logging
from dotenv import find_dotenv, load_dotenv

import re
import uuid

import numpy as np
import pandas as pd

import scipy.ndimage as ndimage
from datetime import datetime
from pyhdf.SD import SD, SDC
from skimage import exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import cv2

import config


def read_myd14(myd14_file):
    return SD(myd14_file, SDC.READ)


def firemask_myd14(myd14_data):
    return myd14_data.select('fire mask').get() >= 7


def read_myd021km(local_filename):
    return SD(local_filename, SDC.READ)


def fcc_myd021km(mod_data, fire_mask):
    mod_params_ref = mod_data.select("EV_1KM_RefSB").attributes()
    mod_params_emm = mod_data.select("EV_1KM_Emissive").attributes()
    ref = mod_data.select("EV_1KM_RefSB").get()
    emm = mod_data.select("EV_1KM_Emissive").get()

    # switch the red and bluse channels, so the we get nice bright red plumes
    ref_chan = 0
    emm_chan = 10
    r = (ref[ref_chan, :, :] - mod_params_ref['radiance_offsets'][ref_chan]) * mod_params_ref['radiance_scales'][
        ref_chan]
    b = (emm[emm_chan, :, :] - mod_params_emm['radiance_offsets'][emm_chan]) * mod_params_emm['radiance_scales'][
        emm_chan]
    g = (r - b) / (r + b)

    r = np.round((r * (255 / np.max(r))) * 1).astype('uint8')
    g = np.round((g * (255 / np.max(g))) * 1).astype('uint8')
    b = np.round((b * (255 / np.max(b))) * 1).astype('uint8')

    mini = 5
    maxi = 95

    r_min, r_max = np.percentile(r, (mini, maxi))
    r = exposure.rescale_intensity(r, in_range=(r_min, r_max))
    r[fire_mask] = 255

    g_min, g_max = np.percentile(g, (mini, maxi))
    g = exposure.rescale_intensity(g, in_range=(g_min, g_max))
    g[fire_mask] = 0

    b_min, b_max = np.percentile(b, (mini, maxi))
    b = exposure.rescale_intensity(b, in_range=(b_min, b_max))
    b[fire_mask] = 0

    rgb = np.dstack((r, g, b))

    return rgb


class Annotate(object):
    def __init__(self, im):
        self.f = plt.figure(figsize=(30, 15))
        self.ax = plt.gca()
        self.im = self.ax.imshow(im, interpolation='nearest')

        # set up the point holders for the polygon
        self.x = []
        self.y = []

        # set up the rectangle to hold the background
        self.rect = Rectangle((0, 0), 1, 1, edgecolor='black', facecolor='none')
        self.ax.add_patch(self.rect)
        self.x0_rect = None
        self.y0_rect = None
        self.x1_rect = None
        self.y1_rect = None

        # set up the events
        self.ax.figure.canvas.mpl_connect('button_press_event', self.click)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.release)

    def click(self, event):
        if event.button == 3:
            self.x.append(int(event.xdata))
            self.y.append(int(event.ydata))
            self.ax.add_patch(Circle((event.xdata, event.ydata), radius=0.25, facecolor='red', edgecolor='black'))
            self.ax.figure.canvas.draw()
        elif event.button == 2:
            self.x0_rect = int(event.xdata)
            self.y0_rect = int(event.ydata)

    def release(self, event):
        if event.button == 2:
            self.x1_rect = int(event.xdata)
            self.y1_rect = int(event.ydata)
            self.rect.set_width(self.x1_rect - self.x0_rect)
            self.rect.set_height(self.y1_rect - self.y0_rect)
            self.rect.set_xy((self.x0_rect, self.y0_rect))
            self.ax.figure.canvas.draw()


def digitise(img):
    img_copy = img.copy()
    plume_img = img.copy()
    smoke_polygons = []
    background_rectangles = []
    plume_ids = []

    plt.figure(figsize=(30, 15))
    plt.imshow(img, interpolation="nearest")
    plt.show()
    arg = raw_input("Do you want to digitise this plume?: [y, N]")
    if arg.lower() in ['', 'no', 'n']:
        return None, None, None

    while True:

        # first set up the annotator and show the image
        annotator = Annotate(img_copy)
        plt.show()

        # show them what they have digitised, and check if they are OK with that
        # if they are append the polygon, and modify the RGB to reflect the digitised region
        pts = zip(annotator.x, annotator.y)

        if not pts:
            print "you must select define a polygon containing smoke pixels"
            continue
        if annotator.x0_rect is None:
            print "you must define a background rectangle"
            continue
        pts = np.array(pts).reshape(-1, 1, 2)

        digitised_copy = img_copy.copy()

        cv2.polylines(digitised_copy, [pts], True, (255, 0, 0, 255), thickness=2)
        cv2.rectangle(digitised_copy,
                      (annotator.x0_rect,
                       annotator.y0_rect),
                      (annotator.x1_rect,
                       annotator.y1_rect),
                      (0, 0, 0, 255))
        plt.figure(figsize=(30, 15))
        plt.imshow(digitised_copy)
        plt.show()

        arg = raw_input("Are you happy with this plume digitisation? [Y,n]")
        if arg.lower() in ["", "y", "yes", 'ye']:
            smoke_polygons.append(zip(annotator.x, annotator.y))
            background_rectangles.append((annotator.x0_rect, annotator.x1_rect, annotator.y0_rect, annotator.y1_rect))
            img_copy = digitised_copy

            # store the plume id
            plume_id = uuid.uuid4()
            plume_ids.append(plume_id)

            # plot the plume
            plume_img_copy = plume_img.copy()
            cv2.polylines(plume_img_copy, [pts], True, (255, 0, 0, 255), thickness=2, lineType=2)
            sub_plume_img = plume_img_copy[np.min(annotator.y) - 20:np.max(annotator.y) + 20,
                                           np.min(annotator.x) - 20:np.max(annotator.x) + 20]
            plt.figure(figsize=(10, 5))
            plt.imshow(sub_plume_img)
            plt.savefig(r"../../data/processed/plume_imgs/" + str(plume_id) + '.png', bbox_inches='tight')
            plt.close()

        # ask if they want to digitise some more?
        arg = raw_input("Do you want to digitise more plumes? [Y,n]")
        if arg.lower() not in ["", "y", "yes", 'ye']:
            break

    return smoke_polygons, background_rectangles, plume_ids


def get_plume_pixels(img, image_pt):
    matrix = np.zeros((img.shape[0], img.shape[1]))
    image_pt_reshape = np.array(image_pt).reshape(-1, 1, 2).squeeze()
    cv2.drawContours(matrix, [image_pt_reshape], -1, (1), thickness=-1)
    return np.nonzero(matrix)


# def load_myd021km(myd021km):
#
#     myd021km_data = {}
#
#     # extract the radiances
#     for group in ['EV_250_Aggr1km_RefSB', 'EV_500_Aggr1km_RefSB', 'EV_1KM_RefSB', 'EV_1KM_Emissive']:
#         group_attributes = myd021km.select(group).attributes()
#         for b, band in enumerate(group_attributes['band_names'].split(',')):
#             # TODO check that the pixels being extracted are in the correct location
#             myd021km_data['band_' + band] = (myd021km.select(group)[b, :, :] -
#                                         group_attributes['radiance_offsets'][b]) * \
#                                         group_attributes['radiance_scales'][b]
#
#     for group in ["SensorZenith", "SensorAzimuth", "SolarZenith", "SolarAzimuth", "Latitude", "Longitude"]:
#         # we drop the last pixel due to the fact that the modis x axis is not a whole number
#         # when divided into five.  So we end up with one pixel too many, this has very limited impact
#         # impact onthe accuracy as it shifts each pixel by 1/1354 - very small.
#         myd021km_data[group] = ndimage.zoom(myd021km.select(group).get(), 5, order=1)[:, :-1]
#
#     return myd021km_data
#
#
# def extract_pixel_info(y, x, myd021km_data, fname, plume_id,  plumes_list):
#
#     row_dict = {}
#
#     row_dict['pixel_id'] = uuid.uuid4()
#     row_dict['plume_id'] = plume_id
#     row_dict['line'] = y
#     row_dict['sample'] = x
#     row_dict['sensor'] = "MYD"
#     row_dict['filename'] = fname
#
#     # extract the data
#     for k in myd021km_data:
#         row_dict[k] = myd021km_data[k][y, x]
#
#     # lastly append to the data dictionary
#     plumes_list.append(row_dict)


def extract_background_bounds(background, plume_id, background_list):
    row_dict = {}
    row_dict['plume_id'] = plume_id
    row_dict['bg_extent'] = background

    background_list.append(row_dict)


def extract_plume_bounds(plume, fname, plume_id, plumes_list):
    row_dict = {}

    row_dict['plume_id'] = plume_id
    row_dict['sensor'] = "MYD"
    row_dict['filename'] = fname
    row_dict['plume_extent'] = plume

    # lastly append to the data dictionary
    plumes_list.append(row_dict)


def main():
    """ Loads MODIS data files for the specified geographic region,
        timeframe, and time stamps.  The user can then process these
        images and extract smoke plumes.
    """

    try:
        myd021km_plume_df = pd.read_pickle(r"../../data/interim/myd021km_plumes_df.pickle")
        myd021km_bg_df = pd.read_pickle(r"../../data/interim/myd021km_bg_df.pickle")
    except:
        logger.info("myd021km dataframe does not exist, creating now")
        myd021km_plume_df = pd.DataFrame()
        myd021km_bg_df = pd.DataFrame()

    for myd021km_fname in os.listdir(r"../../data/raw/l1b"):

        logger.info("Processing modis granule: " + myd021km_fname)

        try:
            timestamp_myd = re.search("[0-9]{7}[.][0-9]{4}[.]", myd021km_fname).group()
        except Exception, e:
            logger.warning("Could not extract time stamp from: ", myd021km_fname, "moving on to next file")
            continue

        try:
            if datetime.strptime(timestamp_myd, '%Y%j.%H%M.') < \
                    datetime.strptime(re.search("[0-9]{7}[.][0-9]{4}[.]",
                                                myd021km_plume_df['filename'].iloc[-1]).group(), '%Y%j.%H%M.'):
                continue

            elif myd021km_plume_df['filename'].str.contains(myd021km_fname).any():
                continue
        except:
            logger.info("filename column not in dataframe - if the dataframe has just been created no problem!")

        myd14_fname = [f for f in os.listdir(r"../../data/raw/frp") if timestamp_myd in f]

        if len(myd14_fname) > 1:
            logger.warning("More that one frp granule matched " + myd021km_fname + "selecting 0th option")
        myd14_fname = myd14_fname[0]

        myd14 = read_myd14(os.path.join(r"../../data/raw/frp/", myd14_fname))
        myd021km = read_myd021km(os.path.join(r"../../data/raw/l1b", myd021km_fname))

        # do the digitising
        myd14_fire_mask = firemask_myd14(myd14)
        img = fcc_myd021km(myd021km, myd14_fire_mask)
        smoke_polygons, background_rectangles, plume_ids = digitise(img)
        if (smoke_polygons is None) | (background_rectangles is None):
            continue

        # if we have a digitisation load in the myd021km data
        # myd021km_data = load_myd021km(myd021km)

        # process plumes and backgrounds
        plumes_list = []
        background_list = []
        for plume, background, plume_id in zip(smoke_polygons, background_rectangles, plume_ids):
            extract_background_bounds(background, plume_id, background_list)
            extract_plume_bounds(plume, myd021km_fname, plume_id, plumes_list)

            # save the plot with the plume_id

            # plume_pixels = get_plume_pixels(img, plume)
            # for y, x in zip(plume_pixels[0], plume_pixels[1]):
            #     extract_pixel_info(int(y), int(x), myd021km_data, myd021km_fname, plume_id, plumes_list)

        # covert pixel/background lists to dataframes and concatenate to main dataframes
        temp_plume_df = pd.DataFrame(plumes_list)
        temp_bg_df = pd.DataFrame(background_list)
        myd021km_plume_df = pd.concat([myd021km_plume_df, temp_plume_df])
        myd021km_bg_df = pd.concat([myd021km_bg_df, temp_bg_df])

        myd021km_plume_df.to_pickle(r"../../data/interim/myd021km_plumes_df.pickle")
        myd021km_bg_df.to_pickle(r"../../data/interim/myd021km_bg_df.pickle")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
