# -*- coding: utf-8 -*-
import os
import logging
from dotenv import find_dotenv, load_dotenv

import re
import uuid

import numpy as np
import pandas as pd

import scipy.ndimage as ndimage
from pyhdf.SD import SD, SDC
from skimage import exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import cv2

import src.config.filepaths as filepaths

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

        # set up the rectangle to hold the background
        self.rect = Rectangle((0, 0), 1, 1,  edgecolor='black', facecolor='none')
        self.ax.add_patch(self.rect)
        self.x0_rect = None
        self.y0_rect = None
        self.x1_rect = None
        self.y1_rect = None

        # set up the events
        self.ax.figure.canvas.mpl_connect('button_press_event', self.click)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.release)

    def click(self, event):
        if event.button == 2:
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
    smoke_rectangles = []

    plt.figure(figsize=(30, 15))
    plt.imshow(img, interpolation='nearest')
    plt.show()
    arg = raw_input("Do you want to digitise this plume?: [y, N]")
    if arg.lower() in ['', 'no', 'n']:
        return None

    while True:

        # first set up the annotator and show the image
        annotator = Annotate(img_copy)
        plt.show()

        # show them what they have digitised, and check if they are OK with that
        # if they are append the polygon, and modify the RGB to reflect the digitised region
        if annotator.x0_rect is None:
            print "you must define a plume rectangle"
            continue

        digitised_copy = img_copy.copy()
        cv2.rectangle(digitised_copy,
                      (annotator.x0_rect,
                       annotator.y0_rect),
                      (annotator.x1_rect,
                       annotator.y1_rect),
                      (0, 0, 0, 255))

        arg = raw_input("Are you happy with this plume rectangle? [Y,n]")
        if arg.lower() in ["", "y", "yes", 'ye']:
            smoke_rectangles.append((annotator.x0_rect, annotator.x1_rect, annotator.y0_rect, annotator.y1_rect))
            img_copy = digitised_copy

        # ask if they want to digitise some more?
        arg = raw_input("Do you want to digitise more plumes? [Y,n]")
        if arg.lower() not in ["", "y", "yes", 'ye']:
            break

    return smoke_rectangles


def get_rect_pixels(image_pt):
    x_pts = np.arange(image_pt[0], image_pt[1], 1)
    y_pts = np.arange(image_pt[2], image_pt[3], 1)
    x_grid, y_grid = np.meshgrid(x_pts, y_pts)
    return y_grid.flatten(), x_grid.flatten()


def load_myd021km(myd021km):

    myd021km_data = {}

    # extract the radiances
    for group in ['EV_250_Aggr1km_RefSB', 'EV_500_Aggr1km_RefSB', 'EV_1KM_RefSB', 'EV_1KM_Emissive']:
        group_attributes = myd021km.select(group).attributes()
        for b, band in enumerate(group_attributes['band_names'].split(',')):
            # TODO check that the pixels being extracted are in the correct location
            myd021km_data['band_' + band] = (myd021km.select(group)[b, :, :] -
                                        group_attributes['radiance_offsets'][b]) * \
                                        group_attributes['radiance_scales'][b]

    for group in ["SensorZenith", "SensorAzimuth", "SolarZenith", "SolarAzimuth", "Latitude", "Longitude"]:
        # we drop the last pixel due to the fact that the modis x axis is not a whole number
        # when divided into five.  So we end up with one pixel too many, this has very limited impact
        # impact onthe accuracy as it shifts each pixel by 1/1354 - very small.
        myd021km_data[group] = ndimage.zoom(myd021km.select(group).get(), 5, order=1)[:, :-1]

    return myd021km_data


def extract_pixel_info(y, x, myd021km_data, fname, rect_id,  rect_list):

    row_dict = {}

    row_dict['pixel_id'] = uuid.uuid4()
    row_dict['rect_id'] = rect_id
    row_dict['line'] = y
    row_dict['sample'] = x
    row_dict['sensor'] = "MYD"
    row_dict['filename'] = fname

    # extract the data
    for k in myd021km_data:
        row_dict[k] = myd021km_data[k][y, x]

    # lastly append to the data dictionary
    rect_list.append(row_dict)


def main():
    """ Loads MODIS data files for the specified geographic region,
        timeframe, and time stamps.  The user can then process these
        images and extract smoke plumes.
    """

    try:
        myd021km_ml_df = pd.read_pickle(filepaths.path_to_ml_smoke_plume_masks)
    except:
        logger.info("myd021km dataframe does not exist, creating now")
        myd021km_ml_df = pd.DataFrame()

    for myd021km_fname in os.listdir(filepaths.path_to_modis_l1b):

        try:
            if myd021km_ml_df['filename'].str.contains(myd021km_fname).any():
                continue
        except:
            logger.info("filename column not in dataframe - if the dataframe has just been created no problem!")

        try:
            timestamp_myd = re.search("[0-9]{7}[.][0-9]{4}[.]", myd021km_fname).group()
        except Exception, e:
            logger.warning("Could not extract time stamp from: ", myd021km_fname, "moving on to next file")
            continue

        myd14_fname = [f for f in os.listdir(filepaths.path_to_myd14) if timestamp_myd in f]

        if len(myd14_fname) > 1:
            logger.warning("More that one frp granule matched " + myd021km_fname + "selecting 0th option")
        myd14_fname = myd14_fname[0]

        myd14 = read_myd14(os.path.join(filepaths.path_to_myd14, myd14_fname))
        myd021km = read_myd021km(os.path.join(filepaths.path_to_modis_l1b, myd021km_fname))

        # do the digitising
        myd14_fire_mask = firemask_myd14(myd14)
        img = fcc_myd021km(myd021km, myd14_fire_mask)
        smoke_rectangle = digitise(img)
        if smoke_rectangle is None:
            continue

        # if we have a digitisation load in the myd021km data
        myd021km_data = load_myd021km(myd021km)

        # process plumes and backgrounds
        rect_list = []
        for rect in smoke_rectangle:

            rect_id = uuid.uuid4()

            rect_pixels = get_rect_pixels(rect)
            for y, x in zip(rect_pixels[0], rect_pixels[1]):
                extract_pixel_info(int(y), int(x), myd021km_data, myd021km_fname, rect_id, rect_list)
        # covert pixel
        temp_rect_df = pd.DataFrame(rect_list)
        myd021km_ml_df = pd.concat([myd021km_ml_df, temp_rect_df])

        myd021km_ml_df.to_pickle(filepaths.path_to_ml_smoke_plume_masks)


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
