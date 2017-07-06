# -*- coding: utf-8 -*-
import os
import logging
from dotenv import find_dotenv, load_dotenv

import re
import uuid

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TKAgg')

import scipy.ndimage as ndimage
from pyhdf.SD import SD, SDC
from skimage import exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from datetime import datetime
import cv2

import src.config.filepaths as filepaths


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def read_myd14(myd14_file):
    return SD(myd14_file, SDC.READ)


def firemask_myd14(myd14_data):
    return myd14_data.select('fire mask').get() >= 7


def read_myd021km(local_filename):
    return SD(local_filename, SDC.READ)


def fcc_myd021km(mod_data, fire_mask):
    mod_params_500 = mod_data.select("EV_500_Aggr1km_RefSB").attributes()
    ref_500 = mod_data.select("EV_500_Aggr1km_RefSB").get()

    mod_params_250 = mod_data.select("EV_250_Aggr1km_RefSB").attributes()
    ref_250 = mod_data.select("EV_250_Aggr1km_RefSB").get()

    r = (ref_250[0, :, :] - mod_params_250['radiance_offsets'][0]) * mod_params_250['radiance_scales'][
        0]  # 2.1 microns
    g = (ref_500[1, :, :] - mod_params_500['radiance_offsets'][1]) * mod_params_500['radiance_scales'][
        1]  # 0.8 microns
    b = (ref_500[0, :, :] - mod_params_500['radiance_offsets'][0]) * mod_params_500['radiance_scales'][
        0]  # 0.6 microns

    r = image_histogram_equalization(r)
    g = image_histogram_equalization(g)
    b = image_histogram_equalization(b)

    r = np.round((r * (255 / np.max(r))) * 1).astype('uint8')
    g = np.round((g * (255 / np.max(g))) * 1).astype('uint8')
    b = np.round((b * (255 / np.max(b))) * 1).astype('uint8')

    r[fire_mask] = 255
    g[fire_mask] = 0
    b[fire_mask] = 0

    rgb = np.dstack((r, g, b))
    return rgb


class Annotate(object):
    def __init__(self, im):
        self.f = plt.figure(figsize=(25, 12))



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
    smoke_samples = []

    plt.figure(figsize=(25, 12))
    plt.imshow(img, interpolation='nearest')
    plt.draw()
    plt.pause(1)
    raw_input("<Hit Enter To Close>")
    plt.close()

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


        arg = raw_input("Are you happy with this plume sample? [Y,n]")
        if arg.lower() in ["", "y", "yes", 'ye']:
            smoke_samples.append((annotator.x0_rect, annotator.x1_rect, annotator.y0_rect, annotator.y1_rect))
            img_copy = digitised_copy

        # ask if they want to digitise some more?
        arg = raw_input("Do you want to digitise more plumes? [Y,n]")
        if arg.lower() not in ["", "y", "yes", 'ye']:
            break
    return smoke_samples


def store_sample(sample_bounds, sample_id, filename, sample_list):

    row_dict = {}

    row_dict['sample_bounds'] = sample_bounds
    row_dict['sample_id'] = sample_id
    row_dict['filename'] = filename

    # lastly append to the data dictionary
    sample_list.append(row_dict)


def main():
    """ Loads MODIS data files for the specified geographic region,
        timeframe, and time stamps.  The user can then process these
        images and extract smoke plumes.
    """

    try:
        ml_df = pd.read_pickle(filepaths.path_to_ml_smoke_plume_masks)
    except:
        logger.info("myd021km dataframe does not exist, creating now")
        ml_df = pd.DataFrame()

    for myd021km_fname in os.listdir(filepaths.path_to_modis_l1b):

        try:
            if ml_df['filename'].str.contains(myd021km_fname).any():
                continue
        except:
            logger.info("filename column not in dataframe - if the dataframe has just been created no problem!")


        try:
            if datetime.strptime(timestamp_myd, '%Y%j.%H%M.') < \
                    datetime.strptime(re.search("[0-9]{7}[.][0-9]{4}[.]",
                                                ml_df['filename'].iloc[-1]).group(), '%Y%j.%H%M.'):
                continue

            elif ml_df['filename'].str.contains(myd021km_fname).any():
                continue
        except:
            logger.info("filename column not in dataframe - if the dataframe has just been created no problem!")


        try:
            timestamp_myd = re.search("[0-9]{7}[.][0-9]{4}[.]", myd021km_fname).group()
        except Exception, e:
            logger.warning("Could not extract time stamp from: ", myd021km_fname, "moving on to next file")
            continue

        myd14_fname = [f for f in os.listdir(filepaths.path_to_modis_frp) if timestamp_myd in f]

        if len(myd14_fname) > 1:
            logger.warning("More that one frp granule matched " + myd021km_fname + "selecting 0th option")
        myd14_fname = myd14_fname[0]

        myd14 = read_myd14(os.path.join(filepaths.path_to_modis_frp, myd14_fname))
        myd021km = read_myd021km(os.path.join(filepaths.path_to_modis_l1b, myd021km_fname))

        # do the digitising
        myd14_fire_mask = firemask_myd14(myd14)
        img = fcc_myd021km(myd021km, myd14_fire_mask)
        smoke_samples = digitise(img)
        if smoke_samples is None:
            continue

        # process plumes and backgrounds
        sample_list = []
        for smoke_sample in smoke_samples:

            sample_id = uuid.uuid4()
            store_sample(smoke_sample, sample_id, myd021km_fname, sample_list)

        # covert pixel
        temporary_sample_df = pd.DataFrame(sample_list)
        ml_df = pd.concat([ml_df, temporary_sample_df])

        ml_df.to_pickle(filepaths.path_to_ml_smoke_plume_masks)


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
