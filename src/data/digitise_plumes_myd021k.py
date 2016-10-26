# -*- coding: utf-8 -*-
import os
import logging
from dotenv import find_dotenv, load_dotenv

import re

import numpy as np
from pyhdf.SD import SD, SDC
from skimage import exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cv2


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
        self.im = self.ax.imshow(im, interpolation='none')

        # set up the circle to hold the fires
        self.circ = Circle((0, 0), radius=0, facecolor='none', edgecolor='black')
        self.ax.add_patch(self.circ)
        self.x0_circ = None
        self.y0_circ = None
        self.x1_circ = None
        self.y1_circ = None
        self.rad = None


        # set up the polygon
        self.x = []
        self.y = []
        self.polygons = []

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
            self.x0_circ = int(event.xdata)
            self.y0_circ = int(event.ydata)

    def release(self, event):
        if event.button == 2:
            self.x1_circ = int(event.xdata)
            self.y1_circ = int(event.ydata)
            self.rad = np.sqrt((self.x0_circ - self.x1_circ)**2 +
                               (self.y0_circ - self.y1_circ)**2)
            self.circ.set_radius(self.rad)
            self.circ.center = self.x0_circ,  self.y0_circ
            self.ax.figure.canvas.draw()


def digitise(img):

    img_copy = img.copy()
    smoke_polygons = []
    fire_circles = []

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
        if annotator.x0_circ is None:
            print "you must define a circle containing fire pixels"
            continue

        digitised_copy = img_copy.copy()

        cv2.fillConvexPoly(digitised_copy, np.array(pts), (255, 255, 255, 125))
        cv2.circle(digitised_copy,
                   (annotator.x0_circ,
                    annotator.y0_circ),
                   int(annotator.rad),
                   (0, 0, 0, 255))
        plt.imshow(digitised_copy)
        plt.show()

        happy = raw_input("Are you happy with this plume digitisation? [Y,n]")
        if happy.lower() in ["", "y", "yes", 'ye']:
            smoke_polygons.append(pts)
            fire_circles.append((annotator.x0_circ,
                                 annotator.y0_circ,
                                 annotator.rad))
            img_copy = digitised_copy

        # ask if they want to digitise some more?
        more = raw_input("Do you want to digitise more plumes? [Y,n]")
        if more.lower() not in ["", "y", "yes", 'ye']:
            break

    return smoke_polygons, fire_circles


def make_mask(img, image_pts):
    matrix = np.zeros((img.shape[0], img.shape[1]))
    for image_pt in image_pts:
        image_pt_reshape = np.array(image_pt).reshape(-1, 1, 2).squeeze()
        cv2.drawContours(matrix, [image_pt_reshape], -1, (1), thickness=-1)
    return matrix


def main():
    """ Loads MODIS data files for the specified geographic region,
        timeframe, and time stamps.  The user can then process these
        images and extract smoke plumes.
    """

    for myd021km_fname in os.listdir(r"../../data/raw/l1b"):

        # find the frp file
        try:
            timestamp_myd = re.search("[0-9]{7}[.][0-9]{4}[.]", myd021km_fname).group()
        except Exception, e:
            logger.warning("Could not extract time stamp from: ", myd021km_fname, "moving on to next file")
            continue

        myd14_fname = [f for f in os.listdir(r"../../data/raw/frp") if timestamp_myd in f]

        if len(myd14_fname) > 1:
            logger.warning("More that one frp granule matched " + myd021km_fname + "selecting 0th option")
        myd14_fname = myd14_fname[0]

        # read in the data
        myd14 = read_myd14(os.path.join(r"../../data/raw/frp/", myd14_fname))
        myd021km = read_myd021km(os.path.join(r"../../data/raw/l1b", myd021km_fname))

        # do the digitising
        myd14_fire_mask = firemask_myd14(myd14)
        img = fcc_myd021km(myd021km, myd14_fire_mask)
        smoke_polygons, fire_circles = digitise(img)
        plume_mask = make_mask(img, smoke_polygons)


    # perform manual feature extraction

    # store in database

    # close off data



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