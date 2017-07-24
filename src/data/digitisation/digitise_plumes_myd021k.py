# -*- coding: utf-8 -*-
import os
import logging
from dotenv import find_dotenv, load_dotenv

import re
import uuid

import numpy as np
import pandas as pd

# import matplotlib
# matplotlib.use('Qt5Agg')

import scipy.ndimage as ndimage
from datetime import datetime
from pyhdf.SD import SD, SDC
from skimage import exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.collections import PatchCollection


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
    def __init__(self, im, polygons):
        self.f = plt.figure(figsize=(30, 15))
        self.ax = plt.gca()
        self.im = self.ax.imshow(im, interpolation='none')
        patches = [Polygon(verts, True) for verts in polygons]
        p = PatchCollection(patches, cmap='Oranges', alpha=0.8)
        self.polygons = self.ax.add_collection(p)

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

    def click(self, event):
        if event.button == 3:
            self.x.append(int(event.xdata))
            self.y.append(int(event.ydata))
            self.ax.add_patch(Circle((event.xdata, event.ydata), radius=0.25, facecolor='red', edgecolor='black'))
            self.ax.figure.canvas.draw()


def run_annotator(img, smoke_polygons):
    annotator = Annotate(img, smoke_polygons)
    plt.show()
    return annotator


def show_image(img, pts=None):

    fig = plt.figure(figsize=(30, 15))
    ax = fig.add_subplot(111)
    ax.imshow(img, interpolation="nearest")
    if pts is not None:
        p = plt.Polygon(pts, color='Red', alpha=0.6)
        ax.add_patch(p)
    plt.show()


def digitise(img):

    smoke_polygons = []
    plume_ids = []

    show_image(img)

    arg = raw_input("Do you want to digitise this plume?: [y, N]")
    if arg.lower() in ['', 'no', 'n']:
        return None, None, None

    annotate = True
    while annotate:

        # first set up the annotator and show the image
        annotator = run_annotator(img, smoke_polygons)

        # show them what they have digitised, and check if they are OK with that
        # if they are append the polygon, and modify the RGB to reflect the digitised region
        pts = zip(annotator.x, annotator.y)

        if not pts:
            print "you must select define a polygon containing smoke pixels"
            continue

        show_image(img, pts)

        arg = raw_input("Are you happy with this plume digitisation? [Y,n]")
        if arg.lower() in ["", "y", "yes", 'ye']:
            smoke_polygons.append(zip(annotator.x, annotator.y))

            # store the plume id
            plume_id = uuid.uuid4()
            plume_ids.append(plume_id)

        # ask if they want to digitise some more?
        arg = raw_input("Do you want to digitise more plumes? [Y,n]")
        if arg.lower() not in ["", "y", "yes", 'ye']:
            annotate = False

        plt.close()

    return smoke_polygons, plume_ids


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
        myd021km_plume_df = pd.read_pickle(filepaths.path_to_smoke_plume_masks)
    except:
        logger.info("myd021km dataframe does not exist, creating now")
        myd021km_plume_df = pd.DataFrame()

    for myd021km_fname in os.listdir(filepaths.path_to_modis_l1b):

        logger.info("Processing modis granule: " + myd021km_fname)

        try:
            timestamp_myd = re.search("[0-9]{7}[.][0-9]{4}[.]", myd021km_fname).group()
        except Exception, e:
            logger.warning("Could not extract time stamp from: " + myd021km_fname + " moving on to next file")
            continue

        try:
            if datetime.strptime(timestamp_myd, '%Y%j.%H%M.') < \
                    datetime.strptime(re.search("[0-9]{7}[.][0-9]{4}[.]",
                                                myd021km_plume_df['filename'].iloc[-1]).group(), '%Y%j.%H%M.'):
                continue

            elif myd021km_plume_df['filename'].str.contains(myd021km_fname).any():
                logger.info(myd021km_fname + ' already processed (contained in DF) moving on')
                continue
        except:
            logger.info("filename column not in dataframe - if the dataframe has just been created no problem!")

        myd14_fname = [f for f in os.listdir(filepaths.path_to_modis_frp) if timestamp_myd in f]

        if len(myd14_fname) > 1:
            logger.warning("More that one frp granule matched " + myd021km_fname + "selecting 0th option")
        myd14_fname = myd14_fname[0]

        myd14 = read_myd14(os.path.join(filepaths.path_to_modis_frp, myd14_fname))
        myd021km = read_myd021km(os.path.join(filepaths.path_to_modis_l1b, myd021km_fname))

        # do the digitising
        myd14_fire_mask = firemask_myd14(myd14)
        img = fcc_myd021km(myd021km, myd14_fire_mask)
        smoke_polygons, plume_ids = digitise(img)
        if smoke_polygons is None:
            continue

        # process plumes and backgrounds
        plumes_list = []
        for plume, plume_id in zip(smoke_polygons, plume_ids):
            extract_plume_bounds(plume, myd021km_fname, plume_id, plumes_list)

        # covert pixel/background lists to dataframes and concatenate to main dataframes
        temp_plume_df = pd.DataFrame(plumes_list)
        myd021km_plume_df = pd.concat([myd021km_plume_df, temp_plume_df])
        myd021km_plume_df.to_pickle(filepaths.path_to_smoke_plume_masks)


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
