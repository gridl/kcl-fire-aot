# -*- coding: utf-8 -*-
import os
import logging
from dotenv import find_dotenv, load_dotenv

import re
import uuid

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Qt5Agg')

import scipy.ndimage as ndimage
from datetime import datetime
from pyhdf.SD import SD, SDC
from skimage import exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.widgets import RadioButtons
from matplotlib.colors import LogNorm


import src.config.filepaths as filepaths


def load_df(path):
    try:
        myd021km_plume_df = pd.read_pickle(path)
    except:
        logger.info("myd021km dataframe does not exist, creating now")
        myd021km_plume_df = pd.DataFrame()
    return myd021km_plume_df


def get_timestamp(myd021km_fname):
    try:
        return re.search("[0-9]{7}[.][0-9]{4}[.]", myd021km_fname).group()
    except Exception, e:
        logger.warning("Could not extract time stamp from: " + myd021km_fname + " with error: " + str(e))
        return ''


def image_seen(myd021km_fname, myd021km_plume_df):
    try:
        return myd021km_plume_df['filename'].str.contains(myd021km_fname).any()
    except Exception, e:
        logger.warning("Could not check time filename for : " + myd021km_fname + ". With error: " + str(e))
        return False  # if we cannot do it, lets just assume we haven't seen the image before


def image_not_proccesed_but_seen(timestamp_myd, myd021km_plume_df):
    try:
        image_time = datetime.strptime(timestamp_myd, '%Y%j.%H%M.')
        df_firsttime = datetime.strptime(re.search("[0-9]{7}[.][0-9]{4}[.]",
                                    myd021km_plume_df['filename'].iloc[0]).group(), '%Y%j.%H%M.')
        df_lasttime = datetime.strptime(re.search("[0-9]{7}[.][0-9]{4}[.]",
                                    myd021km_plume_df['filename'].iloc[-1]).group(), '%Y%j.%H%M.')
        # here we check if we have seen this image before
        return (image_time > df_firsttime) & (image_time < df_lasttime)
    except Exception, e:
        logger.warning("Could check time stamp for : " + timestamp_myd + ". With error: " + str(e))
        return False  # if we cannot do it, lets just assume we haven't seen the image before


def get_fname(path, timestamp_myd, myd021km_fname):
    fname = [f for f in os.listdir(path) if timestamp_myd in f]
    if len(fname) > 1:
        logger.warning("More that one frp granule matched " + myd021km_fname + "selecting 0th option")
    return fname[0]


def read_myd(f):
    return SD(f, SDC.READ)


def firemask_myd14(myd14_data):
    return myd14_data.select('fire mask').get() >= 7


def aod_myd04_3K(myd04_3K):
    aod_params = myd04_3K.select("Optical_Depth_Land_And_Ocean").attributes()
    aod = myd04_3K.select("Optical_Depth_Land_And_Ocean").get()
    aod = (aod + aod_params['add_offset']) * aod_params['scale_factor']
    aod[aod < 0] = 0
    aod[aod > 0.5] = 0.5
    aod = ndimage.zoom(aod, 3, order=1)
    return aod


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def tcc_myd021km(mod_data, fire_mask):
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
    def __init__(self, fcc, tcc, aod, ax, polygons):
        self.ax = ax
        self.fcc = fcc
        self.tcc = tcc
        self.aod = aod
        self.im = self.ax.imshow(self.aod, interpolation='none')
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

        # set up the patch
        self.p = Circle((1,1))

        # set up the events
        self.ax.figure.canvas.mpl_connect('button_press_event', self.click)

        # set up check to see if we keep annotating
        self.do_annotation = True

        # set up radio buttons
        self.axcolor = 'lightgoldenrodyellow'

        self.rax_digitise = plt.axes([0.05, 0.7, 0.15, 0.15], facecolor=self.axcolor)
        self.radio_disitise = RadioButtons(self.rax_digitise, ('Digitise', 'Stop'))
        self.radio_disitise.on_clicked(self.annotation_func)

        self.rax_discard = plt.axes([0.05, 0.4, 0.15, 0.15], facecolor=self.axcolor)
        self.radio_discard = RadioButtons(self.rax_discard, ('Keep', 'Discard'))
        self.radio_discard.on_clicked(self.discard_func)

        self.rax_image = plt.axes([0.05, 0.1, 0.15, 0.15], facecolor=self.axcolor)
        self.radio_image = RadioButtons(self.rax_image, ('AOD', 'FCC', 'TCC'))
        self.radio_image.on_clicked(self.image_func)

    def annotation_func(self, label):
        anno_dict = {'Digitise': True, 'Stop': False}
        self.do_annotation = anno_dict[label]

    def discard_func(self, label):
        keep_dict = {'Keep': False, 'Discard': True}
        if keep_dict[label]:
            self.x = []
            self.y = []

    def image_func(self, label):
        image_dict = {'AOD': self.aod, 'FCC': self.fcc, 'TCC': self.tcc, }
        im_data = image_dict[label]
        self.im.set_data(im_data)
        plt.draw()


    def click(self, event):
        if event.button == 3:
            self.x.append(int(event.xdata))
            self.y.append(int(event.ydata))
            if len(self.x) < 3:
                self.p = Circle((event.xdata, event.ydata), radius=0.25, facecolor='red', edgecolor='black')
                self.ax.add_patch(self.p)
            else:
                self.p.remove()
                self.p = Polygon(zip(self.x, self.y), color='red', alpha=0.5)
                p = self.ax.add_patch(self.p)
            self.ax.figure.canvas.draw()


def digitise(fcc, tcc, aod):

    smoke_polygons = []
    plume_ids = []

    do_annotation = True
    while do_annotation:

        fig, ax = plt.subplots(1, figsize=(12,20))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # first set up the annotator
        annotator = Annotate(fcc, tcc, aod, ax, smoke_polygons)

        # then show the image
        plt.show()

        # get the polygon points from the closed image
        pts = zip(annotator.x, annotator.y)
        if pts:
            smoke_polygons.append(pts)

        do_annotation = annotator.do_annotation

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
    myd021km_plume_df = load_df(filepaths.path_to_smoke_plume_masks)

    for myd021km_fname in os.listdir(filepaths.path_to_modis_l1b):

        logger.info("Processing modis granule: " + myd021km_fname)

        timestamp_myd = get_timestamp(myd021km_fname)
        if not timestamp_myd:
            continue

        if image_seen(myd021km_fname, myd021km_plume_df):
            continue

        if image_not_proccesed_but_seen(timestamp_myd, myd021km_plume_df):
            continue

        myd14_fname = get_fname(filepaths.path_to_modis_frp, timestamp_myd, myd021km_fname)
        myd04_3K_fname = get_fname(filepaths.path_to_modis_aod, timestamp_myd, myd021km_fname)

        myd14 = read_myd(os.path.join(filepaths.path_to_modis_frp, myd14_fname))
        myd04_3K = read_myd(os.path.join(filepaths.path_to_modis_aod, myd04_3K_fname))
        myd021km = read_myd(os.path.join(filepaths.path_to_modis_l1b, myd021km_fname))

        # do the digitising
        myd14_fire_mask = firemask_myd14(myd14)
        aod = aod_myd04_3K(myd04_3K)
        fcc = fcc_myd021km(myd021km, myd14_fire_mask)
        tcc = tcc_myd021km(myd021km, myd14_fire_mask)
        smoke_polygons, plume_ids = digitise(fcc, tcc, aod)
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

    #plt.ioff()

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
