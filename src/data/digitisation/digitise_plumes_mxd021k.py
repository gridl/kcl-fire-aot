# -*- coding: utf-8 -*-
import os
import logging

import re

import numpy as np
import pandas as pd

import matplotlib
#matplotlib.use('TkAgg')

import pyresample as pr
import scipy.ndimage as ndimage
from datetime import datetime
from pyhdf.SD import SD, SDC
from netCDF4 import Dataset
from skimage import exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.lines import Line2D
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


def image_seen(myd021km_fname):
    try:
        with open(filepaths.path_to_processed_filelist_modis, 'r+') as txt_file:
            if myd021km_fname in txt_file.read():
                logger.info( myd021km_fname + " already processed")
                return True
            else:
                txt_file.write(myd021km_fname + '\n')
                return False
    except Exception, e:
        logger.warning("Could not check time filename for : " + myd021km_fname + ". With error: " + str(e))
        return False  # if we cannot do it, lets just assume we haven't seen the image before


def get_modis_fname(path, timestamp_myd, myd021km_fname):
    fname = [f for f in os.listdir(path) if timestamp_myd in f]
    if len(fname) > 1:
        logger.warning("More that one frp granule matched " + myd021km_fname + "selecting 0th option")
        return fname[0]
    elif len(fname) == 1:
        return fname[0]
    else:
        return ''


def get_orac_fname(path, timestamp_myd):
    t = datetime.strptime(timestamp_myd, '%Y%j.%H%M.')
    t = datetime.strftime(t, '%Y%m%d%H%M')
    fname = [f for f in os.listdir(path) if t in f]
    return fname[0]


def read_hdf(f):
    return SD(f, SDC.READ)


def read_orac(f):
    return Dataset(f)


def fires_myd14(myd14_data):
    return np.where(myd14_data.select('fire mask').get() >= 7)


def aod_myd04(myd04):
    aod_params = myd04.select("AOD_550_Dark_Target_Deep_Blue_Combined").attributes()
    aod = myd04.select("AOD_550_Dark_Target_Deep_Blue_Combined").get()
    aod = (aod + aod_params['add_offset']) * aod_params['scale_factor']
    aod = ndimage.zoom(aod, 10, order=1)
    return aod


def aod_orac(orac_ds):
    return orac_ds.variables['cot'][:], orac_ds.variables['costjm'][:]


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def tcc_myd021km(mod_data):
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

    rgb = np.dstack((r, g, b))
    return rgb


def fcc_myd021km(mod_data):
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

    g_min, g_max = np.percentile(g, (mini, maxi))
    g = exposure.rescale_intensity(g, in_range=(g_min, g_max))

    b_min, b_max = np.percentile(b, (mini, maxi))
    b = exposure.rescale_intensity(b, in_range=(b_min, b_max))

    rgb = np.dstack((r, g, b))
    return rgb


class Annotate(object):
    def __init__(self, fcc, tcc, mod_aod, orac_aod, orac_cost, fires, ax,
                 plume_polygons, background_polygons, plume_vectors):
        self.ax = ax
        self.fcc = fcc
        self.tcc = tcc
        self.mod_aod = mod_aod
        self.orac_aod = orac_aod
        self.orac_cost = orac_cost
        self.im = self.ax.imshow(self.orac_aod, interpolation='none', cmap='viridis')
        if fires is not None:
            self.plot = self.ax.plot(fires[1], fires[0], 'r.')
        self.background_polygons = self._add_polygons_to_axis(background_polygons, 'Blues_r')
        self.plume_polygons = self._add_polygons_to_axis(plume_polygons, 'Reds_r')
        self.plume_vectors = self._add_vectors_to_axis(plume_vectors)

        # set up the point holders for the plume and background polygons
        self.plume_x = []
        self.plume_y = []
        self.background_x = []
        self.background_y = []
        self.vector_x = []
        self.vector_y = []

        # set up the digitising patch
        self.plume_p = Circle((1,1))
        self.background_p = Circle((1,1))
        self.vector_p = Circle((1,1))

        # set up the events
        self.ax.figure.canvas.mpl_connect('button_press_event', self.click)

        # set up check to see if we keep annotating
        self.do_annotation = True

        # set up default digitisation type as plume (0 is plume, 1 is background, 2 is vector)
        self.type = 0

        # set up radio buttons
        self.axcolor = 'lightgoldenrodyellow'

        self.rax_digitise = plt.axes([0.05, 0.7, 0.15, 0.15], facecolor=self.axcolor)
        self.radio_disitise = RadioButtons(self.rax_digitise, ('Digitise', 'Stop'))
        self.radio_disitise.on_clicked(self.annotation_func)

        self.rax_discard = plt.axes([0.05, 0.5, 0.15, 0.15], facecolor=self.axcolor)
        self.radio_discard = RadioButtons(self.rax_discard, ('Keep', 'Discard'))
        self.radio_discard.on_clicked(self.discard_func)

        self.rax_type = plt.axes([0.05, 0.3, 0.15, 0.15], facecolor=self.axcolor)
        self.radio_type = RadioButtons(self.rax_type, ('Plume', 'Background', 'Vector'))
        self.radio_type.on_clicked(self.type_func)

        self.rax_image = plt.axes([0.05, 0.1, 0.15, 0.15], facecolor=self.axcolor)
        self.radio_image = RadioButtons(self.rax_image, self._radio_labels())
        self.radio_image.on_clicked(self.image_func)

        self.cax = plt.axes([0.8, 0.1, 0.05, 0.8])
        self.cbar = plt.colorbar(self.im, self.cax)

    def _add_polygons_to_axis(self, polygons, cmap):
        patches = [Polygon(verts, True) for verts in polygons]
        colors = [0] * len(patches)
        p = PatchCollection(patches, cmap=cmap, alpha=0.8)
        p.set_array(np.array(colors))
        return self.ax.add_collection(p)

    def _add_vectors_to_axis(self, vectors):
        for v in vectors:
            x1, y1 = v[0]
            x2, y2 = v[1]
            self.ax.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.5, head_length=1, fc='k', ec='k')

    def _radio_labels(self):

        labels = []
        if self.orac_aod is not None:
            labels.append('ORAC_AOD')
            labels.append('ORAC_COST')
        if self.mod_aod is not None:
            labels.append('MOD_AOD')

        # FCC and TCC always present
        labels.append('FCC')
        labels.append('TCC')

        return tuple(labels)

    def _radio_label_mapping(self):

        label_mapping = {}
        if self.orac_aod is not None:
            label_mapping['ORAC_AOD'] = self.orac_aod
            label_mapping['ORAC_COST'] = self.orac_cost
        if self.mod_aod is not None:
            label_mapping['MOD_AOD'] = self.mod_aod

        # FCC and TCC always present
        label_mapping['FCC'] = self.fcc
        label_mapping['TCC'] = self.tcc
        return label_mapping

    def annotation_func(self, label):
        anno_dict = {'Digitise': True, 'Stop': False}
        self.do_annotation = anno_dict[label]

    def discard_func(self, label):
        keep_dict = {'Keep': False, 'Discard': True}
        if keep_dict[label] and self.type:
            self.plume_x = []
            self.plume_y = []
        elif keep_dict[label] and not self.type:
            self.background_x = []
            self.background_y = []

    def type_func(self, label):
        type_dict = {'Plume': 0, 'Background': 1, 'Vector': 2}
        self.type = type_dict[label]

    def image_func(self, label):

        image_dict = self._radio_label_mapping()
        im_data = image_dict[label]
        self.im.set_data(im_data)

        if label == "ORAC_AOD":
            self.im.set_clim(vmax=5, vmin=0)
            self.im.set_cmap('viridis')
        if label == "ORAC_COST":
            self.im.set_clim(vmax=20, vmin=0)
            self.im.set_cmap('inferno_r')
        if label == "MOD_AOD":
            self.im.set_clim(vmax=5, vmin=0)
            self.im.set_cmap('viridis')

        plt.draw()

    def click(self, event):
        if event.button == 3:
            if self.type == 0:
                self.plume_x.append(int(event.xdata))
                self.plume_y.append(int(event.ydata))
                if len(self.plume_x) < 3:
                    self.plume_p = Circle((event.xdata, event.ydata), radius=0.25, facecolor='red', edgecolor='black')
                    self.ax.add_patch(self.plume_p)
                else:
                    self.plume_p.remove()
                    self.plume_p = Polygon(zip(self.plume_x, self.plume_y), color='red', alpha=0.5)
                    plume_p = self.ax.add_patch(self.plume_p)
                self.ax.figure.canvas.draw()
            elif self.type == 1:
                self.background_x.append(int(event.xdata))
                self.background_y.append(int(event.ydata))
                if len(self.background_x) < 3:
                    self.background_p = Circle((event.xdata, event.ydata), radius=0.25, facecolor='blue',
                                               edgecolor='black')
                    self.ax.add_patch(self.background_p)
                else:
                    self.background_p.remove()
                    self.background_p = Polygon(zip(self.background_x, self.background_y), color='blue', alpha=0.5)
                    background_p = self.ax.add_patch(self.background_p)
                self.ax.figure.canvas.draw()

            elif self.type == 2:
                self.vector_x.append(int(event.xdata))
                self.vector_y.append(int(event.ydata))
                if len(self.vector_x) == 1:
                    self.vector_p = Circle((event.xdata, event.ydata), radius=1, facecolor='black',
                                           edgecolor='blue')
                    self.ax.add_patch(self.vector_p)
                elif len(self.vector_x) == 2:
                    self.vector_p = Line2D([self.vector_x[0], self.vector_x[1]],
                                            [self.vector_y[0], self.vector_y[1]], lw=2, color='black')
                    self.ax.add_line(self.vector_p)
                self.ax.figure.canvas.draw()


def digitise(fcc, tcc, mod_aod, orac_aod, orac_cost, fires, myd021km_fname):

    plume_polygons = []
    background_polygons = []
    plume_vectors = []

    do_annotation = True
    while do_annotation:

        fig, ax = plt.subplots(1, figsize=(11, 8))
        plt.title(myd021km_fname)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # first set up the annotator
        annotator = Annotate(fcc, tcc, mod_aod, orac_aod, orac_cost, fires, ax,
                             plume_polygons, background_polygons, plume_vectors)

        # then show the image
        plt.show()

        # get the polygon points from the closed image, only keep if plume and background polygons
        plume_pts = zip(annotator.plume_x, annotator.plume_y)
        background_pts = zip(annotator.background_x, annotator.background_y)
        plume_vector = zip(annotator.vector_x, annotator.vector_y)

        if plume_pts and background_pts:
            plume_polygons.append(plume_pts)
            background_polygons.append(background_pts)
            plume_vectors.append(plume_vector)

        do_annotation = annotator.do_annotation

    plt.close(fig)

    return plume_polygons, background_polygons, plume_vectors


def append_to_list(plume, background, vector, fname, plumes_list):
    row_dict = {}

    row_dict['sensor'] = "MYD"
    row_dict['filename'] = fname
    row_dict['plume_extent'] = plume
    row_dict['background_extent'] = background
    row_dict['plume_vector'] = vector

    # lastly append to the data dictionary
    plumes_list.append(row_dict)


def main():
    """ Loads MODIS data files for the specified geographic region,
        timeframe, and time stamps.  The user can then process these
        images and extract smoke plumes.
    """
    myd021km_plume_df = load_df(filepaths.path_to_smoke_plume_polygons_modis)

    for myd021km_fname in os.listdir(filepaths.path_to_modis_l1b):

        logger.info("Processing modis granule: " + myd021km_fname)

        timestamp_myd = get_timestamp(myd021km_fname)
        if not timestamp_myd:
            continue

        if image_seen(myd021km_fname):
            continue

        try:
            myd14_fname = get_modis_fname(filepaths.path_to_modis_frp, timestamp_myd, myd021km_fname)
            myd04_fname = get_modis_fname(filepaths.path_to_modis_aod, timestamp_myd, myd021km_fname)
            orac_fname = get_orac_fname(filepaths.path_to_orac_aod, timestamp_myd)
        except Exception, e:
            logger.warning('Could not load aux file for:' + myd021km_fname + '. Failed with ' + str(e))
            continue
        try:
            myd021km = read_hdf(os.path.join(filepaths.path_to_modis_l1b, myd021km_fname))
            fcc = fcc_myd021km(myd021km)
            tcc = tcc_myd021km(myd021km)
        except Exception, e:
            logger.warning('Could not read the input file: ' + myd021km_fname + '. Failed with ' + str(e))
            continue

        # if filenames load in data
        fires, mod_aod, orac_aod, orac_cost = None, None, None, None
        if myd14_fname:
            try:
                myd14 = read_hdf(os.path.join(filepaths.path_to_modis_frp, myd14_fname))
                fires = fires_myd14(myd14)
            except Exception, e:
                logger.warning('Could not read myd14 file: ' + myd14_fname)
        if myd04_fname:
            try:
                myd04 = read_hdf(os.path.join(filepaths.path_to_modis_aod, myd04_fname))
                mod_aod = aod_myd04(myd04)
            except Exception, e:
                logger.warning('Could not read myd04 file: ' + myd04_fname)
        if orac_fname:
            orac_data = read_orac(os.path.join(filepaths.path_to_orac_aod, orac_fname))
            orac_aod, orac_cost = aod_orac(orac_data)

        # do the digitising
        plume_polygons, background_polygons, plume_vectors = digitise(fcc, tcc,
                                                                      mod_aod, orac_aod,
                                                                      orac_cost,
                                                                      fires,
                                                                      myd021km_fname)
        if plume_polygons is None:
            continue

        # process plumes and backgrounds
        plumes_list = []
        for pp, bp, pv in zip(plume_polygons, background_polygons, plume_vectors):
            append_to_list(pp, bp, pv, myd021km_fname, plumes_list)

        # covert pixel/background lists to dataframes and concatenate to main dataframes
        temp_plume_df = pd.DataFrame(plumes_list)
        myd021km_plume_df = pd.concat([myd021km_plume_df, temp_plume_df])
        myd021km_plume_df.to_pickle(filepaths.path_to_smoke_plume_polygons_modis)


if __name__ == '__main__':

    #plt.ioff()
    matplotlib.pyplot.close("all")

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    main()
