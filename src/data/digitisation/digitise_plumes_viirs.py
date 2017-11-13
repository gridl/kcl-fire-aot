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
import h5py
from skimage import exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
from matplotlib.widgets import RadioButtons
from matplotlib.colors import LogNorm


import src.config.filepaths as filepaths
import src.features.fre_to_tpm.ftt_utils as ut


def load_df(path):
    try:
        myd021km_plume_df = pd.read_pickle(path)
    except:
        logger.info("myd021km dataframe does not exist, creating now")
        myd021km_plume_df = pd.DataFrame()
    return myd021km_plume_df


def get_timestamp(viirs_sdr_fname):
    try:
        return re.search("[d][0-9]{8}[_][t][0-9]{6}", viirs_sdr_fname).group()
    except Exception, e:
        logger.warning("Could not extract time stamp from: " + viirs_sdr_fname + " with error: " + str(e))
        return ''


def image_seen(viirs_fname):
    try:
        with open(filepaths.path_to_processed_filelist_viirs, 'r+') as txt_file:
            if viirs_fname in txt_file.read():
                logger.info(viirs_fname + " already processed")
                return True
            else:
                txt_file.write(viirs_fname + '\n')
                return False
    except Exception, e:
        logger.warning("Could not check time filename for : " + viirs_fname + ". With error: " + str(e))
        return False  # if we cannot do it, lets just assume we haven't seen the image before


def get_viirs_fname(path, timestamp_viirs, viirs_sdr_fname):
    fname = [f for f in os.listdir(path) if timestamp_viirs in f]
    if len(fname) > 1:
        logger.warning("More that one frp granule matched " + viirs_sdr_fname + "selecting 0th option")
        return fname[0]
    elif len(fname) == 1:
        return fname[0]
    else:
        return ''


def read_h5(f):
    return h5py.File(f,  "r")


def create_resampler(viirs_data):
    lats = viirs_data['All_Data']['VIIRS-MOD-GEO_All']['Latitude'][:]
    lons = viirs_data['All_Data']['VIIRS-MOD-GEO_All']['Longitude'][:]
    return ut.utm_resampler(lats, lons, 750)


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image[image > 0].flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def tcc_viirs(viirs_data, resampler):
    #m1_params = viirs_data['All_Data']['VIIRS-M1-SDR_All']['RadianceFactors']
    m1 = viirs_data['All_Data']['VIIRS-M1-SDR_All']['Radiance'][:]
    #m4_params = viirs_data['All_Data']['VIIRS-M1-SDR_All']['RadianceFactors']
    m4 = viirs_data['All_Data']['VIIRS-M4-SDR_All']['Radiance'][:]
    #m5_params = viirs_data['All_Data']['VIIRS-M1-SDR_All']['RadianceFactors']
    m5 = viirs_data['All_Data']['VIIRS-M5-SDR_All']['Radiance'][:]

    mask = m5<0
    masked_lats = np.ma.masked_array(resampler.lats, mask)
    masked_lons = np.ma.masked_array(resampler.lons, mask)

    resampled_m1 = resampler.resample_image(m1, masked_lats, masked_lons, fill_value=0)
    resampled_m4 = resampler.resample_image(m4, masked_lats, masked_lons, fill_value=0)
    resampled_m5 = resampler.resample_image(m5, masked_lats, masked_lons, fill_value=0)

    r = image_histogram_equalization(resampled_m5)
    g = image_histogram_equalization(resampled_m4)
    b = image_histogram_equalization(resampled_m1)

    r = np.round((r * (255 / np.max(r))) * 1).astype('uint8')
    g = np.round((g * (255 / np.max(g))) * 1).astype('uint8')
    b = np.round((b * (255 / np.max(b))) * 1).astype('uint8')

    rgb = np.dstack((r, g, b))
    return rgb


def extract_aod(viirs_aod, resampler):
    aod = viirs_aod['All_Data']['VIIRS-Aeros-Opt-Thick-IP_All']['faot550'][:]
    mask = aod < -1
    masked_lats = np.ma.masked_array(resampler.lats, mask)
    masked_lons = np.ma.masked_array(resampler.lons, mask)
    resampled_aod = resampler.resample_image(aod, masked_lats, masked_lons, fill_value=0)

    return resampled_aod




class Annotate(object):
    def __init__(self, tcc, viirs_aod, ax,
                 plume_polygons, background_polygons, plume_vectors):
        self.ax = ax
        self.tcc = tcc
        self.viirs_aod = viirs_aod
        self.im = self.ax.imshow(self.viirs_aod, interpolation='none', cmap='viridis')
        #if fires is not None:
        #    self.plot = self.ax.plot(fires[1], fires[0], 'r.')
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
        # FCC and TCC always present
        labels.append('TCC')
        labels.append('VIIRS_AOD')
        return tuple(labels)

    def _radio_label_mapping(self):
        label_mapping = {}
        label_mapping['TCC'] = self.tcc
        label_mapping['VIIRS_AOD'] = self.viirs_aod
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

        if label == "VIIRS_AOD":
            self.im.set_clim(vmax=2, vmin=0)
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


def digitise(tcc, viirs_aod, viirs_fname):

    plume_polygons = []
    background_polygons = []
    plume_vectors = []

    do_annotation = True
    while do_annotation:

        fig, ax = plt.subplots(1, figsize=(11, 8))
        plt.title(viirs_fname)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # first set up the annotator
        annotator = Annotate(tcc, viirs_aod, ax,
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
    viirs_plume_df = load_df(filepaths.path_to_smoke_plume_polygons_viirs)

    for viirs_sdr_fname in os.listdir(filepaths.path_to_viirs_sdr):

        logger.info("Processing viirs file: " + viirs_sdr_fname)

        if 'DS' in viirs_sdr_fname:
            continue

        timestamp_viirs = get_timestamp(viirs_sdr_fname)
        if not timestamp_viirs:
            continue

        if image_seen(viirs_sdr_fname):
            continue

        try:
            aod_fname = get_viirs_fname(filepaths.path_to_viirs_aod, timestamp_viirs, viirs_sdr_fname)
        except Exception, e:
            logger.warning('Could not load aux file for:' + viirs_sdr_fname + '. Failed with ' + str(e))
            continue

        try:
            viirs_sdr = read_h5(os.path.join(filepaths.path_to_viirs_sdr, viirs_sdr_fname))

            # setup resampler adn extract true colour
            utm_resampler = create_resampler(viirs_sdr)
            tcc = tcc_viirs(viirs_sdr, utm_resampler)

        except Exception, e:
            logger.warning('Could not read the input file: ' + viirs_sdr_fname + '. Failed with ' + str(e))
            continue

        # if filenames load in data
        viirs_aod = None
        if aod_fname:
            try:
                viirs_aod_data = read_h5(os.path.join(filepaths.path_to_viirs_aod, aod_fname))
                viirs_aod = extract_aod(viirs_aod_data, utm_resampler)
            except Exception, e:
                logger.warning('Could not read aod file: ' + aod_fname)
        if viirs_aod is None:
            continue


        # do the digitising
        plume_polygons, background_polygons, plume_vectors = digitise(tcc,
                                                                      viirs_aod,
                                                                      viirs_sdr_fname)
        if plume_polygons is None:
            continue

        # process plumes and backgrounds
        plumes_list = []
        for pp, bp, pv in zip(plume_polygons, background_polygons, plume_vectors):
            append_to_list(pp, bp, pv, viirs_sdr_fname, plumes_list)

        # covert pixel/background lists to dataframes and concatenate to main dataframes
        temp_plume_df = pd.DataFrame(plumes_list)
        viirs_plume_df = pd.concat([viirs_plume_df, temp_plume_df])
        viirs_plume_df.to_pickle(filepaths.path_to_smoke_plume_polygons_modis)


if __name__ == '__main__':

    #plt.ioff()
    matplotlib.pyplot.close("all")

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    main()
