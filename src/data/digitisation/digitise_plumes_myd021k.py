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
        with open(filepaths.path_to_processed_filelist, 'r+') as txt_file:
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


def get_viirs_fname(path, timestamp_myd):
    # only one dataset per day for this analysis
    # so do not need to worry about geolocating the data
    # however put in warning for future to catch errors
    t = timestamp_myd[0:7]
    fname = [f for f in os.listdir(path) if t in f[0:28]]
    if len(fname) > 1:
        logger.warning('Multiple VIIRS AOD files found.  Need to add geolocation check to find right one')
        print fname
        return ''
    elif len(fname) == 1:
        return fname[0]
    else:
        return ''


def read_hdf(f):
    return SD(f, SDC.READ)


def read_orac(f):
    return Dataset(f)


def fires_myd14(myd14_data):
    return np.where(myd14_data.select('fire mask').get() >= 7)


def aod_myd04_3K(myd04_3K):
    aod_params = myd04_3K.select("Optical_Depth_Land_And_Ocean").attributes()
    aod = myd04_3K.select("Optical_Depth_Land_And_Ocean").get()
    aod = (aod + aod_params['add_offset']) * aod_params['scale_factor']
    aod = ndimage.zoom(aod, 3, order=1)
    return aod


def aod_orac(orac_ds):
    return orac_ds.variables['cot'], orac_ds.variables['costjm']


def aod_viirs(viirs_aod_data, viirs_geo_data, myd021km):

    '''
    Resample the VIIRS AOD data to the MODIS grid
    '''

    mod_lats = myd021km.select('Latitude').get()
    mod_lons = myd021km.select('Longitude').get()
    mod_swath_def = pr.geometry.SwathDefinition(lons=mod_lons, lats=mod_lats)

    viirs_aod = viirs_aod_data.select('faot550').get()
    viirs_mask = viirs_aod < 0

    viirs_lats = viirs_geo_data.select('Latitude').get()
    viirs_lons = viirs_geo_data.select('Longitude').get()
    viirs_masked_lats = np.ma.masked_array(viirs_lats, viirs_mask)
    viirs_masked_lons = np.ma.masked_array(viirs_lons, viirs_mask)
    viirs_swath_def = pr.geometry.SwathDefinition(lons=viirs_masked_lons, lats=viirs_masked_lats)

    resampled_viirs_aod = pr.kd_tree.resample_nearest(viirs_swath_def,
                                                      viirs_aod,
                                                      mod_swath_def,
                                                      radius_of_influence=1000,
                                                      epsilon=0.5,
                                                      fill_value=0)
    return resampled_viirs_aod


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
    def __init__(self, fcc, tcc, mod_aod, viirs_aod, orac_aod, orac_cost, fires, ax, polygons):
        self.ax = ax
        self.fcc = fcc
        self.tcc = tcc
        self.mod_aod = mod_aod
        self.viirs_aod = viirs_aod
        self.orac_aod = orac_aod
        self.orac_cost = orac_cost
        self.im = self.ax.imshow(self.aod, interpolation='none')
        if fires is not None:
            self.plot = self.ax.plot(fires[1], fires[0], 'r.')
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
        self.radio_image = RadioButtons(self.rax_image, self._radio_labels())
        self.radio_image.on_clicked(self.image_func)

    def _radio_labels(self):

        labels = []
        if self.mod_aod is not None:
            labels.append('MOD_AOD')
        if self.viirs_aod is not None:
            labels.append('VIIRS_AOD')
        if self.orac_aod is not None:
            labels.append('ORAC_AOD')
            labels.append('ORAC_COST')

        # FCC and TCC always present
        labels.append('FCC')
        labels.append('TCC')

        return tuple(labels)

    def _radio_label_mapping(self):

        label_mapping = {}
        if self.mod_aod is not None:
            label_mapping['MOD_AOD'] = self.mod_aod
        if self.viirs_aod is not None:
            label_mapping['VIIRS_AOD'] = self.viirs_aod
        if self.orac_aod is not None:
            label_mapping['ORAC_AOD'] = self.orac_aod
            label_mapping['ORAC_COST'] = self.orac_cost

        # FCC and TCC always present
        label_mapping['FCC'] = self.fcc
        label_mapping['TCC'] = self.tcc
        return label_mapping

    def annotation_func(self, label):
        anno_dict = {'Digitise': True, 'Stop': False}
        self.do_annotation = anno_dict[label]

    def discard_func(self, label):
        keep_dict = {'Keep': False, 'Discard': True}
        if keep_dict[label]:
            self.x = []
            self.y = []

    def image_func(self, label):
        image_dict = self._radio_label_mapping()
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


def digitise(fcc, tcc, mod_aod, viirs_aod, orac_aod, orac_cost, fires):

    smoke_polygons = []

    do_annotation = True
    while do_annotation:

        fig, ax = plt.subplots(1, figsize=(12,20))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # first set up the annotator
        annotator = Annotate(fcc, tcc, mod_aod, viirs_aod, orac_aod, orac_cost, fires, ax, smoke_polygons)

        # then show the image
        plt.show()

        # get the polygon points from the closed image
        pts = zip(annotator.x, annotator.y)
        if pts:
            smoke_polygons.append(pts)

        do_annotation = annotator.do_annotation

    plt.close(fig)

    return smoke_polygons


def extract_plume_bounds(plume, fname, plumes_list):
    row_dict = {}

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

        if image_seen(myd021km_fname):
            continue

        myd14_fname = get_modis_fname(filepaths.path_to_modis_frp, timestamp_myd, myd021km_fname)
        myd04_3K_fname = get_modis_fname(filepaths.path_to_modis_aod_3k, timestamp_myd, myd021km_fname)
        orac_fname = get_orac_fname(filepaths.path_to_orac_aod, timestamp_myd)
        viirs_aod_fname = get_viirs_fname(filepaths.path_to_viirs_aod, timestamp_myd)
        viirs_geo_fname = get_viirs_fname(filepaths.path_to_viirs_geo, timestamp_myd)

        try:
            myd021km = read_hdf(os.path.join(filepaths.path_to_modis_l1b, myd021km_fname))
            fcc = fcc_myd021km(myd021km)
            tcc = tcc_myd021km(myd021km)
        except Exception, e:
            logger.warning('Could not read the input file: ' + myd021km_fname + '. Failed with ' + str(e))
            continue

        # if filenames load in data
        fires, mod_aod, orac_aod, orac_cost, viirs_aod = None, None, None, None, None
        if myd14_fname:
            myd14 = read_hdf(os.path.join(filepaths.path_to_modis_frp, myd14_fname))
            fires = fires_myd14(myd14)
        if myd04_3K_fname:
            myd04_3K = read_hdf(os.path.join(filepaths.path_to_modis_aod_3k, myd04_3K_fname))
            mod_aod = aod_myd04_3K(myd04_3K)
        if orac_fname:
            orac_data = read_orac(os.path.join(filepaths.path_to_orac_aod, orac_fname))
            orac_aod, orac_cost = aod_orac(orac_data)
        if viirs_aod_fname and viirs_geo_fname:
            viirs_aod_data = read_hdf(os.path.join(filepaths.path_to_viirs_aod, viirs_aod_fname))
            viirs_geo_data = read_hdf(os.path.join(filepaths.path_to_viirs_geo, viirs_geo_fname))
            viirs_aod = aod_viirs(viirs_aod_data, viirs_geo_data, myd021km)

        # do the digitising
        smoke_polygons = digitise(fcc, tcc, mod_aod, viirs_aod, orac_aod, orac_cost, fires)
        if smoke_polygons is None:
            continue

        # process plumes and backgrounds
        plumes_list = []
        for poly in smoke_polygons:
            extract_plume_bounds(poly, myd021km_fname, plumes_list)

        # covert pixel/background lists to dataframes and concatenate to main dataframes
        temp_plume_df = pd.DataFrame(plumes_list)
        myd021km_plume_df = pd.concat([myd021km_plume_df, temp_plume_df])
        myd021km_plume_df.to_pickle(filepaths.path_to_smoke_plume_masks)


if __name__ == '__main__':

    #plt.ioff()
    matplotlib.pyplot.close("all")

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    main()
