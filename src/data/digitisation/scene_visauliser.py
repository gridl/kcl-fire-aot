import os
import logging
import re

import numpy as np
import matplotlib
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt

import src.config.filepaths as filepaths


def get_timestamp(myd021km_fname):
    try:
        return re.search("[0-9]{7}[.][0-9]{4}[.]", myd021km_fname).group()
    except Exception, e:
        logger.warning("Could not extract time stamp from: " + myd021km_fname + " with error: " + str(e))
        return ''


def read_hdf(f):
    return SD(f, SDC.READ)


def get_modis_fname(path, timestamp_myd, myd021km_fname):
    fname = [f for f in os.listdir(path) if timestamp_myd in f]
    if len(fname) > 1:
        logger.warning("More that one frp granule matched " + myd021km_fname + "selecting 0th option")
        return fname[0]
    elif len(fname) == 1:
        return fname[0]
    else:
        return ''


def fires_myd14(myd14_data):
    return np.where(myd14_data.select('fire mask').get() >= 7)


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



def main():
    """ Loads MODIS data files for the specified geographic region,
        timeframe, and time stamps.  The user can then process these
        images and extract smoke plumes.
    """
    for myd021km_fname in os.listdir(filepaths.path_to_modis_l1b):

        if 'MYD' not in myd021km_fname:
            continue

        logger.info("Visualising modis granule: " + myd021km_fname)

        try:
            myd021km = read_hdf(os.path.join(filepaths.path_to_modis_l1b, myd021km_fname))
            timestamp_myd = get_timestamp(myd021km_fname)
            tcc = tcc_myd021km(myd021km)

            myd14_fname = get_modis_fname(filepaths.path_to_modis_frp, timestamp_myd, myd021km_fname)
            myd14 = read_hdf(os.path.join(filepaths.path_to_modis_frp, myd14_fname))
            fires = fires_myd14(myd14)

            plt.figure(figsize=(18,12))
            plt.imshow(tcc)
            plt.plot(fires[1], fires[0], 'r.')
            plt.title(myd021km_fname)
            plt.show()
        except:
            continue


if __name__ == '__main__':

    #plt.ioff()
    matplotlib.pyplot.close("all")

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    main()