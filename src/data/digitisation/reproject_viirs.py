import logging
import os
import re

import h5py
import numpy as np
import scipy.misc as misc
from datetime import datetime

import src.config.filepaths as fp
import src.features.fre_to_tpm.modis.ftt_utils as ut
import src.features.fre_to_tpm.viirs.ftt_fre as ff


def get_timestamp(viirs_sdr_fname):
    try:
        return re.search("[d][0-9]{8}[_][t][0-9]{6}", viirs_sdr_fname).group()
    except Exception, e:
        logger.warning("Could not extract time stamp from: " + viirs_sdr_fname + " with error: " + str(e))
        return ''


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


def tcc_viirs(viirs_data, fires, resampler):
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

    # for the fire coordinates in the image
    resampled_lats = resampler.resample_image(resampler.lats, masked_lats, masked_lons, fill_value=1000)
    resampled_lons = resampler.resample_image(resampler.lons, masked_lats, masked_lons, fill_value=1000)
    coords_set = set(zip(resampled_lats.flatten(),resampled_lons.flatten()))

    r = image_histogram_equalization(resampled_m5)
    g = image_histogram_equalization(resampled_m4)
    b = image_histogram_equalization(resampled_m1)

    r = np.round((r * (255 / np.max(r))) * 1).astype('uint8')
    g = np.round((g * (255 / np.max(g))) * 1).astype('uint8')
    b = np.round((b * (255 / np.max(b))) * 1).astype('uint8')

    for f in fires:
        f_lon, f_lat = f.xy
        # first chekc if the points are inbounds
        if not (f_lat[0], f_lon[0]) in coords_set:
            continue

        coord_abs_diffs = np.abs(resampled_lats-f_lat) + np.abs(resampled_lons - f_lon)
        x, y = divmod(coord_abs_diffs.argmin(), coord_abs_diffs.shape[1])
        r[x, y] = 255
        g[x, y] = 0
        b[x, y] = 0

    rgb = np.dstack((r, g, b))
    return rgb


def extract_aod(viirs_aod, resampler):
    aod = viirs_aod['All_Data']['VIIRS-Aeros-Opt-Thick-IP_All']['faot550'][:]
    mask = aod < -1
    masked_lats = np.ma.masked_array(resampler.lats, mask)
    masked_lons = np.ma.masked_array(resampler.lons, mask)
    resampled_aod = resampler.resample_image(aod, masked_lats, masked_lons, fill_value=0)

    return resampled_aod


def get_mask(arr, bit_pos, bit_len, value):
    '''Generates mask with given bit information.
    Parameters
        bit_pos		-	Position of the specific QA bits in the value string.
        bit_len		-	Length of the specific QA bits.
        value  		-	A value indicating the desired condition.
    '''
    bitlen = int('1' * bit_len, 2)

    if type(value) == str:
        value = int(value, 2)

    pos_value = bitlen << bit_pos
    con_value = value << bit_pos
    mask = (arr & pos_value) == con_value
    return mask


def extract_aod_flags(viirs_aod, resampler):
    aod = viirs_aod['All_Data']['VIIRS-Aeros-Opt-Thick-IP_All']['faot550'][:]
    aod_quality = viirs_aod['All_Data']['VIIRS-Aeros-Opt-Thick-IP_All']['QF1'][:]
    flags = np.zeros(aod_quality.shape)
    for k, v in zip(['00', '01', '10', '11'], [0,1,2,3]):
        mask = get_mask(aod_quality, 0, 2, k)
        flags[mask] = v

    mask = aod < -1
    masked_lats = np.ma.masked_array(resampler.lats, mask)
    masked_lons = np.ma.masked_array(resampler.lons, mask)
    resampled_flags = resampler.resample_image(flags, masked_lats, masked_lons, fill_value=3)

    return resampled_flags


def main():

    # load in himawari fires for visualisation
    frp_df = ut.read_frp_df(fp.path_to_himawari_frp)

    for viirs_sdr_fname in os.listdir(fp.path_to_viirs_sdr):

        if os.path.isfile(os.path.join(
                fp.path_to_viirs_sdr_resampled, viirs_sdr_fname.replace('h5', 'png'))):
            print viirs_sdr_fname, 'already resampled'
            continue

        logger.info("Processing viirs file: " + viirs_sdr_fname)

        if 'DS' in viirs_sdr_fname:
            continue

        timestamp_viirs = get_timestamp(viirs_sdr_fname)
        if not timestamp_viirs:
            continue

        try:
            viirs_sdr = read_h5(os.path.join(fp.path_to_viirs_sdr, viirs_sdr_fname))

            # setup resampler adn extract true colour
            utm_resampler = create_resampler(viirs_sdr)
            t = datetime.strptime(timestamp_viirs, 'd%Y%m%d_t%H%M%S')

            fires = ff.fire_locations_for_digitisation(frp_df, t)
            tcc = tcc_viirs(viirs_sdr, fires, utm_resampler)

        except Exception, e:
            logger.warning('Could not read the input file: ' + viirs_sdr_fname + '. Failed with ' + str(e))
            continue

        # get aod filename
        try:
            aod_fname = get_viirs_fname(fp.path_to_viirs_aod, timestamp_viirs, viirs_sdr_fname)
        except Exception, e:
            logger.warning('Could not load aux file for:' + viirs_sdr_fname + '. Failed with ' + str(e))
            continue

        if not aod_fname:
            continue

        # load in viirs aod
        viirs_aod = None
        if aod_fname:
            try:
                viirs_aod_data = read_h5(os.path.join(fp.path_to_viirs_aod, aod_fname))
                viirs_aod = extract_aod(viirs_aod_data, utm_resampler)
                aod_flags = extract_aod_flags(viirs_aod_data, utm_resampler)

            except Exception, e:
                logger.warning('Could not read aod file: ' + aod_fname)
        if viirs_aod is None:
            continue

        # save the outputs
        misc.imsave(os.path.join(fp.path_to_viirs_sdr_resampled, viirs_sdr_fname.replace('h5', 'png')), tcc)
        misc.imsave(os.path.join(fp.path_to_viirs_aod_resampled, aod_fname.replace('h5', 'png')), viirs_aod)
        misc.imsave(os.path.join(fp.path_to_viirs_aod_flags_resampled, aod_fname.replace('h5', 'png')), aod_flags)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()