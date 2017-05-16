'''
Script to generate MODIS False Colour Composites
'''

import logging
import os
import re

import numpy as np
from skimage import exposure
from pyhdf.SD import SD, SDC

import matplotlib.pyplot as plt


def read_myd14(myd14_file):
    return SD(myd14_file, SDC.READ)


def firemask_myd14(myd14_data):
    return myd14_data.select('fire mask').get() >= 7


def read_myd021km(local_filename):
    return SD(local_filename, SDC.READ)


def fcc_myd021km_250(mod_data, fire_mask):
    mod_params_500 = mod_data.select("EV_500_Aggr1km_RefSB").attributes()
    ref_500 = mod_data.select("EV_500_Aggr1km_RefSB").get()

    mod_params_250 = mod_data.select("EV_250_Aggr1km_RefSB").attributes()
    ref_250 = mod_data.select("EV_250_Aggr1km_RefSB").get()

    r = (ref_500[4, :, :] - mod_params_500['radiance_offsets'][4]) * mod_params_500['radiance_scales'][
        4]  # 2.1 microns
    g = (ref_250[1, :, :] - mod_params_250['radiance_offsets'][1]) * mod_params_250['radiance_scales'][
        1]  # 0.8 microns
    b = (ref_250[0, :, :] - mod_params_250['radiance_offsets'][0]) * mod_params_250['radiance_scales'][
        0]  # 0.6 microns

    r = np.round((r * (255 / np.max(r))) * 1).astype('uint8')
    g = np.round((g * (255 / np.max(g))) * 1).astype('uint8')
    b = np.round((b * (255 / np.max(b))) * 1).astype('uint8')
    #
    # mini = 5
    # maxi = 95
    #
    # r_min, r_max = np.percentile(r, (mini, maxi))
    # r = exposure.rescale_intensity(r, in_range=(r_min, r_max))
    r[fire_mask] = 255
    #
    # g_min, g_max = np.percentile(g, (mini, maxi))
    # g = exposure.rescale_intensity(g, in_range=(g_min, g_max))
    g[fire_mask] = 0
    #
    # b_min, b_max = np.percentile(b, (mini, maxi))
    # b = exposure.rescale_intensity(b, in_range=(b_min, b_max))
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


def main():

    # set figsize
    plt.figure(figsize=(15,18))

    # output dir
    output_dir = '/Users/dnf/git/kcl-fire-aot/data/processed/modis_scenes/'

    # get the files to download
    file_list_path = '/Users/dnf/git/kcl-fire-aot/data/raw/rsync_file_list/files_to_transfer.txt'
    with open(file_list_path, 'r') as f:
        files_list = f.read()

    for myd021km_fname in os.listdir(r"../../data/raw/l1b"):

        if not myd021km_fname in files_list:
            continue

        logger.info("Processing modis granule: " + myd021km_fname)

        try:
            timestamp_myd = re.search("[0-9]{7}[.][0-9]{4}[.]", myd021km_fname).group()
        except Exception, e:
            logger.warning("Could not extract time stamp from: ", myd021km_fname, "moving on to next file")
            continue

        myd14_fname = [f for f in os.listdir(r"../../data/raw/frp") if timestamp_myd in f]

        if len(myd14_fname) > 1:
            logger.warning("More that one frp granule matched " + myd021km_fname + "selecting 0th option")
        myd14_fname = myd14_fname[0]

        myd14 = read_myd14(os.path.join(r"../../data/raw/frp/", myd14_fname))
        myd021km = read_myd021km(os.path.join(r"../../data/raw/l1b", myd021km_fname))

        # do the digitising
        myd14_fire_mask = firemask_myd14(myd14)
        img = fcc_myd021km(myd021km, myd14_fire_mask)

        plt.imshow(img, interpolation='none')
        plt.savefig(output_dir + myd021km_fname + '_fcc.png', bbox_inches='tight')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
