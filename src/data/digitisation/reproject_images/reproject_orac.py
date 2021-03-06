import logging
import os

import numpy as np
import scipy.misc as misc
from netCDF4 import Dataset

import src.config.filepaths_cems as fp
import src.features.fre_to_tpm.modis.ftt_utils as ut


def read_nc(f):
    return Dataset(f)


def create_resampler(orac_data):
    lats = orac_data.variables['lat'][:]
    lons = orac_data.variables['lon'][:]
    return ut.utm_resampler(lats, lons, 750)


def extract_aod(orac_data, resampler):
    aod = orac_data['cot'][:]
    mask = np.ma.getmask(aod)
    masked_lats = np.ma.masked_array(resampler.lats, mask)
    masked_lons = np.ma.masked_array(resampler.lons, mask)
    resampled_aod = resampler.resample_image(aod, masked_lats, masked_lons, fill_value=0)
    return resampled_aod


def extract_cost(orac_data, resampler):
    cost = orac_data['costjm'][:]
    mask = np.ma.getmask(cost)
    masked_lats = np.ma.masked_array(resampler.lats, mask)
    masked_lons = np.ma.masked_array(resampler.lons, mask)
    resampled_cost = resampler.resample_image(cost, masked_lats, masked_lons, fill_value=1000)
    return resampled_cost


def main():

    for viirs_orac_fname in os.listdir(fp.path_to_viirs_orac):

        if 'DS' in viirs_orac_fname:
            continue
        if 'primary' not in viirs_orac_fname:
            continue

        if os.path.isfile(os.path.join(
                fp.path_to_viirs_orac_resampled, viirs_orac_fname.replace('nc', 'png'))):
            print viirs_orac_fname, 'already resampled'
            continue

        logger.info("Processing viirs file: " + viirs_orac_fname)

        try:
            viirs_orac = read_nc(os.path.join(fp.path_to_viirs_orac, viirs_orac_fname))

            # setup resampler adn extract true colour
            utm_resampler = create_resampler(viirs_orac)
            aod = extract_aod(viirs_orac, utm_resampler)
            cost = extract_cost(viirs_orac, utm_resampler)

        except Exception, e:
            logger.warning('Could not read the input file. Failed with ' + str(e))
            continue

        # save the outputs
        misc.imsave(os.path.join(fp.path_to_viirs_orac_resampled, viirs_orac_fname.replace('nc', 'png')), aod)
        misc.imsave(os.path.join(fp.path_to_viirs_orac_cost_resampled, viirs_orac_fname.replace('nc', 'png')), cost)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
