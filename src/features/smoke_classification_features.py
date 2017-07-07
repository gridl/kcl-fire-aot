#!/usr/bin/env python

'''
Generate smoke classification features for MODIS data

- Extract smoke plume mask for each digitised plume
- For each mask apply some threshold to blue channels to remove low signal smoke pixels
- Extract DNs for all MODIS channels for each reduced plume mask
- Insert DNs into a pandas dataframe along with a flag indicating smoke, pixel coords, and filename
- Do the same for a number of non-smoke scenes inserting all image pixels
'''

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC
import logging

import matplotlib.pyplot as plt

import resampling
import src.config.filepaths as filepaths
import src.config.features as features_settings
import src.data.readers as readers


__author__ = "Daniel Fisher"
__email__ = "daniel.fisher@kcl.ac.uk"


def read_myd021km(local_filename):
    return SD(local_filename, SDC.READ)



def fill_dict(plume, flag, fill_dict, fp, modis_fname):

    path_to_data = os.path.join(fp, modis_fname)
    modis_data = read_myd021km(path_to_data)

    try:
        for chan_band_name, chan_data_name in zip(['Band_1KM_RefSB', 'Band_1KM_Emissive'],
                                                  ['EV_1KM_RefSB', 'EV_1KM_Emissive']):

            mod_chan_band = modis_data.select(chan_band_name).get()
            mod_chan_data = modis_data.select(chan_data_name).get()
            for i, band in enumerate(mod_chan_band):

                # check to see if we are working with a plume subset or an entire image
                data_for_band = mod_chan_data[i, plume.sample_bounds[2]:plume.sample_bounds[3],
                                plume.sample_bounds[0]:plume.sample_bounds[1]]
                data_for_band = data_for_band.flatten()

                if band in fill_dict:
                    fill_dict[band].extend(list(data_for_band))
                else:
                    fill_dict[band] = list(data_for_band)

        n_pixels = data_for_band.size
        if 'fname' in fill_dict:
            fill_dict['fname'].extend([path_to_data.split('/')[-1]] * n_pixels)
            fill_dict['smoke_flag'].extend([flag] * n_pixels)
        else:
            fill_dict['fname'] = [path_to_data.split('/')[-1]] * n_pixels
            fill_dict['smoke_flag'] = [flag] * n_pixels

    except Exception, e:
        logger.warning("Failed to process modis granule: " + modis_fname + ' with error: ' + str(e))



def extract_samples(plume_masks, holding_dict, flag):

    current_modis_filename = ''
    for index, plume in plume_masks.iterrows():
        if plume.filename != current_modis_filename:
            current_modis_filename = plume.filename

        # get modis plume data into holding dict
        fill_dict(plume, flag, holding_dict, filepaths.path_to_modis_l1b, current_modis_filename)


def main():

    # create dict to hold the outputs
    holding_dict = dict()

    for path, flag in zip([filepaths.path_to_ml_smoke_plume_masks, filepaths.path_to_ml_smoke_free_masks], [1,0]):

        # read in plume dataframe
        plume_masks = readers.read_plume_data(path)

        # extract modis plume data
        extract_samples(plume_masks, holding_dict, flag)

    # save the dataframe
    df = pd.DataFrame.from_dict(holding_dict)
    df.to_pickle(filepaths.path_to_plume_classification_features)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()
