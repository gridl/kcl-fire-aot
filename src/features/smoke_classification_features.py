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



import resampling
import src.config.filepaths as filepaths
import src.config.features as features_settings
import src.data.readers as readers


__author__ = "Daniel Fisher"
__email__ = "daniel.fisher@kcl.ac.uk"


def read_myd021km(local_filename):
    return SD(local_filename, SDC.READ)


def get_reduced_mask(data, mask):
    '''
    Return all DNs greater than percentile
    '''
    pc = np.percentile(data[mask], features_settings.percentile)
    return data > pc


def fill_dict(fill_dict, modis_fname, roi, mask, smoke_flag=1):

    path_to_data = os.path.join(filepaths.path_to_modis_l1b, modis_fname)
    modis_data = read_myd021km(path_to_data)

    for chan_band_name, chan_data_name in zip(['Band_1KM_RefSB', 'Band_1KM_Emissive'],
                                              ['EV_1KM_RefSB', 'EV_1KM_Emissive']):
        mod_chan_band = modis_data.select(chan_band_name).get()
        mod_chan_data = modis_data.select(chan_data_name).get()
        for i, band in enumerate(mod_chan_band):

            masked_channel = mod_chan_data[i, roi['min_y']:roi['max_y'], roi['min_x']:roi['max_x']][mask]

            if band in fill_dict:
                fill_dict[band].extend(list(masked_channel))
            else:
                fill_dict[band] = list(masked_channel)

    n_pixels = np.sum(mask)
    if 'fname' in fill_dict:
        fill_dict['fname'].extend([path_to_data.split('/')[-1]] * n_pixels)
        fill_dict['smoke_flag'].extend([smoke_flag] * n_pixels)
    else:
        fill_dict['fname'] = [path_to_data.split('/')[-1]] * n_pixels
        fill_dict['smoke_flag'] = [smoke_flag] *n_pixels


def extract_plume_data(plume_masks, features_df):

    # we fill this up withe the data and then place it in the dataframe.  Better way?
    holding_dict = {}

    current_modis_filename = ''
    for index, plume in plume_masks.iterrows():
        if plume.filename != current_modis_filename:
            current_modis_filename = plume.filename

        # get bounding box for plume to speed processing a bit
        roi = resampling.get_roi_bounds(plume)
        mask = resampling.get_plume_mask(plume, roi)

        # get modis plume data into dataframe
        fill_dict(holding_dict, current_modis_filename, roi, mask)






def main():

    # create df to hold the outputs
    features_df = pd.DataFrame()

    # read in plume dataframe
    plume_masks = readers.read_plume_data(filepaths.path_to_smoke_plume_masks)

    # extract modis plume data
    extract_plume_data(plume_masks, features_df)

    # extract modis non-plume data

    # save the dataframe




if __name__ == '__main__':
    main()
