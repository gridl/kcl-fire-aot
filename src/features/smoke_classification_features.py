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

import src.config.filepaths as filepaths
import src.config.features as features_settings
import src.data.readers as readers

__author__ = "Daniel Fisher"
__email__ = "daniel.fisher@kcl.ac.uk"


def read_myd021km(local_filename):
    return SD(local_filename, SDC.READ)


def generate_textures(mod_chan_data, plume, i):
    r = features_settings.glcm_window_radius
    min_y = plume.sample_bounds[2] - r
    max_y = plume.sample_bounds[3] + r
    min_x = plume.sample_bounds[0] - r
    max_x = plume.sample_bounds[1] + r

    image = mod_chan_data[i, min_y:max_y, min_x:max_x]
    texture_generator = textures.CooccurenceMatrixTextures(image, windowRadius=r)

    measures = []
    names = ['glcm_dissimilarity', 'glcm_correlation', 'glcm_variance', 'glcm_mean']

    diss = texture_generator.getDissimlarity()
    corr, var, mean = texture_generator.getCorrVarMean()

    for measure in [diss, corr, var, mean]:
        measure = measure[r:-r, r:-r]
        measures.append(measure.flatten())

    return measures, names


def fill_dict(plume, flag, fill_dict, fp, modis_fname):
    path_to_data = os.path.join(fp, modis_fname)
    modis_data = read_myd021km(path_to_data)

    try:
        for chan_band_name, chan_data_name in zip(['Band_500M', 'Band_250M', 'Band_1KM_Emissive'],
                                                  ['EV_500_Aggr1km_RefSB', 'EV_250_Aggr1km_RefSB', 'EV_1KM_Emissive']):

            mod_chan_band = modis_data.select(chan_band_name).get()
            mod_chan_data = modis_data.select(chan_data_name).get()

            for i, band in enumerate(mod_chan_band):

                # first check if we are reducing the plume extents by percentile
                if features_settings.reduce_features & (band == 3) & flag:
                    data = mod_chan_data[i, plume.sample_bounds[2]:plume.sample_bounds[3],
                           plume.sample_bounds[0]:plume.sample_bounds[1]]
                    per = np.percentile(data, features_settings.reduce_percentile)
                    mask = (data >= per).flatten()

                # now lets generate GLCM texture measures for MODIS band 3
                if band == 3:

                    texture_measure, keys = generate_textures(mod_chan_data, plume, i)

                    for i, k in enumerate(keys):
                        if k in fill_dict:
                            if features_settings.reduce_features & flag:
                                fill_dict[k].extend(list(texture_measure[i][mask]))
                            else:
                                fill_dict[k].extend(list(texture_measure[i]))
                        else:
                            if features_settings.reduce_features & flag:
                                fill_dict[k] = list(texture_measure[i][mask])
                            else:
                                fill_dict[k] = list(texture_measure[i])

                data_for_band = mod_chan_data[i, plume.sample_bounds[2]:plume.sample_bounds[3],
                                plume.sample_bounds[0]:plume.sample_bounds[1]]
                data_for_band = data_for_band.flatten()

                if band in fill_dict:
                    if features_settings.reduce_features & flag:
                        fill_dict[band].extend(list(data_for_band[mask]))
                    else:
                        fill_dict[band].extend(list(data_for_band))
                else:
                    if features_settings.reduce_features & flag:
                        fill_dict[band] = list(data_for_band[mask])
                    else:
                        fill_dict[band] = list(data_for_band)

        if features_settings.reduce_features & flag:
            n_pixels = np.sum(mask)
        else:
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
        try:
            if plume.filename != current_modis_filename:
                current_modis_filename = plume.filename

            # get modis plume data into holding dict
            fill_dict(plume, flag, holding_dict, filepaths.path_to_modis_l1b, current_modis_filename)
        except Exception, e:
            logger.warning("Failed to process plumes in modis granule: " + current_modis_filename +
                           ' with error: ' + str(e))


def main():
    # create dict to hold the outputs
    holding_dict = dict()

    for path, flag in zip([filepaths.path_to_ml_smoke_plume_masks, filepaths.path_to_ml_smoke_free_masks], [1, 0]):
        # read in plume dataframe
        plume_masks = readers.read_plume_data(path)

        # extract modis plume data
        extract_samples(plume_masks, holding_dict, flag)

    # save the dataframe
    df = pd.DataFrame.from_dict(holding_dict)
    if features_settings.reduce_features:
        df.to_pickle(filepaths.path_to_reduced_plume_classification_features)
    else:
        df.to_pickle(filepaths.path_to_plume_classification_features)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()
