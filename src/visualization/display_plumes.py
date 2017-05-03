'''
This function reads in the hand plume digisations for a given
MODIS file.  It also reads in the ORAC main outputs.  It then
combines the mask and the output and plots the result to allow
the user to evaluate the various parameters.
'''

import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

from matplotlib.path import Path
from netCDF4 import Dataset
from datetime import datetime

def get_sub_df(primary_file, mask_df):
    tt = datetime.strptime(primary_file.split('_')[-2], '%Y%m%d%H%M').timetuple()
    primary_datestring = str(tt.tm_year) + \
                         str(tt.tm_yday).zfill(3) + \
                         '.' + \
                         str(tt.tm_hour).zfill(2) + \
                         str(tt.tm_min).zfill(2)
    return mask_df[mask_df['filename'].str.contains(primary_datestring)]


def open_primary(primary_file):
    return Dataset(primary_file)


def make_mask(primary_data, primary_file, mask_df):
    primary_shape = primary_data.variables['cer'].shape

    # get teh sub dataframe associated with the mask
    sub_df = get_sub_df(primary_file, mask_df)

    # create grid to hold the plume mask
    nx = primary_shape[1]
    ny = primary_shape[0]
    mask = np.zeros((ny, nx))

    # generate the mask for each of the plumes
    for i, plume in sub_df.iterrows():
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        poly_verts = plume['plume_extent']

        # apply mask
        path = Path(poly_verts)
        grid = path.contains_points(points)
        grid = grid.reshape((ny, nx))

        mask += grid
    return mask


def label_plumes(mask):
    plume_positions = []
    labelled_mask, n_plumes = ndimage.label(mask)
    plumes = ndimage.find_objects(labelled_mask)
    for pl in plumes:
        plume_positions.append([int(pl[0].start)-5, int(pl[0].stop)+5, int(pl[1].start)-5, int(pl[1].stop)+5])
    return plume_positions


def make_plot(primary_data, plume_mask):
    pass


def main():


    # set up paths
    orac_data_path = '../../data/processed/orac_test_scene/'
    mask_path = '../../data/interim/myd021km_plumes_df.pickle'
    output = ''

    # read in the masks
    mask_df = pd.read_pickle(mask_path)

    # iterate over modis files
    for primary_file in glob.glob(orac_data_path + '*/*/*primary*'):

        # open up the ORAC primary file
        primary_data = open_primary(primary_file)

        # make the smoke plume mask
        plume_mask = make_mask(primary_data, primary_file, mask_df)

        # get the individual plumes
        plume_positions = label_plumes(plume_mask)

        # visualise
        make_plot(primary_data, plume_positions)



if __name__ == '__main__':
    main()