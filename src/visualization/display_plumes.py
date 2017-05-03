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
from pyhdf.SD import SD, SDC
from datetime import datetime


def get_primary_time(primary_file):
    tt = datetime.strptime(primary_file.split('_')[-2], '%Y%m%d%H%M').timetuple()
    primary_datestring = str(tt.tm_year) + \
                         str(tt.tm_yday).zfill(3) + \
                         '.' + \
                         str(tt.tm_hour).zfill(2) + \
                         str(tt.tm_min).zfill(2)
    return primary_datestring


def get_sub_df(primary_time, mask_df):
    return mask_df[mask_df['filename'].str.contains(primary_time)]


def read_vis(l1b_file):
    ds = SD(l1b_file, SDC.READ)
    mod_params_ref = ds.select("EV_1KM_RefSB").attributes()
    ref = ds.select("EV_1KM_RefSB").get()
    ref_chan = 0
    vis = (ref[ref_chan, :, :] - mod_params_ref['radiance_offsets'][ref_chan]) * mod_params_ref['radiance_scales'][
        ref_chan]
    vis = np.round((vis * (255 / np.max(vis))) * 1).astype('uint8')
    return vis


def open_primary(primary_file):
    return Dataset(primary_file)


def make_mask(primary_data, primary_time, mask_df):
    primary_shape = primary_data.variables['cer'].shape

    # get teh sub dataframe associated with the mask
    sub_df = get_sub_df(primary_time, mask_df)

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


def make_plot(visrad, primary_data, plume_positions):

    # iterate over the plumes
    for pp in plume_positions:

        fig, axes = plt.subplots(2, 3)

        for ax, k in zip(axes.flatten(), ['' , 'cer', 'cot', 'ctp', 'ctt', 'costjm']):
            if not k:
                ax.imshow(visrad[pp[0]:pp[1], pp[2]:pp[3]], cmap='gray', interpolation='None')
            else:
                data = primary_data.variables[k][pp[0]:pp[1], pp[2]:pp[3]]
                p = ax.imshow(data)
                #cbar = plt.colorbar(p, ax=ax)
        plt.show()


def main():


    # set up paths
    l1b_path = '../../data/raw/l1b/'
    orac_data_path = '../../data/processed/orac_test_scene/'
    mask_path = '../../data/interim/myd021km_plumes_df.pickle'
    output = ''

    # read in the masks
    mask_df = pd.read_pickle(mask_path)

    # iterate over modis files
    for primary_file in glob.glob(orac_data_path + '*/*/*primary*'):

        # first get the primary file time
        primary_time = get_primary_time(primary_file)

        # find and read the vis channel of the associated l1b file
        l1b_file = glob.glob(l1b_path + '*' + primary_time + '*')[0]
        visrad = read_vis(l1b_file)

        # open up the ORAC primary file
        primary_data = open_primary(primary_file)

        # make the smoke plume mask
        plume_mask = make_mask(primary_data, primary_time, mask_df)

        # get the individual plumes
        plume_positions = label_plumes(plume_mask)

        # visualise
        make_plot(visrad, primary_data, plume_positions)



if __name__ == '__main__':
    main()