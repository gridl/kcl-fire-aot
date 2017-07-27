'''
This function reads in the hand plume digisations for a given
MODIS file.  It also reads in the ORAC main outputs.  It then
combines the mask and the output and plots the result to allow
the user to evaluate the various parameters.
'''

import glob

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

from matplotlib.path import Path
from matplotlib.colors import LogNorm
from netCDF4 import Dataset
from pyhdf.SD import SD, SDC
from datetime import datetime

from mpl_toolkits.basemap import Basemap


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


def make_plume_plot(fname, tcc, primary_data, plume_positions, plume_mask):

    # iterate over the plumes
    for i, pp in enumerate(plume_positions):

        fig, axes = plt.subplots(1, 4, figsize=(20, 6))

        for ax, k in zip(axes.flatten(), ['', 'cer', 'cot', 'costjm']):

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # get the mask
            mask = plume_mask[pp[0]:pp[1], pp[2]:pp[3]]

            if not k:
                p = ax.imshow(tcc[pp[0]:pp[1], pp[2]:pp[3]], interpolation='None')
                cbar = plt.colorbar(p, ax=ax)
                cbar.ax.get_yaxis().labelpad = 15
                cbar.ax.set_ylabel('dn', rotation=270, fontsize=14)

            else:
                data = primary_data.variables[k][pp[0]:pp[1], pp[2]:pp[3]]
                masked_data = np.ma.masked_array(data, mask=~mask)
                p = ax.imshow(masked_data, interpolation='none', norm=LogNorm(vmin=0.01, vmax=1))
                cbar = plt.colorbar(p, ax=ax)
                cbar.ax.get_yaxis().labelpad = 15
                cbar.ax.set_ylabel(k, rotation=270, fontsize=14)
        plt.savefig(fname + '_p' + str(i) + '.png', bbox_inches='tight')
    plt.close()


def main():

    # set up paths
    l1b_path = '/group_workspaces/cems2/nceo_generic/satellite_data/modis_c6/myd021km/2014/'
    orac_data_path = '/home/users/dnfisher/nceo_aerosolfire/data/orac_proc/myd/2014/'
    mask_path = '/home/users/dnfisher/nceo_aerosolfire/data/plume_masks/myd021km_plumes_df.pickle'
    output = '/home/users/dnfisher/nceo_aerosolfire/data/quicklooks/plume_retrievals/'
    output_txt = '/home/users/dnfisher/nceo_aerosolfire/data/plume_masks/'
    lut_classes = ['WAT', 'AMZ', 'BOW']

    # read in the masks
    mask_df = pd.read_pickle(mask_path)

    for lut_class in lut_classes:

        with open(output_txt + lut_class + "_plume_extents.txt", "w") as text_file:

            # iterate over modis files
            for primary_file in glob.glob(orac_data_path + '*/*/*' + lut_class + '.primary*'):

                # first get the primary file time
                primary_time = get_primary_time(primary_file)
                fname = output + primary_file.split('/')[-1][:-3] + '_quicklook'

                # get the tcc of the associated l1b file
                l1b_file = glob.glob(l1b_path + '/*/*' + primary_time + '*')[0]
                tcc = tcc_myd021km(l1b_file)

                # open up the ORAC primary file
                primary_data = open_primary(primary_file)

                # get the plumes
                plume_mask = make_mask(primary_data, primary_time, mask_df)
                plume_positions = label_plumes(plume_mask)

                # visualise
                make_plume_plot(fname, tcc, primary_data, plume_positions, plume_mask)

                # let dump coords of plume to text file for Caroline
                for pp in plume_positions:
                    text_file.write(primary_file.split('/')[-1] + " " +
                                    str(pp[0]) + " " +
                                    str(pp[1]) + " " +
                                    str(pp[2]) + " " +
                                    str(pp[3]) + "\n")

if __name__ == '__main__':
    main()
