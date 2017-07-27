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


def get_primary_times(primary_files):
    times = [datetime.strptime(pf.split('_')[-2], '%Y%m%d%H%M').timetuple() for pf in primary_files]
    unique_times = list(set(times))

    primary_datestrings = []
    for tt in unique_times:
        primary_datestrings.append(str(tt.tm_year) + \
                                  str(tt.tm_yday).zfill(3) + \
                                  '.' + \
                                  str(tt.tm_hour).zfill(2) + \
                                  str(tt.tm_min).zfill(2))
    return unique_times, primary_datestrings


def get_sub_df(primary_time, mask_df):
    return mask_df[mask_df['filename'].str.contains(primary_time)]


def image_histogram_equalization(image, number_bins=512):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def tcc_myd021km(mod_file):
    mod_data = SD(mod_file, SDC.READ)
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


def make_plume_plot(tcc, primary_data, pp, plume_mask, axis):


    for ax, k in zip(axis.flatten(), ['', 'cer', 'cot', 'costjm']):

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # get the mask
        mask = plume_mask[pp[0]:pp[1], pp[2]:pp[3]].astype('bool')

        if not k:
            p = ax.imshow(tcc[pp[0]:pp[1], pp[2]:pp[3]], interpolation='None')
            cbar = plt.colorbar(p, ax=ax)
            cbar.ax.get_yaxis().labelpad = 15
            cbar.ax.set_ylabel('dn', rotation=270, fontsize=14)

        else:
            data = primary_data.variables[k][pp[0]:pp[1], pp[2]:pp[3]]
            masked_data = np.ma.masked_array(data, mask=~mask)
            if k == 'costjm':
                p = ax.imshow(masked_data, interpolation='none', norm=LogNorm(vmin=1, vmax=1000))
            else:
                p = ax.imshow(masked_data, interpolation='none', norm=LogNorm(vmin=0.01, vmax=1))
            cbar = plt.colorbar(p, ax=ax)
            cbar.ax.get_yaxis().labelpad = 15
            cbar.ax.set_ylabel(k, rotation=270, fontsize=14)



def main():

    # set up paths
    l1b_path = '/group_workspaces/cems2/nceo_generic/satellite_data/modis_c6/myd021km/2014/'
    orac_data_path = '/home/users/dnfisher/nceo_aerosolfire/data/orac_proc/myd/2014/'
    mask_path = '/home/users/dnfisher/nceo_aerosolfire/data/plume_masks/myd021km_plumes_df.pickle'
    output = '/home/users/dnfisher/nceo_aerosolfire/data/quicklooks/plume_retrievals/'
    output_txt = '/home/users/dnfisher/nceo_aerosolfire/data/plume_masks/'
    lut_classes = ['WAT', 'AMZ', 'BOR', 'CER', 'AMW', 'BOW', 'CEW']

    # read in the masks
    mask_df = pd.read_pickle(mask_path)

    # get the list of file times
    orac_times, l1b_times = get_primary_times(glob.glob(orac_data_path + '*/*/*' + '.primary*'))


    with open(output_txt + "plume_extents.txt", "w") as text_file:

        for orac_time, l1b_time in zip(orac_times, l1b_times):

            # get the tcc of the associated l1b file
            l1b_file = glob.glob(l1b_path + '/*/*' + l1b_time + '*')[0]
            tcc = tcc_myd021km(l1b_file)
            fname = output + l1b_file.split('/')[-1][:-3] + '_orac_quicklooks'

            # open up the orac products
            prod_dict = {}
            for lut_class in lut_classes:
                orac_file = glob.glob(orac_data_path + '*/*/*' + orac_time + '*' + lut_class + '*.primary*')
                print orac_file
                orac_data = open_primary(orac_file)
                prod_dict[lut_class] = orac_data

            # get the plume mask for the scene
            plume_mask = make_mask(orac_data, l1b_time, mask_df)
            plume_positions = label_plumes(plume_mask)

            # iterate over plumes in scene
            for i, pp in enumerate(plume_positions):
                fig, axes = plt.subplots(len(lut_classes), 4, figsize=(20, 5*len(lut_classes)))

                # iterate over the classes
                for lut_class in lut_classes:

                    # visualise
                    make_plume_plot(tcc, prod_dict[lut_class], pp, plume_mask, axes[i])

                plt.savefig(fname + '_p' + str(i) + '.png', bbox_inches='tight')
                plt.close('all')

                # dump plume pos for caroline
                text_file.write(orac_file.split('/')[-1] + " " +
                                str(pp[0]) + " " +
                                str(pp[1]) + " " +
                                str(pp[2]) + " " +
                                str(pp[3]) + "\n")

if __name__ == '__main__':
    main()
