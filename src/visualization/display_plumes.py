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


def make_plume_plot(fname, visrad, primary_data, plume_positions):

    # iterate over the plumes
    for i, pp in enumerate(plume_positions):

        fig, axes = plt.subplots(2, 3, figsize=(15,12))

        for ax, k in zip(axes.flatten(), ['' , 'cer', 'cot', 'ctp', 'ctt', 'costjm']):

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)


            if not k:
                p = ax.imshow(visrad[pp[0]:pp[1], pp[2]:pp[3]], cmap='gray', interpolation='None')
                cbar = plt.colorbar(p, ax=ax)
                cbar.ax.get_yaxis().labelpad = 15
                cbar.ax.set_ylabel('dn', rotation=270, fontsize=14)

            else:
                data = primary_data.variables[k][pp[0]:pp[1], pp[2]:pp[3]]
                p = ax.imshow(data, interpolation='none')
                cbar = plt.colorbar(p, ax=ax)
                cbar.ax.get_yaxis().labelpad = 15
                cbar.ax.set_ylabel(k, rotation=270, fontsize=14)
        plt.savefig(fname + '_p' + str(i) + '.png', bbox_inches='tight')
	plt.close()


def make_plume_location_plot(plume_positions, primary_data, m):

    for i, pp in enumerate(plume_positions):

        mean_lat = np.mean(primary_data.variables['lon'][pp[0]:pp[1], pp[2]:pp[3]])
        mean_lon = np.mean(primary_data.variables['lat'][pp[0]:pp[1], pp[2]:pp[3]])

        m.plot('o', mean_lon, mean_lat, latlon=True)

    plt.show()




def main():

    # set up paths
    l1b_path = '/group_workspaces/cems2/nceo_generic/satellite_data/modis_c6/myd021km/2014/'
    orac_data_path = '/home/users/dnfisher/nceo_aerosolfire/data/orac_proc/myd/2014/'
    mask_path = '/home/users/dnfisher/nceo_aerosolfire/data/plume_masks/myd021km_plumes_df.pickle'
    output = '/home/users/dnfisher/nceo_aerosolfire/data/quicklooks/plume_retrievals/'
    output_txt = '/home/users/dnfisher/nceo_aerosolfire/data/plume_masks/'
    lut_class = 'BOW'

    # read in the masks
    mask_df = pd.read_pickle(mask_path)

    # set up geostationary projection to hold all the plumes
    m = Basemap(projection='geos', lon_0=-75, resolution='l')



    with open(output_txt + "plume_extents.txt", "w") as text_file:

        # iterate over modis files
        for primary_file in glob.glob(orac_data_path + '*/*/*' + lut_class + '.primary*'):

            # first get the primary file time
            primary_time = get_primary_time(primary_file)
            fname = output + primary_file.split('/')[-1][:-3] + '_quicklook'

            # find and read the vis channel of the associated l1b file
            l1b_file = glob.glob(l1b_path + '/*/*' + primary_time + '*')[0]
            visrad = read_vis(l1b_file)

            # open up the ORAC primary file
            primary_data = open_primary(primary_file)

            # make the smoke plume mask
            plume_mask = make_mask(primary_data, primary_time, mask_df)

            # get the individual plumes
            plume_positions = label_plumes(plume_mask)

            # visualise
            make_plume_plot(fname, visrad, primary_data, plume_positions)
            #make_plume_location_plot(plume_positions, primary_data, m)

            # let dump coords of plume to text file for Caroline
            for pp in plume_positions:
                text_file.write(primary_file.split('/')[-1] + " " +
                                str(pp[0]) + " " +
                                str(pp[1]) + " " +
                                str(pp[2]) + " " +
                                str(pp[3]) + "\n")

if __name__ == '__main__':
    main()
