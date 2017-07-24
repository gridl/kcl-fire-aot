import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

from matplotlib.path import Path
from netCDF4 import Dataset
from datetime import datetime
from mpl_toolkits.basemap import Basemap

from src import config


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



def make_plume_location_plot(plume_positions, primary_data, m):
    for i, pp in enumerate(plume_positions):
        mean_lat = np.mean(primary_data.variables['lat'][pp[0]:pp[1], pp[2]:pp[3]])
        mean_lon = np.mean(primary_data.variables['lon'][pp[0]:pp[1], pp[2]:pp[3]])
        m.plot(mean_lon, mean_lat, 'r.',
               markeredgecolor='k',
               latlon=True)


def main():
    # set up paths
    orac_data_path = config.orac_file_path
    mask_path = config.plume_mask_file_path

    # read in the masks
    mask_df = pd.read_pickle(mask_path)

    # set up geostationary projection to hold all the plumes
    m = Basemap(projection='geos', lon_0=-75, resolution='i')

    # iterate over modis files
    for primary_file in glob.glob(orac_data_path + '*/*/*/*primary*'):

        # first get the primary file time
        primary_time = get_primary_time(primary_file)

        # open up the ORAC primary file
        primary_data = open_primary(primary_file)

        # make the smoke plume mask
        plume_mask = make_mask(primary_data, primary_time, mask_df)

        # get the individual plumes
        plume_positions = label_plumes(plume_mask)

        # visualise
        make_plume_location_plot(plume_positions, primary_data, m)

    m.drawcoastlines()
    # draw parallels and meridians.
    m.drawparallels(np.arange(-90., 120., 30.))
    m.drawmeridians(np.arange(0., 420., 30.))

    plt.savefig(config.root+'plume_locations.png', bbox_inches='tight')


if __name__ == '__main__':
    main()