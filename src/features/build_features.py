import pandas as pd
import numpy as np

import src.data.readers as readers


def lc_subset():

    gt = ds.GetGeoTransform()

    #TODO move this outside of this function, just open file in here.
    # we dont want to load the whole image, just the part of interest
    lon_start = -2  #  run from W to E
    lon_stop = 2

    lat_start = 52  # runs from N to S
    lat_stop = 50

    x_start = (lon_start - gt[0]) / gt[1]
    x_stop = (lon_stop - gt[0]) / gt[1]
    x_range = int(x_stop - x_start)

    y_start = (lat_start - gt[3]) / gt[5]
    y_stop = (lat_stop - gt[3]) / gt[5]
    y_range = int(y_stop - y_start)

    x = np.arange(0, x_range, 1)
    y = np.arange(0, y_range, 1)
    grids = np.meshgrid(x, y)

    lons = lon_start + grids[0] * gt[1]
    lats = lat_start + grids[1] * gt[5]

    lc_data = ds.GetRasterBand(1).ReadAsArray(int(round(x_stop)),
                                              int(round(y_stop)),
                                              x_range,
                                              y_range)


# set up filepaths and similar

def main():

    # create df to hold the outputs
    output_df = pd.DataFrame()

    # read in non-plume specific files
    frp_data = readers.read_goes_frp()
    lc_data = readers.read_lc()

    # iterate over each plume in the plume mask dataframe
    orac_filename = ''
    plumes_masks = readers.read_plume_masks()
    for plume in plumes:

        if plumes.filename != orac_filename:
            orac_data = readers.read_orac()
            orac_filename = plumes.filename

        # open up plume specific data
        bg_masks = readers.read_bg_masks()

        # set up plumes mask (in line sample and geo coords)

        # get plumes AOD (using line sample, check plume manually and continue if unsuitable? Bow-tie effect, need to resample?)

        # get background AOD (using line sample, check background manually and continue if unsuitable?, Bow-tie effect, need to resample?)

        # get fires contained within plume (using geo coords and date time, if none then continue)

        # for fires get landsurface type

        # insert data into dataframe


