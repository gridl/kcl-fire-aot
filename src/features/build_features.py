import glob
import os

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


def get_orac_fname(orac_file_path, plume):
    y = plume.filename[10:14]
    doy = plume.filename[14:17]
    time = plume.filename[18:22]
    return glob.glob(os.path.join(orac_file_path, y, doy, 'main', '*' + time + '*.primary.nc'))[0]



def main():


    # set up filepaths and similar
    root = '/Users/dnf/projects/kcl-fire-aot/data/'

    orac_file_path = root + 'processed/orac_proc/'
    goes_frp_file_path = root + 'processed/goes_frp/goes13_2014_fire_frp_atm.csv'
    plume_mask_file_path = root + 'processed/plume_masks/myd021km_plumes_df.pickle'
    lc_file_path = root + 'external/land_cover/GLOBCOVER_L4_200901_200912_V2.3.tif'

    # create df to hold the outputs
    output_df = pd.DataFrame()

    # read in non-plume specific files
    frp_data = readers.read_goes_frp(goes_frp_file_path)
    lc_data = readers.read_lc(lc_file_path)

    # iterate over each plume in the plume mask dataframe
    modis_filename = ''
    plumes_masks = readers.read_plume_masks(plume_mask_file_path)
    for index, plume in plumes_masks.iterrows():

        # if the plumes is from a different modis file, then
        # load in the correct ORAC processed file
        if plume.filename != modis_filename:
            try:
                orac_filename = get_orac_fname(orac_file_path, plume)
                orac_data = readers.read_orac(orac_filename)
                modis_filename = plume.filename
            except Exception, e:
                print e
                continue

        # open up plume specific data
        bg_masks = readers.read_bg_masks()

        # set up plumes mask (in line sample and geo coords)

        # get plumes AOD (using line sample, check plume manually and continue if unsuitable? Bow-tie effect, need to resample?)

        # get background AOD (using line sample, check background manually and continue if unsuitable?, Bow-tie effect, need to resample?)

        # get fires contained within plume (using geo coords and date time, if none then continue)

        # for fires get landsurface type

        # insert data into dataframe


if __name__ == "__main__":
    main()
