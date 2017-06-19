import glob
import os

import pandas as pd
import numpy as np

import src.data.readers as readers
import config
import resampling


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

    # create df to hold the outputs
    output_df = pd.DataFrame()

    # read in non-plume specific files
    frp_data = readers.read_goes_frp(config.goes_frp_file_path)
    lc_data = readers.read_lc(config.lc_file_path)

    # read in plume specific files
    plume_masks = readers.read_plume_data(config.plume_mask_file_path)
    plume_backgrounds = readers.read_plume_data(config.plume_background_file_path)

    # iterate over each plume in the plume mask dataframe
    modis_filename = ''
    for index, plume in plume_masks.iterrows():

        # load in orac file.  If the plumes is from
        # a different modis file, then load in the
        # correct ORAC processed file
        if plume.filename != modis_filename:
            modis_filename = plume.filename
            try:
                orac_filename = get_orac_fname(config.orac_file_path, plume)
                orac_data = readers.read_orac(orac_filename)
            except Exception, e:
                print e
                continue

        # extract background data for plume
        background = plume_backgrounds[plume_backgrounds.plume_id == plume.plume_id]

        # resample plume AOD to specified grid resolution
        resampled_plume = resampling.resampler(orac_data, plume)

        # resample background AOD to specified grid resolution

        # get fires contained within plume (using geo coords and date time, if none then continue)

        # get fire landsurface type

        # insert into dataframe


if __name__ == "__main__":
    main()
