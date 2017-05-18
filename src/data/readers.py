import pandas as pd
import numpy as np

from netCDF4 import Dataset
from osgeo import gdal


def read_orac(orac_file_path):
    '''

    :param orac_file_path: path to orac nc file
    :return: opened orac nc file
    '''
    #TODO Do we want to extract the ORAC data into a dict in here?
    return Dataset(orac_file_path)


def read_goes_frp(goes_frp_file_path):
    '''

    :param goes_frp_file_path: path to goes frp data
    :return: opened goes FRP data set as a dataframe
    '''
    return pd.read_csv(goes_frp_file_path)


def read_plume_masks(plume_mask_file_path):
    '''

    :param plume_mask_file_path: path to digited plume mask
    :return: plume mask locations
    '''
    return pd.read_pickle(plume_mask_file_path)


def read_lc(lc_file_path):
    '''

    :param lc_file_path: path to landcover file
    :return: opened landcover file
    '''

    ds = gdal.Open(lc_file_path)

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

    return ds

def main():

    root = '/Users/dnf/git/kcl-fire-aot/data/'

    orac_file_path = root + 'processed/orac_proc/2014/092/main/KCL-NCEO-L2-CLOUD-CLD-MODIS_ORAC_AQUA_201404021845_R4591WAT.primary.nc'
    goes_frp_file_path = root + 'processed/goes_frp/goes13_2014_fire_frp_atm.csv'
    plume_mask_file_path = root + 'processed/plume_masks/myd021km_plumes_df.pickle'
    lc_file_path = root + 'external/land_cover/GLOBCOVER_L4_200901_200912_V2.3.tif'

    res = read_lc(lc_file_path)
    res = read_orac(orac_file_path)
    res = read_goes_frp(goes_frp_file_path)
    res = read_plume_masks(plume_mask_file_path)


if __name__ == "__main__":
    main()