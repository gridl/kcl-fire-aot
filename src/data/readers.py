import pandas as pd

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
    return gdal.Open(lc_file_path)


def main():

    root = '/Users/dnf/git/kcl-fire-aot/data/'

    orac_file_path = root + 'processed/orac_proc/2014/092/main/KCL-NCEO-L2-CLOUD-CLD-MODIS_ORAC_AQUA_201404021845_R4591WAT.primary.nc'
    goes_frp_file_path = root + 'processed/goes_frp/goes13_2014_fire_frp_atm.csv'
    plume_mask_file_path = root + 'processed/plume_masks/myd021km_plumes_df.pickle'
    lc_file_path = root + 'external/land_cover/GLOBCOVER_L4_200901_200912_V2.3.tif'

    res = read_orac(orac_file_path)
    res = read_goes_frp(goes_frp_file_path)
    res = read_plume_masks(plume_mask_file_path)
    res = read_lc(lc_file_path)

if __name__ == "__main__":
    main()