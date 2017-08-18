'''
Going from FRE to TPM for different land cover types.

We generate a linear model that relates FRE to TPM.

Basic logic:
    Load in land cover data
    For each digitised file:
        Resample landcover map to modis scene
        For each plume:
            Get plume AOD
            Get background AOD
            Get landcover type from MODIS fire pixels
            Get FRE from geostationary sensor



'''

import ast
import logging
import glob
import os

import pandas as pd
import numpy as np
import pyresample as pr
from netCDF4 import Dataset
from pyhdf.SD import SD, SDC
from matplotlib.path import Path
from scipy import stats


import src.config.filepaths as filepaths


def get_orac_fname(orac_file_path, plume):
    y = plume.filename[10:14]
    doy = plume.filename[14:17]
    time = plume.filename[18:22]
    return glob.glob(os.path.join(orac_file_path, y, doy, 'main', '*' + time + '*.primary.nc'))[0]

def find_landcover_class(plume, landcover):

    y = plume.filename[10:14]
    doy = plume.filename[14:17]
    time = plume.filename[18:22]

    myd14 = glob.glob(os.path.join(filepaths.path_to_modis_frp, '*A' + y + doy + '.' + time + '*.hdf'))[0]
    myd14 = SD(myd14, SDC.READ)
    
    lines = myd14.select('FP_line').get()
    samples = myd14.select('FP_sample').get()
    lats = myd14.select('FP_latitude').get()
    lons = myd14.select('FP_longitude').get()

    poly_verts = plume['plume_extent']
    bb_path = Path(poly_verts)

    # find the geographic coordinates of fires inside the plume mask
    lat_list = []
    lon_list = []
    for l, s, lat, lon in zip(lines, samples, lats, lons):
        if bb_path.contains_point((s, l)):
            lat_list.append(lat)
            lon_list.append(lon)

    # now get the landcover points
    lc_list = []
    for lat, lon in zip(lat_list, lon_list):
        s = int((lon - (-180)) / 360 * landcover['lon'].size)  # lon index
        l = int((lat - 90) * -1 / 180 * landcover['lat'].size)  # lat index
        lc_list.append(np.array(landcover['Band1'][s:s+1, l:l+1][0]))

    # return the most common landcover class for the fire contined in the ROI
    return stats.mode(lc_list)



def main():

    # set the filepaths up here
    root_path = '/Users/dnf/Projects/kcl-fire-aot/data/'
    landcover_path = root_path + 'Global/land_cover/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7_900m.nc'
    mask_path = root_path + 'Americas/processed/plume_masks/myd021km_plumes_df.pickle'

    # open up the landcover dataset
    landcover = Dataset(landcover_path)

    try:
        mask_df = pd.read_pickle(mask_path)
    except:
        mask_df = pd.read_csv(mask_path, quotechar='"', sep=',', converters={'plume_extent': ast.literal_eval})

    for index, plume in mask_df.iterrows():

        # find landcover type
        fire_lc_class = find_landcover_class(plume, landcover)

        # find aod / tpm

        # find fre




if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
