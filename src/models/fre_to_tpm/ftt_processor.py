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
from netCDF4 import Dataset
from pyhdf.SD import SD, SDC
from matplotlib.path import Path
from scipy import stats
from shapely.geometry import Polygon, Point

import src.config.filepaths as filepaths
import scipy.ndimage as ndimage


def load_frp():

    try:
        frp_df = pd.read_pickle(filepaths.path_to_himawari_frp + 'himawari_df.p')
    except Exception, e:
        logger.info('could not load frp dataframe, failed with error ' + str(e) + ' building anew')
        frp_files = os.listdir(filepaths.path_to_himawari_frp)
        df_from_each_file = (pd.read_csv(os.path.join(filepaths.path_to_himawari_frp, f)) for f in frp_files)
        frp_df = pd.concat(df_from_each_file, ignore_index=True)

        # lets dump the columns we don't want
        frp_df = frp_df[['FIRE_CONFIDENCE', 'FRP_0', 'LATITUDE', 'LONGITUDE', 'year','month','day','time']]

        # make geocoords into shapely points
        points = [Point(p[0], p[1]) for p in zip(frp_df['LONGITUDE'].values, frp_df['LATITUDE'].values)]
        frp_df['points'] = points
        frp_df.drop(['LONGITUDE', 'LATITUDE'], axis=1, inplace=True)

        # reindex onto date
        for k in ['year', 'month', 'day', 'time']:
            frp_df[k] = frp_df[k].astype(int).astype(str)
            if k == 'time':
                frp_df[k] = frp_df[k].str.zfill(4)
            if k in ['month', 'day']:
                frp_df[k] = frp_df[k].str.zfill(2)

        format = '%Y%m%d%H%M'
        frp_df['datetime'] = pd.to_datetime(frp_df['year'] +
                                            frp_df['month'] +
                                            frp_df['day'] +
                                            frp_df['time'], format=format)
        frp_df.drop(['year','month','day','time'], axis=1, inplace=True)

        frp_df = frp_df.set_index('datetime')
        frp_df.to_pickle(filepaths.path_to_himawari_frp + 'himawari_df.p')

    return frp_df


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

        # image is flipped, so we need to reverse the lat coordinate
        l = -(l + 1)

        lc_list.append(np.array(landcover['Band1'][(l - 1):l, s:s + 1][0])[0])

    # return the most common landcover class for the fire contined in the ROI
    return stats.mode(lc_list).mode[0]


def get_orac_data(plume, orac_file_path):
    y = plume.filename[10:14]
    doy = plume.filename[14:17]
    time = plume.filename[18:22]
    orac_file = glob.glob(os.path.join(orac_file_path, y, doy, 'main', '*' + time + '*.primary.nc'))[0]
    return Dataset(orac_file)


def build_polygon(plume, orac_data):

    # TODO replace this with ORAC data, not using L1B data
    # get geographic coordinates of plume bounds (first test with l2 data)
    myd_data = SD(os.path.join(filepaths.path_to_modis_l1b, plume.filename), SDC.READ)
    lats = ndimage.zoom(myd_data.select('Latitude').get(), 5)
    lons = ndimage.zoom(myd_data.select('Longitude').get(), 5)

    bounding_lats = [lats[point[0], point[1]] for point in plume.plume_extent]
    bounding_lons = [lons[point[0], point[1]] for point in plume.plume_extent]

    return Polygon(zip(bounding_lons, bounding_lats))


def compute_fre(plume_polygon, frp_data):
    pass



def main():

    # set the filepaths up here
    root_path = '/Users/dnf/Projects/kcl-fire-aot/data/'
    landcover_path = root_path + 'Global/land_cover/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7_900m.nc'
    mask_path = root_path + 'Asia/processed/plume_masks/myd021km_plumes_df.pickle'

    # open up the landcover dataset
    landcover_data = Dataset(landcover_path)

    # load all geostationary frp data into geopandas dataframes
    frp_data = load_frp()

    try:
        mask_df = pd.read_pickle(mask_path)
    except:
        mask_df = pd.read_csv(mask_path, quotechar='"', sep=',', converters={'plume_extent': ast.literal_eval})

    for index, plume in mask_df.iterrows():

        # find landcover type
        # plume_lc_class = find_landcover_class(plume, landcover_data)

        # get orac file for plume
        # orac_data = get_orac_fname(plume, orac_file_path)
        orac_data = []

        # convert plume into polygon
        plume_polygon = build_polygon(plume, orac_data)

        # find fre for the plume
        plume_fre = compute_fre(plume_polygon, frp_data)

        # find aod / tpm for the plume

        # store output




if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
