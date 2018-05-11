import glob
import os
import logging
import re
from datetime import datetime

import h5py
import pandas as pd
import numpy as np

import src.config.filepaths as fp
import src.features.fre_to_tpm.viirs.ftt_utils as ut


def read_aeronet(filename):
    """Read a given AERONET AOT data file, and return it as a dataframe.

    This returns a DataFrame containing the AERONET data, with the index
    set to the timestamp of the AERONET observations. Rows or columns
    consisting entirely of missing data are removed. All other columns
    are left as-is.
    """
    dateparse = lambda x: pd.datetime.strptime(x, "%d:%m:%Y %H:%M:%S")
    aeronet = pd.read_csv(filename, skiprows=6, na_values=[-999.0],
                          parse_dates={'times': [0, 1]},
                          date_parser=dateparse)

    aeronet = aeronet.set_index('times')

    # Drop any rows that are all NaN and any cols that are all NaN
    # & then sort by the index
    an = (aeronet.dropna(axis=1, how='all')
          .dropna(axis=0, how='all')
          .rename(columns={'Last_Processing_Date(dd/mm/yyyy)': 'Last_Processing_Date'})
          .sort_index())

    return an


def load_aeronet():
    aeronet_files = glob.glob(fp.path_to_aeronet + '/*/*.lev15')
    aeronet_dict = {}
    for af in aeronet_files:
        place = af.split('_')[-1].split('.')[0]
        ds = read_aeronet(af)
        aeronet_dict[place] = ds
    return aeronet_dict


def get_orac_timestamp(orac_path):
    orac_fname = orac_path.split('/')[-1]
    return datetime.strptime(orac_fname[37:47], "%Y%m%d%H%M")


def aeronet_intersections(timestamp, aeronet_station_data):
    for station in aeronet_station_data:
        time_deltas = np.abs((aeronet_station_data[station].index - timestamp).total_seconds())
        if min(time_deltas) <= 3600:  # lets say less than an hour diffence in obs time
            return True
    return False


def interpolate_aod550(angstrom, aod):
    return aod * (550./675)**(-angstrom)


def collocate_station(station, balltree, x_shape, timestamp):

    # first check if any datapoints with the hour
    temporal_df = station[np.abs((station.index - timestamp).total_seconds()) < 3600]

    if temporal_df.empty:
        return 0, 0, 0, 0, 0, 0, 0

    # now get temporally closest datapoint
    closest_pos = np.abs((temporal_df.index - timestamp).total_seconds()).argmin()
    time_delta = np.abs((temporal_df.index - timestamp).total_seconds()).min()
    closest_data = temporal_df.iloc[closest_pos]

    # check if scene intersects
    x, y, d = ut.spatial_intersection(balltree, x_shape,
                                      closest_data['Site_Latitude(Degrees)'],
                                      closest_data['Site_Longitude(Degrees)'])

    # lets only consider points less than 1 arcminute distant (2km)
    if d > 1/60.0:
        return 0, 0, 0, 0, 0, 0, 0

    # interpolate aod
    aod500 = closest_data['AOD_500nm']
    aod675 = closest_data['AOD_675nm']
    angstrom = closest_data['440-675_Angstrom_Exponent']
    aod550 = interpolate_aod550(angstrom, aod675)

    return x, y, d, time_delta, aod550, aod500, aod675


# def get_orac_fname(path, timestamp_viirs):
#     viirs_dt = datetime.strptime(timestamp_viirs, 'd%Y%m%d_t%H%M%S')
#     fname = [f for f in os.listdir(path) if
#              abs((viirs_dt - datetime.strptime(f[37:47], "%Y%m%d%H%M")).total_seconds()) <= 30]
#     if len(fname) > 1:
#         logger.warning("More that one frp granule matched selecting 0th option")
#         return fname[0]
#     elif len(fname) == 1:
#         return fname[0]
#     else:
#         return ''

def main():

    aeronet_station_data = load_aeronet()

    viirs_orac_filepaths = glob.glob(fp.path_to_viirs_orac + '*')

    # iterate over VIIRS AOD files
    for o_f in viirs_orac_filepaths:

        print o_f

        timestamp = get_orac_timestamp(o_f)

        # check if station has intersection for given time stamp, if not continue
        if not aeronet_intersections(timestamp, aeronet_station_data):
            continue

        # load in orac aod data
        orac_ds = ut.read_nc(o_f)
        lats, lons = ut.read_orac_geo(orac_ds)
        orac_aod = ut.orac_aod(orac_ds)

        # make balltree for orac coord data
        balltree = ut.make_balltree(lats, lons)

        # iterate aeronet station data
        for station in aeronet_station_data:

            print station

            station_df = aeronet_station_data[station]

            # locate aeronet station in scene
            x, y, dist, time_delta, aod550, aod500, aod675 = collocate_station(station_df, balltree,
                                                                               lats.shape[1], timestamp)

            # if nothing in scene continue
            if not x:
                continue


    # visualise




if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()