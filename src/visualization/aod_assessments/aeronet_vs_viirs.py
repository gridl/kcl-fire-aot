import glob
import os
import logging
import re
from datetime import datetime

import pandas as pd
import numpy as np
import scipy.misc as misc

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
    return datetime.strptime(orac_fname[37:49], "%Y%m%d%H%M")


def aeronet_intersections(timestamp, aeronet_station_data):
    for station in aeronet_station_data:
        time_deltas = np.abs((aeronet_station_data[station].index - timestamp).total_seconds())
        if min(time_deltas) <= 3600:  # lets say less than an hour diffence in obs time
            return True
    return False


def interpolate_aod550(angstrom, aod):
    return aod * (550. / 675) ** (-angstrom)


def collocate_station(station, balltree, cols, rows, timestamp):
    # first check if any datapoints with the hour
    temporal_df = station[np.abs((station.index - timestamp).total_seconds()) < 3600]

    if temporal_df.empty:
        return 0, 0, 0, 0, 0, 0, 0

    # now get temporally closest datapoint
    closest_pos = np.abs((temporal_df.index - timestamp).total_seconds()).argmin()
    time_delta = np.abs((temporal_df.index - timestamp).total_seconds()).min()
    closest_data = temporal_df.iloc[closest_pos]

    # check if scene intersects
    x, y, d = ut.spatial_intersection_subset(balltree,
                                             closest_data['Site_Latitude(Degrees)'],
                                             closest_data['Site_Longitude(Degrees)'],
                                             cols, rows)

    print 'station lat', closest_data['Site_Latitude(Degrees)']
    print 'station lon', closest_data['Site_Longitude(Degrees)']

    # lets only consider points less than 1 arcminute distant (2km)
    if d > 1 / 60.0:
        return 0, 0, 0, 0, 0, 0, 0

    # interpolate aod
    aod500 = closest_data['AOD_500nm']
    aod675 = closest_data['AOD_675nm']
    angstrom = closest_data['440-675_Angstrom_Exponent']
    aod550 = interpolate_aod550(angstrom, aod675)

    return x, y, d, time_delta, aod550, aod500, aod675


def get_fname(path, timestamp):
    for f in os.listdir(path):
        viirs_timestamp = re.search("[d][0-9]{8}[_][t][0-9]{6}", f).group()
        viirs_timestamp = datetime.strptime(viirs_timestamp, 'd%Y%m%d_t%H%M%S')
        if abs((timestamp - viirs_timestamp).total_seconds()) <= 120:
            return f
    return ''


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image[image > 0].flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def create_png(viirs_data, utm_rs, masked_lats, masked_lons, image_id):

    # m1_params = viirs_data['All_Data']['VIIRS-M1-SDR_All']['RadianceFactors']
    m1 = viirs_data['All_Data']['VIIRS-M1-SDR_All']['Radiance'][:]
    # m4_params = viirs_data['All_Data']['VIIRS-M1-SDR_All']['RadianceFactors']
    m4 = viirs_data['All_Data']['VIIRS-M4-SDR_All']['Radiance'][:]
    # m5_params = viirs_data['All_Data']['VIIRS-M1-SDR_All']['RadianceFactors']
    m5 = viirs_data['All_Data']['VIIRS-M5-SDR_All']['Radiance'][:]

    resampled_m1 = utm_rs.resample_image(m1, masked_lats, masked_lons, fill_value=0)
    resampled_m4 = utm_rs.resample_image(m4, masked_lats, masked_lons, fill_value=0)
    resampled_m5 = utm_rs.resample_image(m5, masked_lats, masked_lons, fill_value=0)

    r = image_histogram_equalization(resampled_m5)
    g = image_histogram_equalization(resampled_m4)
    b = image_histogram_equalization(resampled_m1)

    r = np.round((r * (255 / np.max(r))) * 1).astype('uint8')
    g = np.round((g * (255 / np.max(g))) * 1).astype('uint8')
    b = np.round((b * (255 / np.max(b))) * 1).astype('uint8')

    rgb = np.dstack((r, g, b))
    misc.imsave(os.path.join(fp.path_to_viirs_sdr_resampled, image_id + 'png'), rgb)


def main():
    aeronet_station_data = load_aeronet()
    viirs_orac_filepaths = glob.glob(fp.path_to_viirs_orac + '*')

    image_id = 0
    data_dict = dict(x=[], y=[], dist=[], time_delta=[], aod550=[], aod500=[], aod675=[], orac_aod=[], orac_cost=[],
                     viirs_aod=[], viirs_flag=[], image_id=[], orac_file=[], viirs_file=[], station=[])

    # iterate over VIIRS AOD files
    for o_f in viirs_orac_filepaths:

        print o_f

        timestamp = get_orac_timestamp(o_f)

        # check if station has intersection for given time stamp, if not continue
        if not aeronet_intersections(timestamp, aeronet_station_data):
            continue

        # load in orac data and resample
        orac_ds = ut.read_nc(o_f)
        orac_aod = ut.orac_aod(orac_ds)
        lats, lons = ut.read_orac_geo(orac_ds)
        utm_rs = ut.utm_resampler(lats, lons, 750)

        null_mask = np.ma.getmask(orac_aod)
        masked_lats = np.ma.masked_array(utm_rs.lats, null_mask)
        masked_lons = np.ma.masked_array(utm_rs.lons, null_mask)

        resampled_lats = utm_rs.resample_image(utm_rs.lats, masked_lats, masked_lons, fill_value=-999)
        resampled_lons = utm_rs.resample_image(utm_rs.lons, masked_lats, masked_lons, fill_value=-999)

        # generate coordinate array from resampled grid
        rows = np.arange(resampled_lats.shape[0])
        cols = np.arange(resampled_lats.shape[1])
        cols, rows = np.meshgrid(cols, rows)

        # mask all points to valid
        mask = resampled_lats != -999
        resampled_lats_sub = resampled_lats[mask]
        resampled_lons_sub = resampled_lons[mask]
        cols = cols[mask]
        rows = rows[mask]

        balltree = ut.make_balltree_subset(resampled_lats_sub, resampled_lons_sub)

        # iterate aeronet station data
        ds_loaded = False
        for station in aeronet_station_data:

            print station

            station_df = aeronet_station_data[station]

            # locate aeronet station in scene
            x, y, dist, time_delta, aod550, aod500, aod675 = collocate_station(station_df,
                                                                               balltree, cols, rows,
                                                                               timestamp)

            print 'image lat', resampled_lats[y, x]
            print 'image lon', resampled_lons[y, x]

            # if nothing in scene continue
            if not x:
                continue

            # load datasets if not done already
            if not ds_loaded:

                r_orac_aod = utm_rs.resample_image(orac_aod, masked_lats, masked_lons, fill_value=0)
                r_orac_cost = utm_rs.resample_image(ut.orac_cost(orac_ds), masked_lats, masked_lons, fill_value=0)

                viirs_aod_fname = get_fname(fp.path_to_viirs_aod, timestamp)
                viirs_aod_ds = ut.read_h5(os.path.join(fp.path_to_viirs_aod, viirs_aod_fname))
                r_viirs_aod = utm_rs.resample_image(ut.viirs_aod(viirs_aod_ds), masked_lats, masked_lons, fill_value=0)
                r_viirs_flag = utm_rs.resample_image(ut.viirs_flags(viirs_aod_ds), masked_lats, masked_lons,
                                                     fill_value=0)

                viirs_sdr_fname = get_fname(fp.path_to_viirs_sdr, timestamp)
                viirs_sdr_ds = ut.read_h5(os.path.join(fp.path_to_viirs_sdr, viirs_sdr_fname))
                ds_loaded = True

            # sort out image
            # create_png(viirs_sdr_ds, utm_rs, masked_lats, masked_lons, image_id)

            # append to dict
            data_dict['x'].append(x)
            data_dict['y'].append(y)
            data_dict['dist'].append(dist)
            data_dict['dist'].append(dist)
            data_dict['time_delta'].append(time_delta)
            data_dict['aod550'].append(aod550)
            data_dict['aod500'].append(aod500)
            data_dict['aod675'].append(aod675)
            data_dict['orac_aod'].append(r_orac_aod[y, x])
            data_dict['orac_cost'].append(r_orac_cost[y, x])
            data_dict['viirs_aod'].append(r_viirs_aod[y, x])
            data_dict['viirs_flag'].append(r_viirs_flag[y, x])
            data_dict['image_id'].append(image_id)
            data_dict['orac_file'].append(o_f)
            data_dict['viirs_file'].append(viirs_aod_fname)
            data_dict['station'].append(station)

            # update image
            image_id += 1

            # convert dict to dataframe

            # dump to csv


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
