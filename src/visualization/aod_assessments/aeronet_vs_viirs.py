import glob
import os
import logging
import re
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree

import src.config.filepaths as fp
import src.features.fre_to_tpm.viirs.ftt_utils as ut



def make_balltree_subset(array_lats_flat, array_lons_flat):
    array_lat_lon = np.dstack([np.deg2rad(array_lats_flat), np.deg2rad(array_lons_flat)])[0]
    return BallTree(array_lat_lon, metric='haversine')


def spatial_intersection_subset(array_balltree, point_lat, point_lon, x_positions_flat, y_positions_flat):
    # get the unique flare lats and lons for assessment in kdtree
    point_location = np.array([np.deg2rad(point_lat), np.deg2rad(point_lon)]).reshape(1, -1)

    # compare the flare locations to the potential locations in the orbit
    distance, index = array_balltree.query(point_location, k=1)

    # get indexes
    x = x_positions_flat[index][0][0]
    y = y_positions_flat[index][0][0]
    d = distance[0][0]

    return x, y, np.rad2deg(d)


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
    max_t_delta = 1800  # 30 minuts each side
    temporal_df = station[np.abs((station.index - timestamp).total_seconds()) <= max_t_delta]
    n_points_in_mean = temporal_df.shape[0]

    if temporal_df.empty:
        return 0, 0, 0, 0, 0, 0, 0

    # get station location
    station_lat = temporal_df.iloc[0]['Site_Latitude(Degrees)']
    station_lon = temporal_df.iloc[0]['Site_Longitude(Degrees)']

    # check if scene intersects
    x, y, d = spatial_intersection_subset(balltree,
                                             station_lat,
                                             station_lon,
                                             cols, rows)

    # lets only consider stations less than 1 arcminute distant to a pixel (2km)
    if d > 1 / 60.0:
        return 0, 0, 0, 0, 0, 0, 0

    print 'station lat', station_lat
    print 'station lon', station_lon

    # interpolate aod
    aod500 = temporal_df['AOD_500nm'].mean()
    aod675 = temporal_df['AOD_675nm'].mean()
    angstrom = temporal_df['440-675_Angstrom_Exponent'].mean()
    aod550 = interpolate_aod550(angstrom, aod675)

    return x, y, d, n_points_in_mean, aod550, aod500, aod675


def get_fname(path, timestamp):
    file_list = os.listdir(path)
    min_diff = 999999
    min_diff_ind = 0
    for i, f in enumerate(file_list):
        viirs_timestamp = re.search("[d][0-9]{8}[_][t][0-9]{4}", f).group()
        viirs_timestamp = datetime.strptime(viirs_timestamp, 'd%Y%m%d_t%H%M')
        diff = abs((timestamp - viirs_timestamp).total_seconds())
        if diff < min_diff:
            min_diff = diff
            min_diff_ind = i

    if min_diff <= 60:
        return file_list[min_diff_ind]
    else:
        return ''


def get_aod(sat_data_utm, x, y):

    sample_size = 10
    half_sample_size = sample_size / 2.0
    min_y = int(y-half_sample_size) if y-half_sample_size > 0 else 0
    min_x = int(x-half_sample_size) if x-half_sample_size > 0 else 0
    max_y = int(y+half_sample_size) if y+half_sample_size < sat_data_utm['orac_aod_utm'].shape[0] else \
        sat_data_utm['orac_aod_utm'].shape[0]
    max_x = int(x+half_sample_size) if x+half_sample_size < sat_data_utm['orac_aod_utm'].shape[1] else \
        sat_data_utm['orac_aod_utm'].shape[1]

    # do orac proc
    orac_aod_subset = sat_data_utm['orac_aod_utm'][min_y:max_y, min_x:max_x]
    orac_cost_subset = sat_data_utm['orac_flags_utm'][min_y:max_y, min_x:max_x]

    mask = orac_cost_subset <= 3
    n_orac = np.sum(mask)
    if n_orac:
        mean_orac_aod = np.mean(orac_aod_subset[mask])
        mean_orac_cost = np.mean(orac_cost_subset[mask])
    else:
        mean_orac_aod = -999
        mean_orac_cost = -999

    # do viirs
    viirs_aod_subset = sat_data_utm['viirs_aod_utm'][min_y:max_y, min_x:max_x]
    viirs_flag_subset = sat_data_utm['viirs_flags_utm'][min_y:max_y, min_x:max_x]

    mask = viirs_flag_subset == 0
    n_viirs = np.sum(mask)
    if n_viirs:
        mean_viirs_aod = np.mean(viirs_aod_subset[mask])
        mean_viirs_flag = np.mean(viirs_flag_subset[mask])
    else:
        mean_viirs_aod = -999
        mean_viirs_flag = -999

    return mean_orac_aod, mean_orac_cost, mean_viirs_aod, mean_viirs_flag, n_orac, n_viirs


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image[image > 0].flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def create_png(viirs_data, utm_rs, masked_lats, masked_lons, image_id, x, y, station):

    im_size = 200
    half_im_size = im_size/2

    # m1_params = viirs_data['All_Data']['VIIRS-M1-SDR_All']['RadianceFactors']
    m1 = viirs_data['All_Data']['VIIRS-M1-SDR_All']['Radiance'][:]
    # m4_params = viirs_data['All_Data']['VIIRS-M1-SDR_All']['RadianceFactors']
    m4 = viirs_data['All_Data']['VIIRS-M4-SDR_All']['Radiance'][:]
    # m5_params = viirs_data['All_Data']['VIIRS-M1-SDR_All']['RadianceFactors']
    m5 = viirs_data['All_Data']['VIIRS-M5-SDR_All']['Radiance'][:]

    resampled_m1 = utm_rs.resample_image(m1, masked_lats, masked_lons, fill_value=0)
    resampled_m4 = utm_rs.resample_image(m4, masked_lats, masked_lons, fill_value=0)
    resampled_m5 = utm_rs.resample_image(m5, masked_lats, masked_lons, fill_value=0)

    min_y = int(y-half_im_size) if y-half_im_size > 0 else 0
    min_x = int(x-half_im_size) if x-half_im_size > 0 else 0
    max_y = int(y+half_im_size) if y+half_im_size < resampled_m1.shape[0] else resampled_m1.shape[0]
    max_x = int(x+half_im_size) if x+half_im_size < resampled_m1.shape[1] else resampled_m1.shape[1]

    # subset to roi
    resampled_m1 = resampled_m1[min_y:max_y, min_x:max_x]
    resampled_m4 = resampled_m4[min_y:max_y, min_x:max_x]
    resampled_m5 = resampled_m5[min_y:max_y, min_x:max_x]

    # get station loc in scene
    diff_y = max_y - min_y
    diff_x = max_x - min_x
    if diff_y < im_size:
        if min_y == 0:
            pos_y = half_im_size - 1 - (im_size - diff_y)
        else:
            pos_y = half_im_size  # if greater we still use same position
    else:
        pos_y = half_im_size - 1
    if diff_x < im_size:
        if min_x == 0:
            pos_x = half_im_size - 1 - (im_size - diff_x)
        else:
            pos_x = half_im_size
    else:
        pos_x = half_im_size - 1

    # r = image_histogram_equalization(resampled_m5)
    # g = image_histogram_equalization(resampled_m4)
    # b = image_histogram_equalization(resampled_m1)
    r = resampled_m5
    g = resampled_m4
    b = resampled_m1

    r = np.round((r * (255 / np.max(r))) * 1).astype('uint8')
    g = np.round((g * (255 / np.max(g))) * 1).astype('uint8')
    b = np.round((b * (255 / np.max(b))) * 1).astype('uint8')

    rgb = np.dstack((r, g, b))

    plt.imshow(rgb)
    plt.plot(pos_x, pos_y, 'rx')

    # plot the area over which mean is computed
    plt.plot(pos_x-5, pos_y-5, 'bx')
    plt.plot(pos_x+5, pos_y-5, 'bx')
    plt.plot(pos_x-5, pos_y+5, 'bx')
    plt.plot(pos_x+5, pos_y+5, 'bx')

    plt.savefig(os.path.join(fp.path_to_aeronet_visuals, 'images',
                             'id_' + str(image_id) + '_station_' + station + '.png'),
                bbox_inches='tight')


def setup_sat_data(ts):

    dd = dict()
    dd['viirs_aod'] = ut.sat_data_reader(fp.path_to_viirs_aod, 'viirs', 'aod', ts)
    dd['viirs_flags'] = ut.sat_data_reader(fp.path_to_viirs_aod, 'viirs', 'flags', ts)
    dd['orac_aod'] = ut.sat_data_reader(fp.path_to_viirs_orac, 'orac', 'aod', ts)
    dd['orac_flags'] = ut.sat_data_reader(fp.path_to_viirs_orac, 'orac', 'flags', ts)

    lats, lons = ut.sat_data_reader(fp.path_to_viirs_orac, 'orac', 'geo', ts)
    dd['lats'] = lats
    dd['lons'] = lons
    return dd


def main():
    aeronet_station_data = load_aeronet()
    viirs_orac_filepaths = glob.glob(fp.path_to_viirs_orac + '*')

    image_id = 0
    data_dict = dict(x=[], y=[], dist=[], aod550=[], aod500=[], aod675=[], orac_aod=[], orac_cost=[],
                     viirs_aod=[], viirs_flag=[], image_id=[], orac_file=[], viirs_file=[], station=[],
                     n_points_in_aeronet_mean=[], n_orac=[], n_viirs=[])

    # iterate over VIIRS AOD files
    for o_f in viirs_orac_filepaths:

        print o_f
        # i_f = '/Volumes/INTENSO/Asia/processed/orac/viirs/KCL-NCEO-L2-CLOUD-CLD-VIIRS_ORAC_NPP_201508160633_R4591AMW.primary.nc'
        # if o_f != i_f:
        #    continue

        timestamp = ut.get_timestamp(o_f, 'orac')

        # check if station has intersection for given time stamp, if not continue
        if not aeronet_intersections(timestamp, aeronet_station_data):
            continue


        try:
            sat_data = setup_sat_data(timestamp)
        except Exception, e:
            logger.info('Could not load all datasets for: ' + str(timestamp) + '. Failed with error: ' + str(e))
            continue

        sat_data_utm = ut.resample_satellite_datasets(sat_data, fill_value=-999)

        # generate coordinate array from resampled grid
        rows = np.arange(sat_data_utm['lats'].shape[0])
        cols = np.arange(sat_data_utm['lats'].shape[1])
        cols, rows = np.meshgrid(cols, rows)

        # mask all points to valid
        mask = sat_data_utm['lats'] != -999
        resampled_lats_sub = sat_data_utm['lats'][mask]
        resampled_lons_sub = sat_data_utm['lons'][mask]
        cols = cols[mask]
        rows = rows[mask]

        balltree = make_balltree_subset(resampled_lats_sub, resampled_lons_sub)
        for station in aeronet_station_data:

            print
            print station

            station_df = aeronet_station_data[station]

            # locate aeronet station in scene
            x, y, dist, n_aeronet, aod550, aod500, aod675 = collocate_station(station_df,
                                                                               balltree, cols, rows,
                                                                               timestamp)

            # if nothing in scene continue
            if not x:
                print 'no station points within 30 minutes of overpass or within 2 arcminutes of image '
                continue

            print 'image lat', sat_data_utm['lats'][y, x]
            print 'image lon', sat_data_utm['lons'][y, x]

            # take valid mean AOD within 10km
            mean_orac_aod, mean_orac_cost, \
            mean_viirs_aod, mean_viirs_flag, \
                n_orac, n_viirs = get_aod(sat_data_utm, x, y)

            # append to dict
            data_dict['x'].append(x)
            data_dict['y'].append(y)
            data_dict['dist'].append(dist)
            data_dict['n_points_in_aeronet_mean'].append(n_aeronet)
            data_dict['n_orac'].append(n_orac)
            data_dict['n_viirs'].append(n_viirs)
            data_dict['aod550'].append(aod550)
            data_dict['aod500'].append(aod500)
            data_dict['aod675'].append(aod675)
            data_dict['orac_aod'].append(mean_orac_aod)
            data_dict['orac_cost'].append(mean_orac_cost)
            data_dict['viirs_aod'].append(mean_viirs_aod)
            data_dict['viirs_flag'].append(mean_viirs_flag)
            data_dict['image_id'].append(image_id)
            data_dict['orac_file'].append(o_f)
            data_dict['station'].append(station)

            # update image
            image_id += 1

    # convert dict to dataframe
    df = pd.DataFrame.from_dict(data_dict)

    # dump to csv
    df.to_csv(os.path.join(fp.path_to_dataframes, 'aeronet_comp.csv'))


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
