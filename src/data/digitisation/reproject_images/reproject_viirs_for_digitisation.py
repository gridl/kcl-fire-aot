import logging
import os
import re
import glob

import h5py
import numpy as np
import scipy.misc as misc
from datetime import datetime, timedelta
from netCDF4 import Dataset
from shapely.geometry import Point
import pandas as pd

import src.config.filepaths as fp
import src.features.fre_to_tpm.viirs.ftt_utils as ut
import src.features.fre_to_tpm.viirs.ftt_fre as ff


def get_timestamp(viirs_sdr_fname):
    try:
        return re.search("[d][0-9]{8}[_][t][0-9]{6}", viirs_sdr_fname).group()
    except Exception, e:
        logger.warning("Could not extract time stamp from: " + viirs_sdr_fname + " with error: " + str(e))
        return ''


def get_viirs_fname(path, timestamp_viirs, viirs_sdr_fname):
    fname = [f for f in os.listdir(path) if timestamp_viirs in f]
    if len(fname) > 1:
        logger.warning("More that one frp granule matched " + viirs_sdr_fname + "selecting 0th option")
        return fname[0]
    elif len(fname) == 1:
        return fname[0]
    else:
        return ''


def read_h5(f):
    return h5py.File(f, "r")


def read_nc(f):
    nc_file = Dataset(f)

    # get geo
    lon_dims = nc_file['x_range'][:]
    lat_dims = nc_file['y_range'][:]
    spacing = nc_file['spacing'][:]
    lons = np.arange(lon_dims[0], lon_dims[1], spacing[0])
    lats = np.flipud(np.arange(lat_dims[0], lat_dims[1], spacing[1]))
    lons, lats = np.meshgrid(lons, lats)

    # get mask
    z = nc_file['z'][:].reshape([3000, 3000])
    mask = z > 0
    return {"mask": mask, "lats": lats, "lons": lons}


def create_resampler(viirs_data):
    lats = viirs_data['All_Data']['VIIRS-MOD-GEO_All']['Latitude'][:]
    lons = viirs_data['All_Data']['VIIRS-MOD-GEO_All']['Longitude'][:]
    return ut.utm_resampler(lats, lons, 750)


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image[image > 0].flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def get_image_coords(fires, resampled_lats, resampled_lons):
    inverse_lats = resampled_lats * -1  # invert lats for correct indexing

    y_size, x_size = resampled_lats.shape

    min_lat = np.min(inverse_lats[inverse_lats > -1000])
    range_lat = np.max(inverse_lats) - min_lat

    min_lon = np.min(resampled_lons)
    range_lon = np.max(resampled_lons[resampled_lons < 1000]) - min_lon

    padding = 10

    x_coords = []
    y_coords = []

    for f in fires:
        f_lon, f_lat = f.xy

        # get approximate fire location, remembering to invert the lat
        y = int(((f_lat[0] * -1) - min_lat) / range_lat * y_size)
        x = int((f_lon[0] - min_lon) / range_lon * x_size)

        if (y <= padding) | (y >= y_size - padding):
            continue
        if (x <= padding) | (x >= x_size - padding):
            continue

        lat_subset = resampled_lats[y - padding:y + padding, x - padding:x + padding]
        lon_subset = resampled_lons[y - padding:y + padding, x - padding:x + padding]

        # find the location of the fire in the subset
        dists = np.abs(f_lat - lat_subset) + np.abs(f_lon - lon_subset)
        sub_y, sub_x = divmod(dists.argmin(), dists.shape[1])

        # using the subset location get the adjusted location
        y_coords.append(y - padding + sub_y)
        x_coords.append(x - padding + sub_x)

    return y_coords, x_coords


def get_arcmin(x):
    '''
    rounds the data decimal fraction of a degree
    to the nearest arc minute
    '''
    neg_values = x < 0

    abs_x = np.abs(x)
    floor_x = np.floor(abs_x)
    decile = abs_x - floor_x
    minute = np.around(decile * 60)  # round to nearest arcmin
    minute_fraction = minute * 0.01  # convert to fractional value (ranges from 0 to 0.6)

    max_minute = minute_fraction > 0.59

    floor_x[neg_values] *= -1
    floor_x[neg_values] -= minute_fraction[neg_values]
    floor_x[~neg_values] += minute_fraction[~neg_values]

    # deal with edge cases, and just round them all up
    if np.sum(max_minute) > 0:
        floor_x[max_minute] = np.around(floor_x[max_minute])

    # now to get rid of rounding errors and allow comparison multiply by 100 and convert to int
    floor_x = (floor_x * 100).astype(int)

    # round now to nearest 2 arcmin
    #floor_x = myround(floor_x, base=3)

    return floor_x


def myround(x, dec=20, base=.000005):
    return np.round(base * np.round(x / base), dec)


def fire_sampling(frp_df, time_stamp):

    # restrict to only fires within one hour of the overpass
    stop_time = time_stamp
    start_time = time_stamp - timedelta(minutes=70)
    frp_subset = ff.temporal_subset(frp_df, start_time, stop_time)
    frp_subset['occurrences'] = 1
    frp_subset['lons'] = [l.xy[0][0] for l in frp_subset.point.values]
    frp_subset['lats'] = [l.xy[1][0] for l in frp_subset.point.values]

    # round lats and lons to nearest arcminute
    frp_subset['lons_arcmin'] = get_arcmin(frp_subset['lons'].values)
    frp_subset['lats_arcmin'] = get_arcmin(frp_subset['lats'].values)

    # find all unique fire locations and count occurences
    agg_dict = {'occurrences': np.sum}
    grouped = frp_subset.groupby(['lons_arcmin', 'lats_arcmin'], as_index=False).agg(agg_dict)

    # get the point values back in the df
    points = frp_subset[['lats_arcmin', 'lons_arcmin', 'point']]
    points.drop_duplicates(['lats_arcmin', 'lons_arcmin'], inplace=True)
    grouped = pd.merge(grouped, points, on=['lats_arcmin', 'lons_arcmin'])

    return grouped


def fires_for_occurrence_level(frp_df, occurrences):

    # find and return fires with at least given number of occurence
    mask = frp_df.occurrences >= occurrences
    return frp_df[mask]


def tcc_viirs(viirs_data, fires_for_day, peat_mask, aeronet_stations, resampler, viirs_overpass_time):
    # m1_params = viirs_data['All_Data']['VIIRS-M1-SDR_All']['RadianceFactors']
    m1 = viirs_data['All_Data']['VIIRS-M1-SDR_All']['Radiance'][:]
    # m4_params = viirs_data['All_Data']['VIIRS-M1-SDR_All']['RadianceFactors']
    m4 = viirs_data['All_Data']['VIIRS-M4-SDR_All']['Radiance'][:]
    # m5_params = viirs_data['All_Data']['VIIRS-M1-SDR_All']['RadianceFactors']
    m5 = viirs_data['All_Data']['VIIRS-M5-SDR_All']['Radiance'][:]

    mask = m5 < 0
    masked_lats = np.ma.masked_array(resampler.lats, mask)
    masked_lons = np.ma.masked_array(resampler.lons, mask)

    resampled_m1 = resampler.resample_image(m1, masked_lats, masked_lons, fill_value=0)
    resampled_m4 = resampler.resample_image(m4, masked_lats, masked_lons, fill_value=0)
    resampled_m5 = resampler.resample_image(m5, masked_lats, masked_lons, fill_value=0)

    # for the fire coordinates in the image
    resampled_lats = resampler.resample_image(resampler.lats, masked_lats, masked_lons, fill_value=1000)
    resampled_lons = resampler.resample_image(resampler.lons, masked_lats, masked_lons, fill_value=1000)

    r = image_histogram_equalization(resampled_m5)
    g = image_histogram_equalization(resampled_m4)
    b = image_histogram_equalization(resampled_m1)

    r = np.round((r * (255 / np.max(r))) * 1).astype('uint8')
    g = np.round((g * (255 / np.max(g))) * 1).astype('uint8')
    b = np.round((b * (255 / np.max(b))) * 1).astype('uint8')

    rgb = np.dstack((r, g, b))
    rgb_peat = rgb.copy()

    # blend the mask and the image
    blend_ratio = 0.3
    colour_mask = np.dstack((peat_mask * 205, peat_mask * 74, peat_mask * 74))
    for i in xrange(rgb_peat.shape[2]):
        rgb_peat[:, :, i] = blend_ratio * colour_mask[:, :, i] + (1 - blend_ratio) * rgb_peat[:, :, i]

    # insert fires with colours based on sampling
    fire_occurrence_df = fire_sampling(fires_for_day, viirs_overpass_time)
    occurrences = [1, 3, 5, 6]  # number of himawari images fire is present in prior to overpass
    colour_sets = [[127, 76, 76], [255, 0, 0], [253, 106, 2], [0, 255, 0]]  # red/gray, red, orange, green
    for occurrence, colours in zip(occurrences, colour_sets):

        fires_with_occurrence = fires_for_occurrence_level(fire_occurrence_df, occurrence)

        if not fires_with_occurrence.empty:
            fy, fx = get_image_coords(fires_with_occurrence.point.values,
                                      resampled_lats, resampled_lons)
            if fy:
                rgb[fy, fx, 0] = colours[0]
                rgb[fy, fx, 1] = colours[1]
                rgb[fy, fx, 2] = colours[2]

                rgb_peat[fy, fx, 0] = colours[0]
                rgb_peat[fy, fx, 1] = colours[1]
                rgb_peat[fy, fx, 2] = colours[2]

    # insert aeronet stations
    fy, fx = get_image_coords(aeronet_stations, resampled_lats, resampled_lons)
    if fy:
        for x, y in zip(fx, fy):
            rgb[y-2:y+3, x-2:x+3, 0] = 0
            rgb[y-2:y+3, x-2:x+3, 1] = 255
            rgb[y-2:y+3, x-2:x+3, 2] = 255

            rgb_peat[y - 2:y + 3, x - 2:x + 3, 0] = 0
            rgb_peat[y - 2:y + 3, x - 2:x + 3, 1] = 255
            rgb_peat[y - 2:y + 3, x - 2:x + 3, 2] = 255

    return rgb, rgb_peat


def extract_aod(viirs_aod, resampler):
    aod = viirs_aod['All_Data']['VIIRS-Aeros-Opt-Thick-IP_All']['faot550'][:]
    mask = aod < -1
    masked_lats = np.ma.masked_array(resampler.lats, mask)
    masked_lons = np.ma.masked_array(resampler.lons, mask)
    resampled_aod = resampler.resample_image(aod, masked_lats, masked_lons, fill_value=0)

    return resampled_aod


def get_mask(arr, bit_pos, bit_len, value):
    '''Generates mask with given bit information.
    Parameters
        bit_pos		-	Position of the specific QA bits in the value string.
        bit_len		-	Length of the specific QA bits.
        value  		-	A value indicating the desired condition.
    '''
    bitlen = int('1' * bit_len, 2)

    if type(value) == str:
        value = int(value, 2)

    pos_value = bitlen << bit_pos
    con_value = value << bit_pos
    mask = (arr & pos_value) == con_value
    return mask


def extract_aod_flags(viirs_aod, resampler):
    aod = viirs_aod['All_Data']['VIIRS-Aeros-Opt-Thick-IP_All']['faot550'][:]
    aod_quality = viirs_aod['All_Data']['VIIRS-Aeros-Opt-Thick-IP_All']['QF1'][:]
    flags = np.zeros(aod_quality.shape)
    for k, v in zip(['00', '01', '10', '11'], [0, 1, 2, 3]):
        mask = get_mask(aod_quality, 0, 2, k)
        flags[mask] = v

    mask = aod < -1
    masked_lats = np.ma.masked_array(resampler.lats, mask)
    masked_lons = np.ma.masked_array(resampler.lons, mask)
    resampled_flags = resampler.resample_image(flags, masked_lats, masked_lons, fill_value=3)

    return resampled_flags


def get_peat_mask(peat_map_dict, utm_resampler):
    sum_array = None

    # extract three peat maps for the image
    for pm in peat_map_dict:

        resampled_mask = utm_resampler.resample_image(peat_map_dict[pm]['mask'],
                                                      peat_map_dict[pm]['lats'],
                                                      peat_map_dict[pm]['lons'],
                                                      fill_value=0)

        if sum_array is None:
            sum_array = resampled_mask.astype(int)
        else:
            sum_array += resampled_mask.astype(int)

    # merge the three peat maps
    return sum_array > 0


def get_aeronet():
    stations = dict(badung=(-6.888, 107.610), makassar=(-4.998, 119.572), puspiptek=(-6.356, 106.664),
                    jambi=(-1.632, 103.642), palangkaraya=(-2.228, 113.946), singapore=(1.298, 103.780),
                    kuching=(1.491, 110.349), pontianak=(0.075, 109.191))
    stations_points = []
    for k in stations:
        p = stations[k]
        stations_points.append(Point(p[1], p[0]))  # points needs to go in lon/lat

    return stations_points


def main():
    # load in himawari fires for visualisation
    frp_df = ut.read_frp_df(fp.path_to_himawari_frp)

    # load in the peat maps
    peat_map_dict = {}
    peat_maps_paths = glob.glob(fp.path_to_peat_maps + '*/*/*.nc')
    for peat_maps_path in peat_maps_paths:
        peat_map_key = peat_maps_path.split("/")[-1].split(".")[0]
        peat_map_dict[peat_map_key] = read_nc(peat_maps_path)

    for viirs_sdr_fname in os.listdir(fp.path_to_viirs_sdr):

        if os.path.isfile(os.path.join(
                fp.path_to_viirs_sdr_resampled, viirs_sdr_fname.replace('h5', 'png'))):
            print viirs_sdr_fname, 'already resampled'
            continue

        logger.info("Processing viirs file: " + viirs_sdr_fname)

        if 'DS' in viirs_sdr_fname:
            continue

        timestamp_viirs = get_timestamp(viirs_sdr_fname)
        if not timestamp_viirs:
            continue

        try:
            viirs_sdr = read_h5(os.path.join(fp.path_to_viirs_sdr, viirs_sdr_fname))
        except Exception, e:
            logger.warning('Could not read the input file: ' + viirs_sdr_fname + '. Failed with ' + str(e))
            continue

        try:
            # setup resampler adn extract true colour
            utm_resampler = create_resampler(viirs_sdr)
        except Exception, e:
            logger.warning('Could make resampler for file: ' + viirs_sdr_fname + '. Failed with ' + str(e))
            continue

        # get aod filename
        try:
            aod_fname = get_viirs_fname(fp.path_to_viirs_aod, timestamp_viirs, viirs_sdr_fname)
        except Exception, e:
            logger.warning('Could not load aux file for:' + viirs_sdr_fname + '. Failed with ' + str(e))
            continue

        if aod_fname:
            # load in viirs aod
            if aod_fname:
                try:
                    viirs_aod_data = read_h5(os.path.join(fp.path_to_viirs_aod, aod_fname))
                    viirs_aod = extract_aod(viirs_aod_data, utm_resampler)
                    aod_flags = extract_aod_flags(viirs_aod_data, utm_resampler)

                    misc.imsave(os.path.join(fp.path_to_viirs_aod_resampled, aod_fname.replace('h5', 'png')), viirs_aod)
                    misc.imsave(os.path.join(fp.path_to_viirs_aod_flags_resampled, aod_fname.replace('h5', 'png')),
                                aod_flags)

                except Exception, e:
                    logger.warning('Could not display aod file: ' + aod_fname)

        try:
            # setup resampler adn extract true colour
            t = datetime.strptime(timestamp_viirs, 'd%Y%m%d_t%H%M%S')

            peat_mask = get_peat_mask(peat_map_dict, utm_resampler)
            fires_for_day = ff.fire_locations_for_digitisation(frp_df, t)
            aeronet_stations = get_aeronet()
            tcc, tcc_peat = tcc_viirs(viirs_sdr, fires_for_day, peat_mask, aeronet_stations, utm_resampler, t)
            misc.imsave(os.path.join(fp.path_to_viirs_sdr_resampled, viirs_sdr_fname.replace('h5', 'png')), tcc)
            misc.imsave(os.path.join(fp.path_to_viirs_sdr_resampled,
                                     viirs_sdr_fname.replace('.h5', '_peat.png')), tcc_peat)
        except Exception, e:
            logger.warning('Could make image for file: ' + viirs_sdr_fname + '. Failed with ' + str(e))


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()
