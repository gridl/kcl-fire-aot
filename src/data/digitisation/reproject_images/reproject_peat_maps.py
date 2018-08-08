import logging
import os
import glob
import re

import h5py
import numpy as np
import scipy.misc as misc
from netCDF4 import Dataset

import src.config.filepaths_cems as fp
import src.features.fre_to_tpm.modis.ftt_utils as ut

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
    z = nc_file['z'][:].reshape([3000,3000])
    mask = z > 0
    return {"mask": mask, "lats": lats, "lons": lons}


def get_timestamp(viirs_sdr_fname):
    try:
        return re.search("[d][0-9]{8}[_][t][0-9]{6}", viirs_sdr_fname).group()
    except Exception, e:
        logger.warning("Could not extract time stamp from: " + viirs_sdr_fname + " with error: " + str(e))
        return ''


def ds_names_dict(key, key_alt=None):
    if key == 'M03':
        return {'k1': 'VIIRS-M3-SDR_All', 'k2': 'Radiance'}
    if key == 'M04':
        return {'k1': 'VIIRS-M4-SDR_All', 'k2': 'Radiance'}
    if key == 'M05':
        return {'k1': 'VIIRS-M5-SDR_All', 'k2': 'Radiance'}
    if key_alt == 'Latitude':
        return {'k1': 'VIIRS-MOD-GEO-TC_All', 'k2': key_alt}
    if key_alt == 'Longitude':
        return {'k1': 'VIIRS-MOD-GEO-TC_All', 'k2': key_alt}
    if key_alt == 'faot550':
        return {'k1': 'VIIRS-Aeros-Opt-Thick-IP_All', 'k2': key_alt}
    if key_alt == 'QF1':
        return {'k1': 'VIIRS-Aeros-Opt-Thick-IP_All', 'k2': key_alt}


def get_viirs_fname(path, timestamp_viirs, key):
    fname = [f for f in os.listdir(path) if
             ((timestamp_viirs in f) and (key in f))]
    if len(fname) > 1:
        logger.warning("More that one frp granule matched STOP and check why")
        return fname[0]
    elif len(fname) == 1:
        return fname[0]
    else:
        logger.warning("No matching granule STOP and check why")
        return None


def read_h5(f):
    return h5py.File(f, "r")


def read_ds(path, ts, key, key_alt=None):
    # setup key
    p = ds_names_dict(key, key_alt)

    # get filename
    fname = get_viirs_fname(path, ts, key)

    # read h5
    ds = read_h5(os.path.join(path, fname))

    # return dataset
    return ds['All_Data'][p['k1']][p['k2']][:]


def setup_data(base_name):
    data_dict = {}

    # get timestampe
    ts = get_timestamp(base_name)

    data_dict['m3'] = read_ds(fp.path_to_viirs_sdr, ts, 'M03')
    data_dict['m4'] = read_ds(fp.path_to_viirs_sdr, ts, 'M04')
    data_dict['m5'] = read_ds(fp.path_to_viirs_sdr, ts, 'M05')
    data_dict['aod'] = read_ds(fp.path_to_viirs_aod, ts, 'AOT', key_alt='faot550')
    data_dict['flags'] = read_ds(fp.path_to_viirs_aod, ts, 'AOT', key_alt='QF1')
    data_dict['lats'] = read_ds(fp.path_to_viirs_geo, ts, 'TCO', key_alt='Latitude')
    data_dict['lons'] = read_ds(fp.path_to_viirs_geo, ts, 'TCO', key_alt='Longitude')
    print data_dict['lats']
    return data_dict


def create_resampler(data_dict):
    return ut.utm_resampler(data_dict['lats'],
                            data_dict['lons'],
                            750)


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image[image > 0].flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def tcc_viirs(data_dict, resampler):
    m3 = data_dict['m3']
    m4 = data_dict['m4']
    m5 = data_dict['m5']

    mask = m5<0
    masked_lats = np.ma.masked_array(resampler.lats, mask)
    masked_lons = np.ma.masked_array(resampler.lons, mask)

    resampled_m3 = resampler.resample_image(m3, masked_lats, masked_lons, fill_value=0)
    resampled_m4 = resampler.resample_image(m4, masked_lats, masked_lons, fill_value=0)
    resampled_m5 = resampler.resample_image(m5, masked_lats, masked_lons, fill_value=0)

    r = image_histogram_equalization(resampled_m5)
    g = image_histogram_equalization(resampled_m4)
    b = image_histogram_equalization(resampled_m3)

    r = np.round((r * (255 / np.max(r))) * 1).astype('uint8')
    g = np.round((g * (255 / np.max(g))) * 1).astype('uint8')
    b = np.round((b * (255 / np.max(b))) * 1).astype('uint8')

    rgb = np.dstack((r, g, b))
    return rgb


def main():

    # load in the peat maps
    peat_map_dict = {}
    peat_maps_paths = glob.glob(fp.path_to_peat_maps + '*/*/*.nc')
    for peat_maps_path in peat_maps_paths:
        peat_map_key = peat_maps_path.split("/")[-1].split(".")[0]
        peat_map_dict[peat_map_key] = read_nc(peat_maps_path)

    # iterate over viirs files
    viirs_sdr_paths = glob.glob(fp.path_to_viirs_sdr + 'SVM01*')
    for viirs_sdr_path in viirs_sdr_paths:

        sum_array = None
        viirs_sdr_filename = viirs_sdr_path.split('/')[-1]

        logger.info("Processing viirs file: " + viirs_sdr_filename)

        if 'DS' in viirs_sdr_filename:
            continue

        try:
            # read in the needed SDR data and create a data dict
            try:
                data_dict = setup_data(viirs_sdr_filename)
            except Exception, e:
                logger.warning('Could load data. Failed with ' + str(e))
                continue

            try:
                # setup resampler adn extract true colour
                utm_resampler = create_resampler(data_dict)
            except Exception, e:
                logger.warning('Could not make resampler for file. Failed with ' + str(e))
                continue

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
            merged_mask = sum_array > 0

            # blend the mask and the image
            blend_ratio = 0.3
            color_mask = np.dstack((merged_mask * 205, merged_mask * 74, merged_mask * 74))
            tcc = tcc_viirs(data_dict, utm_resampler)
            for i in xrange(tcc.shape[2]):
                tcc[:,:,i] = blend_ratio * color_mask[:,:,i] + (1-blend_ratio)*tcc[:,:,i]

            # write out
            output = viirs_sdr_filename.replace('h5', 'png')
            misc.imsave(os.path.join(fp.path_to_resampled_peat_map, output), tcc)

        except Exception, e:
            logger.warning('Could make resample peat map for file: ' + viirs_sdr_filename
                           + '. Failed with error: ' + str(e))
            continue





if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()