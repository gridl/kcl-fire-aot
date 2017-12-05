import logging
import os
import glob

import h5py
import numpy as np
import scipy.misc as misc
from netCDF4 import Dataset

import matplotlib.pyplot as plt

import src.config.filepaths as fp
import src.features.fre_to_tpm.modis.ftt_utils as ut


def read_h5(f):
    return h5py.File(f,  "r")


def create_resampler(viirs_data):
    lats = viirs_data['All_Data']['VIIRS-MOD-GEO_All']['Latitude'][:]
    lons = viirs_data['All_Data']['VIIRS-MOD-GEO_All']['Longitude'][:]
    return ut.utm_resampler(lats, lons, 750)


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


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image[image > 0].flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def tcc_viirs(viirs_data, resampler):
    #m1_params = viirs_data['All_Data']['VIIRS-M1-SDR_All']['RadianceFactors']
    m1 = viirs_data['All_Data']['VIIRS-M1-SDR_All']['Radiance'][:]
    #m4_params = viirs_data['All_Data']['VIIRS-M1-SDR_All']['RadianceFactors']
    m4 = viirs_data['All_Data']['VIIRS-M4-SDR_All']['Radiance'][:]
    #m5_params = viirs_data['All_Data']['VIIRS-M1-SDR_All']['RadianceFactors']
    m5 = viirs_data['All_Data']['VIIRS-M5-SDR_All']['Radiance'][:]

    mask = m5<0
    masked_lats = np.ma.masked_array(resampler.lats, mask)
    masked_lons = np.ma.masked_array(resampler.lons, mask)

    resampled_m1 = resampler.resample_image(m1, masked_lats, masked_lons, fill_value=0)
    resampled_m4 = resampler.resample_image(m4, masked_lats, masked_lons, fill_value=0)
    resampled_m5 = resampler.resample_image(m5, masked_lats, masked_lons, fill_value=0)

    r = image_histogram_equalization(resampled_m5)
    g = image_histogram_equalization(resampled_m4)
    b = image_histogram_equalization(resampled_m1)

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
    for viirs_sdr_fname in os.listdir(fp.path_to_viirs_sdr):

        sum_array = None

        logger.info("Processing viirs file: " + viirs_sdr_fname)

        if 'DS' in viirs_sdr_fname:
            continue

        try:
            viirs_sdr = read_h5(os.path.join(fp.path_to_viirs_sdr, viirs_sdr_fname))

            # setup resampler
            utm_resampler = create_resampler(viirs_sdr)

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
            tcc = tcc_viirs(viirs_sdr, utm_resampler)
            for i in xrange(tcc.shape[2]):
                tcc[:,:,i] = blend_ratio * color_mask[:,:,i] + (1-blend_ratio)*tcc[:,:,i]

            # write out
            output = viirs_sdr_fname.replace('h5', 'png')
            misc.imsave(os.path.join(fp.path_to_resampled_peat_map, output), tcc)

        except Exception, e:
            logger.warning('Could make resample peat map for file: ' + viirs_sdr_fname
                           + '. Failed with error: ' + str(e))
            continue





if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()