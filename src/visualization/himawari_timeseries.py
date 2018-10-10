import os
from datetime import datetime, timedelta
import glob
import re

import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
plt.switch_backend('agg')


import src.data.readers.load_hrit as load_hrit





def approx_loc_index(pp):
    """
    Get approximate location of a geographic point is an array of
    geographic coordinates
    """

    lat_abs_diff = np.abs(pp['geostationary_lats'] - pp['lat'])
    lon_abs_diff = np.abs(pp['geostationary_lons'] - pp['lon'])
    approx_y, approx_x  = np.unravel_index((lat_abs_diff + lon_abs_diff).argmin(), lat_abs_diff.shape)
    return approx_y, approx_x


def find_segment(approx_y):
    # there are ten 1100 pixel segments in himawari 1 km data
    seg_size = 1100
    min_segment = approx_y / seg_size + 1
    return min_segment


def adjust_y_segment(approx_y, segment):
    seg_size = 1100
    return approx_y - ((segment - 1) * seg_size)


def get_geostationary_fnames(pp, day, image_segment):
    """

    :param plume_time: the time of the MYD observation of the plume
    :param image_segment: the Himawari image segment
    :return: the geostationary files for the day of and the day before the fire
    """
    ym = str(pp['start_time'].year) + str(pp['start_time'].month).zfill(2)
    day = str(int(pp['start_time'].day) + day).zfill(2)

    # get all files in the directory using glob with band 3 for main segment
    p = os.path.join(pp['him_base_path'], ym, day)
    return glob.glob(p + '/*/*/B01/*S' + str(image_segment).zfill(2) + '*')


def load_image(f, segment, channel):

    if channel != 'B01':
        f = f.replace('B01', channel)
    if channel == 'B03':
        f = f.replace('R10', 'R05')
    # load geostationary files for the segment
    rad_segment_1, _ = load_hrit.H8_file_read(f)

    # load for the next segment
    f_new = f.replace('S' + str(segment).zfill(2), 'S' + str(segment + 1).zfill(2))
    rad_segment_2, _ = load_hrit.H8_file_read(f_new)

    # concat the himawari files
    rad = np.vstack((rad_segment_1, rad_segment_2))

    return rad


def subset_image(im, y, x):
    
    y_shift = 50    
    x_shift = 100

    min_y = y - y_shift
    min_x = x - x_shift

    max_y = min_y + 550
    max_x = x + 250

    return im[min_y:max_y, min_x:max_x]


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image[image > 0].flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def equalise(arr):
    gamma = 2
    arr =  (arr - arr.min()) * (1./(arr.max() - arr.min()) * 255)
    arr = ((arr / 255.)**(1./gamma) * 255).astype(int) 
    return arr


def proc_params():
    d = {}

    d['start_time'] = datetime(2015, 9, 2, 0, 0)
    d['n_days'] = 15

    d['lat'] = 1.24
    d['lon'] = 103.84

    him_base_path = '/group_workspaces/cems2/nceo_generic/users/xuwd/Himawari8/'
    d['him_base_path'] = him_base_path

    geo_file = os.path.join(him_base_path, 'lcov', 'Himawari_lat_lon.img')
    geostationary_lats, geostationary_lons = load_hrit.geo_read(geo_file)
    zoom = 2  # zoom if using 1km himawara data (B03) for tracking
    d['geostationary_lats'] = ndimage.zoom(geostationary_lats, zoom)
    d['geostationary_lons'] = ndimage.zoom(geostationary_lons, zoom)

    d['output_path'] = '/home/users/dnfisher/data/kcl-fire-aot/him_gif_data'
    return d


def main():
    pp = proc_params()
    approx_y, approx_x = approx_loc_index(pp)
    him_segment = find_segment(approx_y)
    approx_y = adjust_y_segment(approx_y, him_segment)

    # iterate over the days
    for day in xrange(pp['n_days']):
        geostationary_files_for_day = get_geostationary_fnames(pp, day, him_segment)
        for geo_f in geostationary_files_for_day:
            fname = geo_f.split('/')[-1]
            ts = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", fname).group(),
                                   '%Y%m%d_%H%M')    
            if (int(ts.hour) > 11) & (int(ts.hour) < 22):
                continue
            print geo_f  
 
            im_B01 = load_image(geo_f, him_segment, 'B01')
            im_B02 = load_image(geo_f, him_segment, 'B02')
            im_B03 = load_image(geo_f, him_segment, 'B03')
            

            # rescale B03
            im_B03 = ndimage.zoom(im_B03, 0.5)

            # subset to ROI
            im_B01 = subset_image(im_B01, approx_y, approx_x)
            im_B02 = subset_image(im_B02, approx_y, approx_x)
            im_B03 = subset_image(im_B03, approx_y, approx_x)
            
            # equalise
            im_B01 = equalise(im_B01)
            im_B02 = equalise(im_B02)
            im_B03 = equalise(im_B03)
            rgb = np.dstack([im_B03, im_B02, im_B01])

            # save image
            fname = fname.replace('DAT.bz2', 'png')
            #plt.imshow(im_B03, cmap='gray')
            plt.figure(figsize=(8,15))
            plt.imshow(rgb)
            ts = ts.replace(microsecond=0).isoformat()
            plt.text(0, -30, ts)
            plt.plot(100, 50, 'r*')
            plt.axis('off')
            plt.savefig(os.path.join(pp['output_path'], fname), bbox_inches='tight')
            plt.close()


if __name__ == "__main__":
    main()
