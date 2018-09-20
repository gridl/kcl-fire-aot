import os
from datetime import datetime, timedelta
import glob
import re

import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

import src.data.readers.load_hrit as load_hrit





def approx_loc_index(pp):
    """
    Get approximate location of a geographic point is an array of
    geographic coordinates
    """

    lat_abs_diff = np.abs(pp['geostationary_lats'] - pp['lat'])
    lon_abs_diff = np.abs(pp['geostationary_lons'] - pp['lon'])
    approx_y, approx_x = np.argmin(lat_abs_diff + lon_abs_diff)
    return approx_y, approx_x


def find_segment(approx_y):
    # there are ten 1100 pixel segments in himawari 1 km data
    seg_size = 1100
    min_segment = approx_y / seg_size + 1
    return min_segment


def adjust_y_segment(approx_y, segment):
    seg_size = 1100
    return approx_y - (segment * seg_size)


def get_geostationary_fnames(pp, day, image_segment):
    """

    :param plume_time: the time of the MYD observation of the plume
    :param image_segment: the Himawari image segment
    :return: the geostationary files for the day of and the day before the fire
    """
    ym = str(pp['start_time'].year) + str(pp['start_time'].month).zfill(2)
    day = str(pp['start_time'].day + timedelta(days=day)).zfill(2)

    # get all files in the directory using glob with band 3 for main segment
    p = os.path.join(pp['him_base_path'], ym, day)
    return glob.glob(p + '/*/*/B01/*S' + str(image_segment).zfill(2) + '*')


def load_image(f, segment, channel):

    if channel != 'B01':
        f.replace('B01', channel)

    # load geostationary files for the segment
    rad_segment_1, _ = load_hrit.H8_file_read(f)

    # load for the next segment
    f_new = f.replace('S' + str(segment).zfill(2), 'S' + str(segment + 1).zfill(2))
    rad_segment_2, _ = load_hrit.H8_file_read(f_new)

    # concat the himawari files
    rad = np.vstack((rad_segment_1, rad_segment_2))

    return rad


def subset_image(im, y, x):
    min_y = y - 50
    min_x = x - 200
    if min_x < 0: min_x = 50
    if min_y < 0: min_y = 50

    max_y = min_y + 1000
    max_x = x + 200

    return im[min_y:max_y, min_x:max_x]


def proc_params():
    d = {}

    d['start_time'] = datetime(2012, 8, 20, 0, 0)
    d['n_days'] = 10

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
    for day in xrange(pp.days):

        geostationary_files_for_day = get_geostationary_fnames(pp, day, him_segment)
        for geo_f in geostationary_files_for_day:
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

            # save image
            fname = geo_f.split('/')[-1]
            fname.replace('h5', 'png')
            plt.imshow([im_B03, im_B02, im_B01])
            ts = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", fname).group(),
                                   '%Y%m%d_%H%M').replace(microsecond=0).isoformat()
            plt.text(-50, -50, ts)
            plt.savefig(os.path.join(pp['output_path'], fname), bbox_inches='tight')
            plt.close()


if __name__ == "__main__":
    main()
