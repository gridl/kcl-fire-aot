#!/home/users/dnfisher/soft/virtual_envs/kcl-fire-aot/bin/python2
import os
from datetime import datetime
import re
import sys

import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import src.data.readers.load_hrit as load_hrit


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




def main():

    # read in the atsr prodcut and land water
    geo_f = sys.argv[1]
    fname = sys.argv[2]
    him_segment = int(sys.argv[3])
    approx_y =int(sys.argv[4])
    approx_x = int(sys.argv[5])
    output_path = int(sys.argv[6])

    ts = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", fname).group(),
                           '%Y%m%d_%H%M')

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
    plt.figure(figsize=(8,15))
    plt.imshow(rgb)
    ts = ts.replace(microsecond=0).isoformat()
    plt.text(0, -30, ts)
    plt.plot(100, 50, 'r*')
    plt.axis('off')
    plt.savefig(os.path.join(output_path, fname), bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
