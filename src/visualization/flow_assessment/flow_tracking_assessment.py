import os
import glob
import re
from datetime import datetime

import scipy.ndimage as ndimage
import numpy as np
import matplotlib.pyplot as plt

import src.data.readers.load_hrit as load_hrit
import src.config.filepaths as fp

# CONSTANTS
MAX_REFLEC = 0.05


def get_geostationary_fnames(ym, day, image_segment):
    """

    :param plume_time: the time of the MYD observation of the plume
    :param image_segment: the Himawari image segment
    :return: the geostationary files for the day of and the day before the fire
    """


    # get all files in the directory using glob with band 3 for main segment
    p = os.path.join(fp.path_to_himawari_imagery, ym, day)
    return glob.glob(p + '/*/*/B03/*S' + str(image_segment).zfill(2) + '*')


def sort_geostationary_by_time(geostationary_fnames):
    """

    :param geostationary_fnames goestationary filenames
    :return: the geostationary filenames in time order
    """
    times = [datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", f).group()
                               , '%Y%m%d_%H%M') for f in geostationary_fnames]
    return [f for _, f in sorted(zip(times, geostationary_fnames))]


def setup_geostationary_files(ym, day, image_segment):
    geostationary_fnames = get_geostationary_fnames(ym, day, image_segment)
    geostationary_fnames = sort_geostationary_by_time(geostationary_fnames)
    geostationary_fnames.reverse()
    return geostationary_fnames


def extract_observation(f, bb):
    # load geostationary files for the segment
    #rad_segment, _ = load_hrit.H8_file_read(os.path.join(fp.path_to_himawari_imagery, f))
    rad_segment, ref_segment = load_hrit.H8_file_read(os.path.join(fp.path_to_himawari_imagery, f))

    # extract geostationary image subset using adjusted bb
    rad_subset = rad_segment[bb['min_y']:bb['max_y'], bb['min_x']:bb['max_x']]
    ref_subset = ref_segment[bb['min_y']:bb['max_y'], bb['min_x']:bb['max_x']]
    return rad_subset, ref_subset



def main():

    bb = {'min_x': 4500,
          'min_y': 750,
          'max_x': 6500,
          'max_y': -1}

    # get vis filenames
    geostationary_file_names = setup_geostationary_files('201507', '06', 5)

    # iterate over vis files
    for f1, f2 in zip(geostationary_file_names[0:-1], geostationary_file_names[1:]):

        # read in the data for roi
        rad_1, ref_1 = extract_observation(f1, bb)
        rad_2, ref_2 = extract_observation(f2, bb)

        # generate cloud mask
        cm_1 = ref_1 > MAX_REFLEC
        cm_2 = ref_2 > MAX_REFLEC

        # do sift tracking (looking for image shifts)

        # do dense tracking (looking for plume motion)

        # mask samples

        # record

    # visualise





if __name__ == "__main__":
    main()