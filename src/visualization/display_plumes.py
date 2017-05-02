'''
This function reads in the hand plume digisations for a given
MODIS file.  It also reads in the ORAC main outputs.  It then
combines the mask and the output and plots the result to allow
the user to evaluate the various parameters.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import glob
from datetime import datetime

def main():


    # set up paths
    orac_data_path = '../../data/processed/orac_test_scene/'
    mask_path = '../../data/interim/myd021km_plumes_df.pickle'
    output = ''

    # read in the masks
    mask_df = pd.read_pickle(mask_path)

    # iterate over modis files
    for primary_file in glob.glob(orac_data_path + '*/*/*primary*'):

        # get the right masks and generate
        tt = datetime.strptime(primary_file.split('_')[-2], '%Y%m%d%H%M').timetuple()
        primary_datestring = str(tt.tm_year) + \
                             str(tt.tm_yday).zfill(3) + \
                             '.' + \
                             str(tt.tm_hour).zfill(2) + \
                             str(tt.tm_min).zfill(2)
        primary_mask_df = mask_df[mask_df['filename'].str.contains(primary_datestring)]


        # read in relevant params

        # apply mask

        # visualise and save output

        pass


if __name__ == '__main__':
    main()