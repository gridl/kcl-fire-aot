import pandas as pd
import numpy as np

import src.data.readers as readers

# set up filepaths and similar

def main():

    # create df to hold the outputs
    output_df = pd.DataFrame()

    # read in non-plume specific files
    frp_data = readers.read_goes_frp()
    lc_data = readers.read_lc()

    # iterate over each plume in the plume mask dataframe
    orac_filename = ''
    plumes_masks = readers.read_plume_masks()
    for plume in plumes:

        if plumes.filename != orac_filename:
            orac_data = readers.read_orac()
            orac_filename = plumes.filename

        # open up plume specific data
        bg_masks = readers.read_bg_masks()

        # set up plumes mask (in line sample and geo coords)

        # get plumes AOD (using line sample, check plume manually and continue if unsuitable? Bow-tie effect, need to resample?)

        # get background AOD (using line sample, check background manually and continue if unsuitable?, Bow-tie effect, need to resample?)

        # get fires contained within plume (using geo coords and date time, if none then continue)

        # for fires get landsurface type

        # insert data into dataframe


