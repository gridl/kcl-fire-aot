# -*- coding: utf-8 -*-
import os
import logging
from dotenv import find_dotenv, load_dotenv

import numpy as np
import ftplib
import datetime
import re

import requests
import shutil

import scipy.ndimage as ndimage
import matplotlib.pyplot as plt


def main():
    """ Loads MODIS data files for the specified geographic region,
        timeframe, and time stamps.  The user can then process these
        images and extract smoke plumes.
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    year = 2012
    doy_range = np.arange(267, 300, 1)
    min_time = 1600

    ftp = ftplib.FTP("ladsweb.nascom.nasa.gov")
    ftp.login()
    ftp.cwd('allData/6/MYD021KM/' + str(year) + '/')

    for doy in doy_range:

        ftp.cwd(str(doy))
        file_list = []
        ftp.retrlines("LIST", file_list.append)

        ftp.cwd('..')

        for f in file_list:
            time_stamp = re.search("[.][0-9]{4}[.]", f).group()

            if int(time_stamp[1:-1]) < min_time:
                continue

            date = datetime.datetime(year, 1, 1) + datetime.timedelta(doy - 1)
            date = date.strftime("%Y_%m_%d")
            mod_url = "http://modis-atmos.gsfc.nasa.gov/IMAGES/MYD02/GRANULE/{0}/{1}rgb143.jpg".format(date,
                                                                                                       str(doy) +
                                                                                                       time_stamp)
            r = requests.get(mod_url, stream=True)
            if r.status_code == 200:

                with open('test.jpg', 'wb') as fname:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, fname)

                im = ndimage.imread('test.jpg', mode="RGB")
                plt.figure(figsize=(18,18))
                plt.imshow(im)
                plt.show()



                    # if desiredata features in image then load data
    #ftp://ladsweb.nascom.nasa.gov/allData/6/MYD021KM/2011/267/

    # perform manual feature extraction

    # store in database

    # close off data



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
