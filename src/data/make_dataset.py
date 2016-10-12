# -*- coding: utf-8 -*-
import os
import sys
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

import config


def ftp_connect():
    ftp = ftplib.FTP("ladsweb.nascom.nasa.gov")
    ftp.login()
    ftp.cwd('allData/6/MYD021KM/' + str(config.year) + '/')
    return ftp


def get_files(ftp, doy):
    file_list = []
    ftp.retrlines("LIST", file_list.append)
    return file_list


def get_mod_url(doy, time_stamp):
    date = datetime.datetime(config.year, 1, 1) + datetime.timedelta(doy - 1)
    date = date.strftime("%Y_%m_%d")
    mod_url = "http://modis-atmos.gsfc.nasa.gov/IMAGES/MYD02/GRANULE/{0}/{1}rgb143.jpg".format(date,
                                                                                           str(doy) +
                                                                                           time_stamp)
    return mod_url


def get_image(mod_url):
    r = requests.get(mod_url, stream=True)
    if r.status_code == 200:
        with open('current.jpg', 'wb') as fname:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, fname)
    return r.status_code


def display_image():
    im = ndimage.imread('current.jpg', mode="RGB")
    plt.figure(figsize=(16, 16))
    plt.imshow(im)
    plt.draw()
    plt.pause(1)  # <-------

    # raw_input returns the empty string for "enter"
    yes = set(['yes', 'y', 'ye', ''])
    no = set(['no', 'n'])

    choice = raw_input("Process image: [y,n]").lower()
    if choice in yes:
        plt.close()
        return True
    elif choice in no:
        plt.close()
        return False
    else:
        sys.stdout.write("Please respond with 'yes' or 'no'")


def main():
    """ Loads MODIS data files for the specified geographic region,
        timeframe, and time stamps.  The user can then process these
        images and extract smoke plumes.
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    ftp = ftp_connect()

    for doy in config.doy_range:

        # change into the correct ftp dir
        ftp.cwd(str(doy))

        file_list = get_files(ftp, doy)
        for f in file_list:

            filename = f.split(None, 8)[-1].lstrip()

            time_stamp = re.search("[.][0-9]{4}[.]", filename).group()
            if int(time_stamp[1:-1]) < config.min_time:
                continue

            mod_url = get_mod_url(doy, time_stamp)
            status = get_image(mod_url)
            if status != 200:  # no access then continue
                continue
            process_flag = display_image()

            if process_flag:

                print os.getcwd()

                # save the png quicklook

                # download the file
                local_filename = os.path.join(r"../../data/raw/l1b", filename)
                lf = open(local_filename, "wb")
                logger.info('downloading MODIS file', filename)
                ftp.retrbinary("RETR " + filename, lf.write, 8*1024)
                lf.close()

                # do the digitising

        # change back the ftp dir
        ftp.cwd('..')







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
