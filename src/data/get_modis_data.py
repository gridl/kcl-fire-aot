# -*- coding: utf-8 -*-
import os
import logging
from dotenv import find_dotenv, load_dotenv

import ftplib
import re

from pyhdf.SD import SD, SDC

import config


def ftp_connect_laads(doy, dir):
    ftp = ftplib.FTP("ladsweb.nascom.nasa.gov")
    ftp.login()
    ftp.cwd(dir + str(config.year) + '/')
    ftp.cwd(str(doy))
    return ftp


def get_files(ftp):
    file_list = []
    ftp.retrlines("LIST", file_list.append)
    return file_list


def assess_fire_pixels(doy, local_filename, filename_frp):

    # see if data exists in frp dir, if not then dl it
    if not os.path.isfile(local_filename):
        ftp_laads = ftp_connect_laads(doy, dir='allData/6/MYD14/')
        lf = open(local_filename, "wb")
        ftp_laads.retrbinary("RETR " + filename_frp, lf.write, 8 * 1024)
        lf.close()
        ftp_laads.close()

    frp_data = SD(local_filename, SDC.READ)
    fire_mask = frp_data.select('fire mask').get() >= 7

    # determine whether to process the scene or not
    if frp_data.select('FP_power').checkempty():
        process_flag = False
    else:
        power = frp_data.select('FP_power').get()
        total_fires = len(power)
        total_power = np.sum(power)
        if total_fires > config.min_fires and total_power > config.min_power:
            process_flag = True
        else:
            process_flag = False

    return fire_mask, process_flag


def retrieve_l1(doy, local_filename, filename_l1):
    ftp_laads = ftp_connect_laads(doy, dir='allData/6/MYD021KM/')
    lf = open(local_filename, "wb")
    ftp_laads.retrbinary("RETR " + filename_l1, lf.write, 8 * 1024)
    lf.close()
    ftp_laads.close()


def main():

    logger = logging.getLogger(__name__)
    logger.info('making data set from raw data')

    for doy in config.doy_range:

        # connect to ftp and move to correct doy
        ftp_laads = ftp_connect_laads(doy, dir='allData/6/MYD021KM/')
        l1_file_list = get_files(ftp_laads)
        ftp_laads.close()
        ftp_laads = ftp_connect_laads(doy, dir='allData/6/MYD14/')
        frp_file_list = get_files(ftp_laads)
        ftp_laads.close()

        # TODO need to ensure that the filenames are correctly associatied
        for f_l1, f_frp in zip(l1_file_list, frp_file_list):
            filename_l1 = f_l1.split(None, 8)[-1].lstrip()
            filename_frp = f_frp.split(None, 8)[-1].lstrip()

            time_stamp = re.search("[.][0-9]{4}[.]", filename_l1).group()
            if (int(time_stamp[1:-1]) < config.min_time) | \
               (int(time_stamp[1:-1]) > config.max_time):
                continue

            # asses is fire pixels in scene
            local_filename = os.path.join(r"../../data/raw/frp/", filename_frp)
            fire_mask, fires = assess_fire_pixels(doy, local_filename, filename_frp)

        if not fires:
            continue

        # download the file
        local_filename = os.path.join(r"../../data/raw/l1b", filename_l1)

        if not os.path.isfile(local_filename):  # if we dont have the file, then dl it
            retrieve_l1(doy, local_filename, filename_l1)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
