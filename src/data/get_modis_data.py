# -*- coding: utf-8 -*-
import os
import logging
from dotenv import find_dotenv, load_dotenv

import ftplib
import re
import time

from pyhdf.SD import SD, SDC
import numpy as np

import config


def ftp_connect_laads():
    try:
        ftp_laads = ftplib.FTP("ladsweb.nascom.nasa.gov")
        ftp_laads.login()
        return ftp_laads
    except:
        logger.info("Could not acces laadsweb - trying again")
        attempt = 1
        while True:
            try:
                ftp_laads = ftplib.FTP("ladsweb.nascom.nasa.gov")
                ftp_laads.login()
                logger.info("Accessed laadsweb on attempt: " + str(attempt))
                return ftp_laads
            except:
                logger.info("Could not access laadsweb - trying again")
                time.sleep(5)
                attempt += 1


def ftp_cd(ftp_laads, doy, directory):
    ftp_laads.cwd("/")
    ftp_laads.cwd(directory + str(config.myd['year']) + '/')
    ftp_laads.cwd(str(doy))


def get_files(ftp_laads):
    file_list = []
    ftp_laads.retrlines("LIST", file_list.append)
    return file_list


def matchup_files(l1_filenames, frp_filenames):

    l1_times = [re.search("[.][0-9]{4}[.]", f).group() for f in l1_filenames]
    frp_times = [re.search("[.][0-9]{4}[.]", f).group() for f in frp_filenames]
    common_times = list(set.intersection(set(l1_times), set(frp_times)))

    l1_matching = [f for f in l1_filenames if re.search("[.][0-9]{4}[.]", f).group() in common_times]
    frp_matching = [f for f in frp_filenames if re.search("[.][0-9]{4}[.]", f).group() in common_times]

    if len(l1_times) != len(frp_times):
        logger.warning("Incomplete number of granules for one product: "
                       + str(len(l1_times)) + " FRP granules vs. "
                       + str(len(frp_times)) + "L1 granules")
        logger.warning("Granules have been time matched for processing")

    return l1_matching, frp_matching

def assess_fires_present(ftp_laads, doy, local_filename, filename_frp):

    # see if data exists in frp dir, if not then dl it
    if not os.path.isfile(local_filename):

        # try accessing ftp, if fail then reconnect
        try:
            ftp_cd(ftp_laads, doy, 'allData/6/MYD14/')
        except:
            ftp_laads = ftp_connect_laads()
            ftp_cd(ftp_laads, doy, 'allData/6/MYD14/')


        lf = open(local_filename, "wb")
        ftp_laads.retrbinary("RETR " + filename_frp, lf.write, 8 * 1024)
        lf.close()

    # determine whether to process the scene or not
    frp_data = SD(local_filename, SDC.READ)
    if frp_data.select('FP_power').checkempty():
        process_flag = False
    else:
        szn = frp_data.select('FP_SolZenAng').get()
        power = frp_data.select('FP_power').get()
        total_fires = len(power)
        total_power = np.sum(power)

        if np.mean(szn) > 85:
            process_flag = False
        elif total_fires > config.myd['min_fires'] and total_power > config.myd['min_power']:
            logger.info('Suitable scene: ' + filename_frp)
            logger.info('Total fires: ' + str(total_fires)
                        + " | Total Power: " + str(total_power))
            process_flag = True
        else:
            process_flag = False

    return process_flag


def retrieve_l1(ftp_laads, doy, local_filename, filename_l1):
    # try accessing ftp, if fail then reconnect
    try:
        ftp_cd(ftp_laads, doy, 'allData/6/MYD021KM/')
    except:
        ftp_laads = ftp_connect_laads()
        ftp_cd(ftp_laads, doy, 'allData/6/MYD021KM/')


    lf = open(local_filename, "wb")
    ftp_laads.retrbinary("RETR " + filename_l1, lf.write, 8 * 1024)
    lf.close()
    ftp_laads.close()


def main():

    # first connect to ftp site
    ftp_laads = ftp_connect_laads()

    for doy in config.myd['doy_range']:

        logger.info("Downloading MODIS data with fires for DOY: " + str(doy))

        # get files lists from laads
        ftp_cd(ftp_laads, doy, 'allData/6/MYD021KM/')
        l1_file_list = get_files(ftp_laads)
        ftp_cd(ftp_laads, doy, 'allData/6/MYD14/')
        frp_file_list = get_files(ftp_laads)

        l1_filenames = [f.split(None, 8)[-1].lstrip() for f in l1_file_list]
        frp_filenames = [f.split(None, 8)[-1].lstrip() for f in frp_file_list]

        # ensure file lists are matching
        l1_filenames, frp_filenames = matchup_files(l1_filenames, frp_filenames)

        for l1_filename, frp_filename in zip(l1_filenames, frp_filenames):

            time_stamp = re.search("[.][0-9]{4}[.]", l1_filename).group()
            if (int(time_stamp[1:-1]) < config.myd['min_time']) | \
               (int(time_stamp[1:-1]) > config.myd['max_time']):
                continue

            # asses is fire pixels in scene
            local_filename = os.path.join(r"../../data/raw/frp/", frp_filename)
            fires_present = assess_fires_present(ftp_laads, doy, local_filename, frp_filename)

            if not fires_present:
                continue

            # download the file
            local_filename = os.path.join(r"../../data/raw/l1b", l1_filename)

            if not os.path.isfile(local_filename):  # if we dont have the file, then dl it
                logger.info("Downloading: " + l1_filename)
                retrieve_l1(ftp_laads, doy, local_filename, l1_filename)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
