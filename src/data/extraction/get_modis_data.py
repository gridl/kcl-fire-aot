# -*- coding: utf-8 -*-
import ftplib
import logging
import os
import re
import time

import numpy as np
from dotenv import find_dotenv, load_dotenv
from pyhdf.SD import SD, SDC

import src.config.data as data_settings
import src.config.filepaths as filepaths


def ftp_connect_laads():
    try:
        ftp_laads = ftplib.FTP(filepaths.path_to_ladsweb_ftp)
        ftp_laads.login()
        return ftp_laads
    except:
        logger.info("Could not acces laadsweb - trying again")
        attempt = 1
        while True:
            try:
                ftp_laads = ftplib.FTP(filepaths.path_to_ladsweb_ftp)
                ftp_laads.login()
                logger.info("Accessed laadsweb on attempt: " + str(attempt))
                return ftp_laads
            except:
                logger.info("Could not access laadsweb - trying again")
                time.sleep(5)
                attempt += 1


def ftp_cd(ftp_laads, doy, directory):
    ftp_laads.cwd("/")
    ftp_laads.cwd(directory + data_settings.myd_year + '/')
    ftp_laads.cwd(doy)


def get_file_lists(ftp_laads, doy):
    try:
        ftp_cd(ftp_laads, doy, filepaths.path_to_myd021km)
        l1_file_list = get_files(ftp_laads)
        ftp_cd(ftp_laads, doy, filepaths.path_to_myd14)
        frp_file_list = get_files(ftp_laads)
        return l1_file_list, frp_file_list
    except:
        logger.info('Could not download data for DOY: ' + doy + " Reattempting...")
        attempt = 1
        while True:
            try:
                ftp_laads = ftp_connect_laads()
                ftp_cd(ftp_laads, doy, filepaths.path_to_myd021km)
                l1_file_list = get_files(ftp_laads)
                ftp_cd(ftp_laads, doy, filepaths.path_to_myd14)
                frp_file_list = get_files(ftp_laads)
                return l1_file_list, frp_file_list
            except:
                logger.info('Could not download data for DOY: ' + doy + " Reattempting...")
                time.sleep(5)
                attempt += 1


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

def assess_fires_present_inbounds(ftp_laads, doy, local_filename, filename_frp):

    # see if data exists in frp dir, if not then dl it
    if not os.path.isfile(local_filename):

        # try accessing ftp, if fail then reconnect
        try:
            ftp_cd(ftp_laads, doy, filepaths.path_to_myd14)
        except:
            ftp_laads = ftp_connect_laads()
            ftp_cd(ftp_laads, doy, filepaths.path_to_myd14)


        lf = open(local_filename, "wb")
        ftp_laads.retrbinary("RETR " + filename_frp, lf.write, 8 * 1024)
        lf.close()

    # determine whether to download the scene or not
    try:
        frp_data = SD(local_filename, SDC.READ)
    except:
        download_flag = False
        return download_flag

    if frp_data.select('FP_power').checkempty():
        download_flag = False
    else:

        lat = np.mean(frp_data.select('FP_latitude').get())
        lon = np.mean(frp_data.select('FP_longitude').get())

        if (lon, lat) in data_settings.geo_area_def:

            szn = frp_data.select('FP_SolZenAng').get()
            power = frp_data.select('FP_power').get()
            total_fires = len(power)
            total_power = np.sum(power)

            if np.mean(szn) > data_settings.myd_min_szn:
                download_flag = False
            elif total_fires > data_settings.myd_min_fires and total_power > data_settings.myd_min_power:
                logger.info('Suitable scene: ' + filename_frp)
                logger.info('Total fires: ' + str(total_fires)
                            + " | Total Power: " + str(total_power))
                download_flag = True
            else:
                download_flag = False
        else:
            download_flag = False

    return download_flag


def retrieve_l1(ftp_laads, doy, local_filename, filename_l1):
    # try accessing ftp, if fail then reconnect
    try:
        ftp_cd(ftp_laads, doy, filepaths.path_to_myd021km)
    except:
        ftp_laads = ftp_connect_laads()
        ftp_cd(ftp_laads, doy, filepaths.path_to_myd021km)


    lf = open(local_filename, "wb")
    ftp_laads.retrbinary("RETR " + filename_l1, lf.write, 8 * 1024)
    lf.close()
    ftp_laads.close()


def main():

    # first connect to ftp site
    ftp_laads = ftp_connect_laads()

    for doy in data_settings.myd_doy_range:

        doy = str(doy).zfill(3)

        logger.info("Downloading MODIS data with fires for DOY: " + doy)

        # get files lists from laads
        l1_file_list, frp_file_list = get_file_lists(ftp_laads, doy)

        l1_filenames = [f.split(None, 8)[-1].lstrip() for f in l1_file_list]
        frp_filenames = [f.split(None, 8)[-1].lstrip() for f in frp_file_list]

        # ensure file lists are matching
        l1_filenames, frp_filenames = matchup_files(l1_filenames, frp_filenames)

        for l1_filename, frp_filename in zip(l1_filenames, frp_filenames):

            # assess if fire pixels in scene
            local_filename = os.path.join(filepaths.path_to_modis_frp, frp_filename)
            fires_present = assess_fires_present_inbounds(ftp_laads, doy, local_filename, frp_filename)

            if not fires_present:
                continue

            # download the file
            local_filename = os.path.join(filepaths.path_to_modis_l1b, l1_filename)

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
