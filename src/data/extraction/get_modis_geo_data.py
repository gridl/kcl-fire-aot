# -*- coding: utf-8 -*-
import os
import logging

import ftplib
import time

import src.config.filepaths as filepaths
import src.config.data as data_settings


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


def get_file(ftp_laads, doy, myd021km_file):
    try:
        ftp_cd(ftp_laads, doy, filepaths.path_to_myd03)
        file_list = get_files(ftp_laads)

        # find the right file
        file_list = [f.split(None, 8)[-1].lstrip() for f in file_list]
        myd03_file = [f for f in file_list if myd021km_file[10:23] in f][0]
        return myd03_file

    except:
        logger.info('Could not access file list for DOY: ' + doy + " on attempt 1. Reattempting...")
        attempt = 1
        while True:
            try:
                ftp_laads = ftp_connect_laads()
                ftp_cd(ftp_laads, doy, filepaths.path_to_myd03)
                file_list = get_files(ftp_laads)

                # find the right file
                file_list = [f.split(None, 8)[-1].lstrip() for f in file_list]
                myd03_file = [f for f in file_list if myd021km_file[10:23] in f][0]
                return myd03_file
            except:
                attempt += 1
                logger.info('Could not access file list for DOY: ' + doy + " on attempt " + str(attempt) +
                            " Reattempting...")
                time.sleep(5)
                if attempt == 10:
                    return str()


def get_files(ftp_laads):
    file_list = []
    ftp_laads.retrlines("LIST", file_list.append)
    return file_list


def retrieve_l1(ftp_laads, doy, local_filename, filename_l1):
    # try accessing ftp, if fail then reconnect
    try:
        ftp_cd(ftp_laads, doy, filepaths.path_to_myd03)
    except:
        ftp_laads = ftp_connect_laads()
        ftp_cd(ftp_laads, doy, filepaths.path_to_myd03)


    lf = open(local_filename, "wb")
    ftp_laads.retrbinary("RETR " + filename_l1, lf.write, 8 * 1024)
    lf.close()
    ftp_laads.close()


def ftp_cd(ftp_laads, doy, directory):
    ftp_laads.cwd("/")
    ftp_laads.cwd(directory + data_settings.myd_year + '/')
    ftp_laads.cwd(doy)


def check_downloading_status(temp_path, f):
    # a small function to check if a goes file is being downloaded
    files_downloading = os.listdir(temp_path)
    if f in files_downloading:
        return True
    else:
        return False


def append_to_download_list(temp_path, f):
    open(temp_path + f, 'a').close()


def remove_from_download_list(temp_path, f):
    os.remove(temp_path + f)


def main():

    temp_path = filepaths.path_to_modis_tmp

    # first connect to ftp site
    ftp_laads = ftp_connect_laads()

    # get the files to download
    for f in os.listdir(filepaths.path_to_modis_l1b):

        if not f:
            continue

        # check if year we are working on is in the file, if not move on
        if data_settings.myd_year not in f[0:16]:
            continue

        # check if the file is being downloaded by another script already
        downloading = check_downloading_status(temp_path, f)
        if downloading:
            continue
        else:
            append_to_download_list(temp_path, f)


        # find the correct myd03 file
        doy = f[14:17]
        if not doy:
            continue
        myd03_filename = get_file(ftp_laads, doy, f)

        if not myd03_filename:
            logger.warning('Could not download geo file for file: ' + f)
            continue

        # download the file
        local_filename = os.path.join(filepaths.path_to_modis_geo, myd03_filename)

        if not os.path.isfile(local_filename):  # if we dont have the file, then dl it
            logger.info("Downloading: " + myd03_filename)
            retrieve_l1(ftp_laads, doy, local_filename, myd03_filename)
        else:
            logger.info(myd03_filename + ' already exists on the system')

        # remote temp empty file from currently downloading list
        remove_from_download_list(temp_path, f)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()
