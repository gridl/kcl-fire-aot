# -*- coding: utf-8 -*-
import os
import logging
from dotenv import find_dotenv, load_dotenv

import ftplib
import time


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


def get_file(ftp_laads, doy, myd021km_file):
    try:
        ftp_cd(ftp_laads, doy, 'allData/6/MYD03/')
        file_list = get_files(ftp_laads)
    except:
        logger.info('Could not download data for DOY: ' + doy + " Reattempting...")
        attempt = 1
        while True:
            try:
                ftp_laads = ftp_connect_laads()
                ftp_cd(ftp_laads, doy, 'allData/6/MYD03/')
                file_list = get_files(ftp_laads)
            except:
                logger.info('Could not download data for DOY: ' + doy + " Reattempting...")
                time.sleep(5)
                attempt += 1

    # find the right file
    file_list = [f.split(None, 8)[-1].lstrip() for f in file_list]
    myd03_file = [f for f in file_list if myd021km_file[10:23] in f][0]
    return myd03_file


def get_files(ftp_laads):
    file_list = []
    ftp_laads.retrlines("LIST", file_list.append)
    return file_list


def retrieve_l1(ftp_laads, doy, local_filename, filename_l1):
    # try accessing ftp, if fail then reconnect
    try:
        ftp_cd(ftp_laads, doy, 'allData/6/MYD03/')
    except:
        ftp_laads = ftp_connect_laads()
        ftp_cd(ftp_laads, doy, 'allData/6/MYD03/')


    lf = open(local_filename, "wb")
    ftp_laads.retrbinary("RETR " + filename_l1, lf.write, 8 * 1024)
    lf.close()
    ftp_laads.close()


def ftp_cd(ftp_laads, doy, directory):
    ftp_laads.cwd("/")
    ftp_laads.cwd(directory + '2014' + '/')
    ftp_laads.cwd(doy)


def main():

    # first connect to ftp site
    ftp_laads = ftp_connect_laads()

    # get the files to download
    file_list = '/Users/dnf/git/kcl-fire-aot/data/raw/rsync_file_list/files_to_transfer.txt'
    with open(file_list, 'r') as f:
        files_to_get = f.read()

    files_to_get = files_to_get.split('\n')
    files_to_get = [f.split('/')[-1] for f in files_to_get]
    files_to_get = list(set(files_to_get))
    for f in files_to_get:

        if not f:
            continue

        logger.info("Downloading MODIS 03 data for file f: " + f)

        # find the correct myd03 file
        doy = f[14:17]
        myd03_filename = get_file(ftp_laads, doy, f)

        # download the file
        local_filename = os.path.join(r"../../data/raw/geo", myd03_filename)

        if not os.path.isfile(local_filename):  # if we dont have the file, then dl it
            logger.info("Downloading: " + myd03_filename)
            retrieve_l1(ftp_laads, doy, local_filename, myd03_filename)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()
