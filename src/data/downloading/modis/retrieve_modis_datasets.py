# -*- coding: utf-8 -*-

'''
For a given filelist, extract all the relevant modis files.
'''
import ftplib
import logging
import os
import sys
import re
import time

import numpy as np
from pyhdf.SD import SD, SDC

import src.config.data as data_settings
import src.config.filepaths as fp


def ftp_connect_laads():
    try:
        ftp_laads = ftplib.FTP(fp.path_to_ladsweb_ftp)
        ftp_laads.login()
        return ftp_laads
    except:
        logger.info("Could not acces laadsweb - trying again")
        attempt = 1
        while True:
            try:
                ftp_laads = ftplib.FTP(fp.path_to_ladsweb_ftp)
                ftp_laads.login()
                logger.info("Accessed laadsweb on attempt: " + str(attempt))
                return ftp_laads
            except:
                logger.info("Could not access laadsweb - trying again")
                time.sleep(5)
                attempt += 1


def get_files(ftp_laads):
    file_list = []
    ftp_laads.retrlines("LIST", file_list.append)
    return file_list


def get_filename(ftp_laads, doy, path, myd021km_file):
    try:
        ftp_cd(ftp_laads, doy, path)
        file_list = get_files(ftp_laads)

        # find the right file
        file_list = [f.split(None, 8)[-1].lstrip() for f in file_list]
        myd04_file = [f for f in file_list if myd021km_file[10:23] in f][0]
        return myd04_file

    except:
        logger.info('Could not access file list for DOY: ' + doy + " on attempt 1. Reattempting...")
        attempt = 1
        while True:
            try:
                ftp_laads = ftp_connect_laads()
                ftp_cd(ftp_laads, doy, fp.path_to_myd04)
                file_list = get_files(ftp_laads)

                # find the right file
                file_list = [f.split(None, 8)[-1].lstrip() for f in file_list]
                myd04_file = [f for f in file_list if myd021km_file[10:23] in f][0]
                return myd04_file
            except:
                attempt += 1
                logger.info('Could not access file list for DOY: ' + doy + " on attempt " + str(attempt) +
                            " Reattempting...")
                time.sleep(5)
                if attempt == 10:
                    return str()


def ftp_cd(ftp_laads, doy, directory):
    ftp_laads.cwd("/")
    ftp_laads.cwd(directory + data_settings.myd_year + '/')
    ftp_laads.cwd(doy)


def retrieve(ftp_laads, doy, path, local_filename, filename):
    # try accessing ftp, if fail then reconnect
    try:
        ftp_cd(ftp_laads, doy, path)
    except:
        ftp_laads = ftp_connect_laads()
        ftp_cd(ftp_laads, doy, path)

    lf = open(local_filename, "wb")
    ftp_laads.retrbinary("RETR " + filename, lf.write, 8 * 1024)
    lf.close()
    ftp_laads.close()


def check_downloading_status(temp_path, mod_doy):
    # a small function to check if a goes file is being downloaded
    files_downloading = os.listdir(temp_path)
    if mod_doy + '.tmp' in files_downloading:
        return True
    else:
        return False


def append_to_download_list(temp_path, mod_doy):
    open(temp_path + mod_doy + '.tmp', 'a').close()


def main():
    file_list = '/Users/dnf/Projects/kcl-fire-aot/data/Asia/filelists/indonesia_filepaths.txt' #sys.argv[1]
    ftp_laads = ftp_connect_laads()
    temp_path = fp.path_to_modis_tmp

    with open(file_list, 'rb') as fl:
        myd021km_filenames = fl.readlines()

    for myd021km_filename in myd021km_filenames:

        myd021km_filename = myd021km_filename.rstrip()

        downloading = check_downloading_status(temp_path, myd021km_filename)
        if downloading:
            continue
        else:
            append_to_download_list(temp_path, myd021km_filename)

        # get ftp filename
        doy = myd021km_filename[14:17]
        myd14_filename = get_filename(ftp_laads, doy, fp.path_to_myd14, myd021km_filename)
        myd03_filename = get_filename(ftp_laads, doy, fp.path_to_myd03, myd021km_filename)
        myd04_filename = get_filename(ftp_laads, doy, fp.path_to_myd04, myd021km_filename)

        # pull them down
        logger.info("Getting products for L1B file " + myd021km_filename + "...")

        filenames = [myd021km_filename, myd14_filename, myd03_filename, myd04_filename]
        ftp_dirs = [fp.path_to_myd021km, fp.path_to_myd14, fp.path_to_myd03, fp.path_to_myd04]
        local_dirs = [fp.path_to_modis_l1b, fp.path_to_modis_frp, fp.path_to_modis_geo, fp.path_to_modis_aod]

        for fname, ftp_dir, local_dir in zip(filenames, ftp_dirs, local_dirs):
            # check if local dir exists if not make it
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            local_fp = os.path.join(local_dir, fname)

            if not os.path.isfile(local_fp):  # if we dont have the file, then dl it
                logger.info("   ...Downloading: " + fname)
                retrieve(ftp_laads, doy, ftp_dir, local_fp, fname)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()
