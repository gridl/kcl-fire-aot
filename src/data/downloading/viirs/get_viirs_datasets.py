# -*- coding: utf-8 -*-
import os
import logging

import ftplib
import time
from datetime import datetime

import numpy as np

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


def check_downloading_status(temp_path, mod_doy):
    # a small function to check if a goes file is being downloaded
    files_downloading = os.listdir(temp_path)
    if mod_doy + '.tmp' in files_downloading:
        return True
    else:
        return False


def append_to_download_list(temp_path, mod_doy):
    open(temp_path + mod_doy + '.tmp', 'a').close()


def get_filenames(ftp_laads, year, doy, directory, product, mxd021km_file):

    product_path = os.path.join(fp.path_to_all_data, directory, product, year, doy)

    try:
        ftp_cd(ftp_laads, product_path)
        file_list = get_files(ftp_laads)

        # find the right files
        viirs_file_list = [f.split(None, 8)[-1].lstrip() for f in file_list]
        viirs_file_times = [f.split(product)[-1] for f in viirs_file_list]

        # get times differences
        modis_time = datetime.strptime(mxd021km_file[10:22], "%Y%j.%H%M")
        viirs_times = [datetime.strptime(f[2:14], "%Y%j.%H%M") for f in viirs_file_times]
        product_files = []
        time_limit = 20 * 60
        for i, t in enumerate(viirs_times):
            abs_time_diff = np.abs((modis_time - t).total_seconds())
            if abs_time_diff < time_limit:
                product_files.append(viirs_file_list[i])
        return product_files, product_path

    except Exception, e:
        logger.info('Could not access file list for DOY: ' + doy)
        logger.warning(str(e))
        attempt = 1
        while True:
            try:
                ftp_laads = ftp_connect_laads()
                ftp_cd(ftp_laads, product_path)
                file_list = get_files(ftp_laads)

                # find the right files
                viirs_file_list = [f.split(None, 8)[-1].lstrip() for f in file_list]
                viirs_file_times = [f.split(product)[-1].lstrip() for f in viirs_file_list]

                # get times differences
                modis_time = datetime.strptime(mxd021km_file[10:22], "%Y%j.%H%M")
                viirs_times = [datetime.strptime(f[2:14], "%Y%j.%H%M") for f in viirs_file_times]
                product_files = []
                time_limit = 20 * 60
                for i, t in enumerate(viirs_times):
                    abs_time_diff = np.abs((modis_time - t).total_seconds())
                    if abs_time_diff < time_limit:
                        product_files.append(viirs_file_list[i])
                return product_files, product_path

            except:
                attempt += 1
                logger.info('Could not access file list for DOY: ' + doy + " on attempt " + str(attempt) +
                            " Reattempting...")
                time.sleep(5)
                if attempt == 10:
                    return str(), str()


def ftp_cd(ftp_laads, path):
    ftp_laads.cwd("/")
    ftp_laads.cwd(path)


def get_files(ftp_laads):
    file_list = []
    ftp_laads.retrlines("LIST", file_list.append)
    return file_list


def retrieve(ftp_laads, doy, path, local_filename, filename):
    # try accessing ftp, if fail then reconnect
    try:
        ftp_cd(ftp_laads, doy, path)

        lf = open(local_filename, "wb")
        ftp_laads.retrbinary("RETR " + filename, lf.write, 8 * 1024)
        lf.close()

    except:
        attempt = 1
        run = True
        while run:
            try:
                lf = open(local_filename, "wb")
                ftp_laads = ftp_connect_laads()
                ftp_cd(ftp_laads, doy, path)
                ftp_laads.retrbinary("RETR " + filename, lf.write, 8 * 1024)
                lf.close()
                ftp_laads.close()
                run = False
            except Exception, e:
                attempt += 1
                logger.info('Could not download ' + filename + " on attempt " + str(attempt) +
                            " Failed with error: "+ str(e) + "Reattempting...")
                time.sleep(5)


def main():

    file_list = os.path.join(fp.path_to_filelists, 'indonesia_filepaths_for_viirs_download.txt')
    ftp_laads = ftp_connect_laads()
    temp_path = fp.path_to_viirs_tmp

    with open(file_list, 'rb') as fl:
        mxd021km_filenames = fl.readlines()

        for mxd021km_file in mxd021km_filenames:

            mxd021km_file = mxd021km_file.rstrip()

            downloading = check_downloading_status(temp_path, mxd021km_file)
            if downloading:
                continue
            else:
                append_to_download_list(temp_path, mxd021km_file)

            # get ftp filename
            year = mxd021km_file[10:14]
            doy = mxd021km_file[14:17]
            VAOTIP_filenames, VAOTIP_filepath = get_filenames(ftp_laads, year, doy,
                                                              '3110', 'NPP_VAOTIP_L2', mxd021km_file)
            SRFL_filenames, SRFL_filepath = get_filenames(ftp_laads, year, doy,
                                                              '3110', 'NPP_SRFLMIP_L2', mxd021km_file)
            VGAERO_filenames, VGAERO_filepath = get_filenames(ftp_laads, year, doy,
                                                          '3110', 'NPP_VGAERO_L2', mxd021km_file)

            #VNP03_filenames, VNP03_filepath = get_filenames(ftp_laads,year,  doy, '5000', 'VNP03MODLL', mxd021km_file)
            #VNP09_filenames, VNP09_filepath = get_filenames(ftp_laads, year, doy, '5000', 'VNP09', mxd021km_file)
            #VNP04_filenames, VNP04_filepath = get_filenames(ftp_laads, year, doy, '5000', 'VNP04E_L2', mxd021km_file)

            continue

            filenames = [VAOTIP_filenames, VNP03_filenames, VNP09_filenames,VNP04_filenames]
            ftp_dirs = [VAOTIP_filepath, VNP03_filepath, VNP09_filepath, VNP04_filepath]
            local_dirs = [fp.path_to_viirs_ip_aod, fp.path_to_viirs_geo, fp.path_to_viirs_ref, fp.path_to_viirs_aod6k]

            for fnames, ftp_dir, local_dir in zip(filenames, ftp_dirs, local_dirs):

                if not os.path.exists(local_dir):
                    os.makedirs(local_dir)

                for fname in fnames:
                    local_fp = os.path.join(local_dir, fname)

                    if not os.path.isfile(local_fp):  # if we dont have the file, then dl it
                        logger.info("   ...Downloading: " + fname)
                        retrieve(ftp_laads, doy, ftp_dir, local_fp, fname)
                    else:
                        logger.info(fname + " ... already exists")


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()
