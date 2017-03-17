# -*- coding: utf-8 -*-
import os
import logging

import ftplib
import time
import urllib



def ftp_connect_class():
    try:
        ftp_class = ftplib.FTP("ftp.class.ncdc.noaa.gov ")
        ftp_class.login()
        return ftp_class
    except:
        logger.info("Could not access class - trying again")
        attempt = 1
        while True:
            try:
                ftp_class = ftplib.FTP("ftp.class.ncdc.noaa.gov")
                ftp_class.login()
                logger.info("Accessed class on attempt: " + str(attempt))
                return ftp_class
            except:
                logger.info("Could not access class - trying again")
                time.sleep(5)
                attempt += 1


def ftp_cd(ftp_class, directory, tag):
    ftp_class.cwd(directory + '/' + tag + '/')


def get_files(ftp_class):
    file_list = []
    ftp_class.retrlines("LIST", file_list.append)
    return file_list


def get_file_lists(ftp_laads, file_id):
    try:
        ftp_cd(ftp_laads, file_id, '001')
        file_list = get_files(ftp_laads)
        return file_list
    except:
        logger.info('Could not access data for FID ' + file_id + " Reattempting...")
        attempt = 1
        while True:
            try:
                ftp_laads = ftp_connect_class()
                ftp_cd(ftp_laads, file_id, '001')
                file_list = get_files(ftp_laads)
                return file_list
            except:
                logger.info('Could not access data for FID ' + file_id + " Reattempting...")
                time.sleep(5)
                attempt += 1


def retrieve_l1(ftp_class, order_id, local_filename, filename):
    # now lets have a go using https to see if that will work for us
    source = 'https://download.class.ncdc.noaa.gov/download/'
    order = order_id + '/001/'
    urllib.urlretrieve(source + order + filename, local_filename)


def check_downloading_status(goes_file):
    # a small function to check if a goes file is being downloaded
    files_downloading = os.listdir(r"../../data/tmp/goes/")
    if goes_file in files_downloading:
        return True
    else:
        return False


def append_to_download_list(goes_file):
    files_downloading = r"../../data/tmp/goes/"
    open(files_downloading+goes_file, 'a').close()


def remove_from_download_list(goes_file):
    files_downloading = r"../../data/tmp/goes/"
    os.remove(files_downloading+goes_file)


def main():

    # first connect to ftp site
    ftp_class = ftp_connect_class()

    # order id's
    order_ids = ['2720282193', '2720283243', '2720283253', '2720285893',
                 '2720285903', '2720285923', '2720285913', '2720285973',
                 '2720285983', '2720286013', '2720283233', '2720283263',
                 '2720283273', '2720284513', '2720285883', '2720285933',
                 '2720285943', '2720285953', '2720285963', '2720285993',
                 '2720286003', '2720284213', '2720286023']

    for order_id in order_ids:

        logger.info("Downloading GOES files for order: " + order_id)

        # get files lists from class
        file_list = get_file_lists(ftp_class, order_id)

        for goes_file in file_list:

            goes_file = goes_file.split(' ')[-1]

            # download the file
            local_filename = os.path.join(r"../../data/raw/goes", goes_file)

            # check if goes file is being downloaded elsewhere
            downloading = check_downloading_status(goes_file)

            if (not os.path.isfile(local_filename)) & (not downloading):
                # add temp empty file to currently downloading list
                append_to_download_list(goes_file)

                # do the download
                logger.info("Downloading: " + goes_file)
                retrieve_l1(ftp_class, order_id, local_filename, goes_file)

                # remote temp empty file from currently downloading list
                remove_from_download_list(goes_file)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables

    main()
