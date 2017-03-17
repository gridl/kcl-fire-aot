# -*- coding: utf-8 -*-
import os
import logging
from dotenv import find_dotenv, load_dotenv

import ftplib
import time



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


def retrieve_l1(ftp_class, file_id, local_filename, filename):
    # try accessing ftp, if fail then reconnect
    lf = open(local_filename, "wb")
    ftp_class.retrbinary("RETR " + filename, lf.write, 8 * 1024)
    lf.close()


def main():

    # first connect to ftp site
    ftp_class = ftp_connect_class()

    # order id's
    order_id = ['2720282193', '2720283243', '2720283253', '2720285893',
                '2720285903', '2720285923', '2720285913', '2720285973',
                '2720285983', '2720286013']

    for file_id in order_id:

        logger.info("Downloading GOES files for order: " + file_id)

        # get files lists from class
        file_list = get_file_lists(ftp_class, file_id)

        for goes_file in file_list:

            try:

                goes_file = goes_file.split(' ')[-1]

                # download the file
                local_filename = os.path.join(r"../../data/raw/goes", goes_file)

                if not os.path.isfile(local_filename):  # if we dont have the file, then dl it
                    logger.info("Downloading: " + goes_file)
                    retrieve_l1(ftp_class , file_id, local_filename, goes_file)
            except Exception, e:
                print 'could not open goes file:', goes_file, ' - with error:', e
                continue


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