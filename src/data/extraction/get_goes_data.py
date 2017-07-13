# -*- coding: utf-8 -*-

'''
Script to download goes files from the class https site.
To download an order of files just change the list containing the
file orders on line 88.  Also set the download and temporary
directory to something that is suitable.

'''

import os
import logging

import urllib
import urllib2
from bs4 import BeautifulSoup
import re

import src.config.filepaths as filepaths
import src.config.data as data_settings


def get_file_list(order_id):
    '''
    Finds the list4 of files from the https site.

    :param order_id: the oder tag to look under
    :return: list of goes files on the https site
    '''
    file_list = []
    source = filepaths.path_to_class_https
    order = order_id + '/001/'
    html_page = urllib2.urlopen(source+order)
    soup = BeautifulSoup(html_page, "lxml")
    for link in soup.findAll('a', attrs={'href': re.compile("^001/goes13")}):
        file_list.append(link.get('href')[4:])  # 4: to get rid of the 001/ at the start
    return file_list


def retrieve_l1(order_id, local_filename, filename):
    '''
    Downloads the given file

    :param order_id: order tag on the url
    :param local_filename: the name that the file will download to
    :param filename: the name of the file to be downloaded
    :return: nothing
    '''
    source = filepaths.path_to_class_https
    order = order_id + '/001/'
    urllib.urlretrieve(source + order + filename, local_filename)


def check_downloading_status(temp_path, goes_file):
    # a small function to check if a goes file is being downloaded
    files_downloading = os.listdir(temp_path)
    if goes_file in files_downloading:
        return True
    else:
        return False


def append_to_download_list(temp_path, goes_file):
    '''
    Tracks which file is being downloaded by storing
    the file name as an empty file in the temporary
    folder

    :param temp_path: path to hold temporary file
    :param goes_file: the file being downloaded
    :return: nothing
    '''
    open(temp_path + goes_file, 'a').close()


def remove_from_download_list(temp_path, goes_file):
    '''
    Removes the empty file with the goes_file name from
    the temporary folder

    :param temp_path: path to hold the temporary file
    :param goes_file: the file being downloaded
    :return: nothing
    '''
    os.remove(temp_path + goes_file)


def main():

    # path to write to
    data_store_path = filepaths.path_to_goes_l1b
    temp_path = filepaths.path_to_goes_tmp  # nothings gets stored here, just keeps track of what file is being dwnldd

    for order_id in data_settings.class_order_ids:

        logger.info("Downloading GOES files for order: " + order_id)

        # get files lists from class
        file_list = get_file_list(order_id)

        for goes_file in file_list:

            # download the file
            local_filename = os.path.join(data_store_path, goes_file)

            # check if goes file is being downloaded elsewhere
            downloading = check_downloading_status(temp_path, goes_file)

            if (not os.path.isfile(local_filename)) & (not downloading):
                # add temp empty file to currently downloading list
                append_to_download_list(temp_path, goes_file)

                # do the download
                logger.info("Downloading: " + goes_file)
                try:
                    retrieve_l1(order_id, local_filename, goes_file)
                except Exception, e:
                    logger.warning("Failed to download file: " + goes_file, 'with warning:', str(e))
                    logger.info("Second download attempt: " + goes_file)

                # remove temp empty file from currently downloading list
                remove_from_download_list(temp_path, goes_file)

            else:
                logger.info("The following GOES file already on system: " + goes_file)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()