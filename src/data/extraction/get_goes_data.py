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
from BeautifulSoup import BeautifulSoup
import re


def get_file_list(order_id):
    '''
    Finds the list4 of files from the https site.

    :param order_id: the oder tag to look under
    :return: list of goes files on the https site
    '''
    file_list = []
    source = 'https://download.class.ncdc.noaa.gov/download/'
    order = order_id + '/001/'
    html_page = urllib2.urlopen(source+order)
    soup = BeautifulSoup(html_page)
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
    source = 'https://download.class.ncdc.noaa.gov/download/'
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

    # order id's
    order_ids = ['2720282193', '2720283243', '2720283253', '2720285893',
                 '2720285903', '2720285923', '2720285913', '2720285973',
                 '2720285983', '2720286013', '2720283233', '2720283263',
                 '2720283273', '2720284513', '2720285883', '2720285933',
                 '2720285943', '2720285953', '2720285963', '2720285993',
                 '2720286003', '2720284213', '2720286023']

    # path to write to
    data_store_path = r"../../data/raw/goes"  # data gets stored in here
    temp_path = r"../../data/tmp/goes/"  # nothings gets stored here, just keeps track of what file is being dwnldrd

    for order_id in order_ids:

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
                retrieve_l1(order_id, local_filename, goes_file)

                # remote temp empty file from currently downloading list
                remove_from_download_list(temp_path, goes_file)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()