# -*- coding: utf-8 -*-
import os
import logging

import urllib
import urllib2
from BeautifulSoup import BeautifulSoup
import re




def get_file_list(order_id):
    file_list = []
    source = 'https://download.class.ncdc.noaa.gov/download/'
    order = order_id + '/001/'
    html_page = urllib2.urlopen(source+order)
    soup = BeautifulSoup(html_page)
    for link in soup.findAll('a', attrs={'href': re.compile("^001/goes13")}):
        file_list.append(link.get('href')[4:])  # 4: to get rid of the 001/ at the start
    return file_list


def retrieve_l1(order_id, local_filename, filename):
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
        file_list = get_file_list(order_id)

        for goes_file in file_list:

            # download the file
            local_filename = os.path.join(r"../../data/raw/goes", goes_file)

            # check if goes file is being downloaded elsewhere
            downloading = check_downloading_status(goes_file)

            if (not os.path.isfile(local_filename)) & (not downloading):
                # add temp empty file to currently downloading list
                append_to_download_list(goes_file)

                # do the download
                logger.info("Downloading: " + goes_file)
                retrieve_l1(order_id, local_filename, goes_file)

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
