"""
See: https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/lws-classic/api.php
for API commands

"""

import urllib
import ftplib
from xml.etree.ElementTree import parse  # py2.7 syntax
from datetime import datetime
import os
import time

import logging


def generic_request(method, base_url, **kargs):
    ret_list = []
    url = base_url + method
    if len(kargs) > 0:
        url += "?"

        for key, value in kargs.items():
            url += key + "=" + str(value) + "&"  # Extra & on the end doesn't seem to hurt anything

    root = parse(urllib.urlopen(url)).getroot()

    for element in root:
        ret_list.append(element.text)

    return ret_list


def search_for_files(base_url,
                     product="MOD021KM",
                     collection="6",
                     start=datetime.utcnow().strftime("%Y-%m-%d"),
                     stop=datetime.utcnow().strftime("%Y-%m-%d"),
                     north="90", south="-90",
                     west="-180", east="180",
                     coordsOrTiles="coords",
                     dayNightBoth="DNB"):
    r_dict = {"product": product,
              "collection": collection,
              "start": start,
              "stop": stop,
              "north": north,
              "south": south,
              "east": east,
              "west": west,
              "coordsOrTiles": coordsOrTiles,
              "dayNightBoth": dayNightBoth}

    return generic_request("searchForFiles", base_url, **r_dict)


def get_file_urls(base_url, files):
    assert (len(files) > 0)
    file_id_str = ",".join(files)
    return generic_request("getFileUrls", base_url, fileIds=file_id_str)


def ftp_connect(ftp_loc, username, password):
    try:
        ftp_conn = ftplib.FTP(ftp_loc)
        ftp_conn.login(username, password)
        return ftp_conn
    except Exception, e:
        logger.info("Could not access ftp with error " + str(e))
        attempt = 1
        while True:
            try:
                ftp_laads = ftplib.FTP(ftp_loc)
                ftp_laads.login()
                logger.info("Accessed ftp on attempt: " + str(attempt))
                return ftp_laads
            except:
                logger.info("Could not access laadsweb - trying again")
                time.sleep(5)
                attempt += 1


def retrieve(ftp_loc, username, password, year, doy, ftp_directory, ftp_filename, local_filepath):
    attempt = 0
    run = True
    while run:
        try:
            lf = open(local_filepath, "wb")
            ftp_conn = ftp_connect(ftp_loc, username, password)
            ftp_cd(ftp_conn, year, doy, ftp_directory)
            ftp_conn.retrbinary("RETR " + ftp_filename, lf.write, 8 * 1024)
            lf.close()
            ftp_conn.close()
            run = False
        except Exception, e:
            attempt += 1
            logger.info('Could not download ' + ftp_filename + " on attempt " + str(attempt) +
                        " Failed with error: " + str(e) + "Reattempting...")
            time.sleep(5)


def ftp_cd(ftp_conn, year, doy, ftp_product_path):
    ftp_conn.cwd("/")
    ftp_conn.cwd(os.path.join(ftp_product_path, year, doy))


def main():

    # ftp url
    base_url = "http://lance.modaps.eosdis.nasa.gov/axis2/services/MWSLance/"  # lance

    # output filepaths
    output_root = '/Users/danielfisher/Projects/kcl-fire-aot/data/nrt_test/modis'

    # set up ftp stuff
    ftp_loc = 'nrt3.modaps.eosdis.nasa.gov'
    username = 'dnfisher'
    password = '&5ii5yHMtGX9'
    ftp_root = 'allData/61/'
    ftp_product = "MOD021KM"
    ftp_directory = os.path.join(ftp_root, ftp_product)

    # roi
    roi_lat = 57.44
    roi_lon = 15.16
    rad = 0.01

    while True:

        # get current UTC time fo correct folder access
        year = str(datetime.utcnow().year)
        doy = str(datetime.utcnow().timetuple().tm_yday)

        # setup output path for doy
        output_directory = os.path.join(output_root, year, doy)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

            # also setup log file
            try:
                logger.handlers[0].stream.close()
                logger.removeHandler(logger.handlers[0])
            except:
                pass

            file_handler = logging.FileHandler(os.path.join(output_directory, 'doy.log'), 'a')
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s %(filename)s, %(lineno)d, %(funcName)s: %(message)s")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        files_roi = search_for_files(base_url,
                                     product=ftp_product,
                                     start=datetime.utcnow().strftime("%Y-%m-%d"),
                                     stop=datetime.utcnow().strftime("%Y-%m-%d"),
                                     north=roi_lat+rad, south=roi_lat-rad, west=roi_lon-rad, east=roi_lon+rad)
        if files_roi[0] == 'No results':
            continue
        urls_roi = get_file_urls(base_url, files_roi)

        for url in urls_roi:
            ftp_filename = url.split("/")[-1]
            local_filepath = os.path.join(output_directory, ftp_filename)
            if not os.path.isfile(local_filepath):
                retrieve(ftp_loc, username, password, year, doy, ftp_directory, ftp_filename, local_filepath)

                # log the time difference between the current time and the product time
                current_time = datetime.utcnow()
                product_time = datetime.strptime(ftp_filename[10:22], "%Y%j.%H%M")
                logger.info('Downloaded product:' + ftp_filename)
                logger.info('Products final download time: ' + str(current_time))
                logger.info('Products aquisition time: ' + str(product_time))
                logger.info('Products time diff.: ' + str(current_time - product_time))
                logger.info('')

        time.sleep(180)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()



