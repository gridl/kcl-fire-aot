import ftplib
import logging
import os
import time
from datetime import datetime


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


def get_filelist(ftp_conn, year, doy, product):

    product_path = 'allData/5001/' + product + '/'

    try:
        ftp_cd(ftp_conn, year, doy, product_path)
        product_list = get_files(ftp_conn)

        # find the right file
        product_list = [f.split(None, 8)[-1].lstrip() for f in product_list]
        return product_list, product_path

    except:
        logger.info('Could not access file list for DOY: ' + doy + " on attempt 1. Reattempting...")
        attempt = 1
        while True:
            try:
                ftp_conn = ftp_connect()
                ftp_cd(ftp_conn, year, doy, product_path)
                product_list = get_files(ftp_conn)

                # find the right file
                product_list = [f.split(None, 8)[-1].lstrip() for f in product_list]
                return product_list, product_path

            except:
                attempt += 1
                logger.info('Could not access file list for DOY: ' + doy + " on attempt " + str(attempt) +
                            " Reattempting...")
                time.sleep(5)
                if attempt == 10:
                    return str(), str()


def ftp_cd(ftp_conn, year, doy, directory):
    ftp_conn.cwd("/")
    ftp_conn.cwd(directory + year + '/')
    ftp_conn.cwd(doy)


def get_files(ftp_conn):
    file_list = []
    ftp_conn.retrlines("LIST", file_list.append)
    return file_list


def main():

    ftp_loc = 'nrt4.modaps.eosdis.nasa.gov'
    username = 'dnfisher'
    password = '&5ii5yHMtGX9'

    year = str(datetime.utcnow().year)
    doy = str(datetime.utcnow().timetuple().tm_yday)
    product = 'VNP02IMG_NRT'

    ftp_conn = ftp_connect(ftp_loc, username, password)
    product_list, product_path = get_filelist(ftp_conn, year, doy, product)

    for prod in product_list:
        print "ftp://" + ftp_loc + product_path + prod

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()