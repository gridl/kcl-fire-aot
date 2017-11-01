import ftplib
import logging
import os
import time
import shutil
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


def get_filelist(ftp_loc, username, password, year, doy, ftp_product_path):

    attempt = 0
    while True:
        try:
            ftp_conn = ftp_connect(ftp_loc, username, password)
            ftp_cd(ftp_conn, year, doy, ftp_product_path)
            product_list = get_files(ftp_conn)

            # find the right file
            product_list = [f.split(None, 8)[-1].lstrip() for f in product_list]
            return product_list

        except:
            attempt += 1
            logger.info('Could not access file list for DOY: ' + doy + " on attempt " + str(attempt) +
                        " Reattempting...")
            time.sleep(5)
            if attempt == 10:
                return str()


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


def get_files(ftp_conn):
    file_list = []
    ftp_conn.retrlines("LIST", file_list.append)
    return file_list


def check_inbounds(local_filepath_geo, roi_lat, roi_lon):
    pass


def main():

    # ftp connection info
    ftp_loc = 'nrt4.modaps.eosdis.nasa.gov'
    ftp_root = 'allData/5001/'
    username = 'dnfisher'
    password = '&5ii5yHMtGX9'

    # output filepaths
    output_root =


    # product setup
    geo_product = 'VNP03MOD_NRT'
    img_product = 'VNP02IMG_NRT'
    mod_product = 'VNP02MOD_NRT'

    # roi setup (somewhere over North Atlantic)
    roi_lat = 57.44
    roi_lon = -15.16

    # some iteration variables
    current_doy = 999
    temp_doy_directory = ''

    while True:

        # get current UTC time fo correct folder access
        year = str(datetime.utcnow().year)
        doy = str(datetime.utcnow().timetuple().tm_yday)

        # setup output path for doy
        output_directory = os.path.join(output_root, year, doy)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

            # also setup log file
            hdlr = logging.FileHandler(os.path.join(output_directory, 'doy.log'))
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            hdlr.setFormatter(formatter)
            logger.addHandler(hdlr)
            logger.setLevel(logging.WARNING)

        # create temp folder for doy to hold geo files in
        if doy != current_doy:

            # remove older temp doy folder if it exists
            if temp_doy_directory:
                shutil.rmtree(temp_doy_directory)

            # create new temp folder
            temp_doy_directory = os.path.join(output_root, year, doy + '_temp')
            if not os.path.exists(temp_doy_directory):
                os.makedirs(temp_doy_directory)
            current_doy = doy

        ftp_directory_geo = os.path.join(ftp_root, geo_product)
        ftp_filenames_geo = get_filelist(ftp_loc, username, password, year, doy, ftp_directory_geo)

        for ftp_filename_geo in ftp_filenames_geo:

            temp_filepath_geo = os.path.join(temp_doy_directory, ftp_filename_geo)

            # see if we have the file
            if not os.path.isfile(temp_filepath_geo):
                logger.info("   ...Downloading: " + ftp_filename_geo)
                retrieve(ftp_loc, username, password, year, doy, ftp_directory_geo, ftp_filename_geo,
                         temp_filepath_geo)

                # check if roi in geo_product
                inbounds = True # check_inbounds(temp_filepath_geo, roi_lat, roi_lon)

                # if it is within roi download other products and log times
                if inbounds:
                    ftp_directory_img = os.path.join(ftp_root, img_product)
                    ftp_filename_img = ftp_filename_geo.replace('VNP03IMG', 'VNP02IMG')
                    local_filepath_img = os.path.join(output_directory, ftp_filename_img)
                    retrieve(ftp_loc, username, password, year, doy, ftp_directory_img, ftp_filename_img,
                             local_filepath_img)

                    ftp_directory_mod = os.path.join(ftp_root, mod_product)
                    ftp_filename_mod = ftp_filename_geo.replace('VNP03IMG', 'VNP02MOD')
                    local_filepath_mod = os.path.join(output_directory, ftp_filename_mod)
                    retrieve(ftp_loc, username, password, year, doy, ftp_directory_mod, ftp_filename_mod,
                             local_filepath_mod)

                    # copy over the geo file also
                    local_filepath_geo = os.path.join(output_directory, ftp_filename_geo)
                    shutil.copyfile(temp_filepath_geo, local_filepath_geo)

                    # log the time difference between the current time and the product time
                    current_time = datetime.utcnow()
                    product_time = datetime.strptime(ftp_filename_geo[14:26], "%Y%j.%H%M")
                    logger.info('Downloaded products...')
                    logger.info('Products final download time: ' + str(current_time))
                    logger.info('Products aquisition time: ' + str(product_time))
                    logger.info('Products time diff.: ' + str(current_time - product_time))

        # sleep for a bit
        time.sleep(60)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    main()