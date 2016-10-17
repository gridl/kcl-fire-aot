# -*- coding: utf-8 -*-
import os
import sys
import logging
from dotenv import find_dotenv, load_dotenv

import ftplib
import datetime
import re

import requests
import shutil

import numpy as np
from pyhdf.SD import SD, SDC
from skimage import exposure
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import cv2


import config


def ftp_connect(doy):
    ftp = ftplib.FTP("ladsweb.nascom.nasa.gov")
    ftp.login()
    ftp.cwd('allData/6/MYD021KM/' + str(config.year) + '/')
    ftp.cwd(str(doy))
    return ftp


def get_files(ftp, doy):
    file_list = []
    ftp.retrlines("LIST", file_list.append)
    return file_list


def get_mod_url(doy, time_stamp):
    date = datetime.datetime(config.year, 1, 1) + datetime.timedelta(doy - 1)
    date = date.strftime("%Y_%m_%d")
    mod_url = "http://modis-atmos.gsfc.nasa.gov/IMAGES/MYD02/GRANULE/{0}/{1}rgb143.jpg".format(date,
                                                                                           str(doy) +
                                                                                           time_stamp)
    return mod_url


def get_image(mod_url):
    r = requests.get(mod_url, stream=True)
    if r.status_code == 200:
        with open('current.jpg', 'wb') as fname:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, fname)
    return r.status_code


def display_image():
    im = ndimage.imread('current.jpg', mode="RGB")
    plt.figure(figsize=(16, 16))
    plt.imshow(im)
    plt.draw()
    plt.pause(1)  # <-------

    # raw_input returns the empty string for "enter"
    yes = set(['yes', 'y', 'ye', ''])
    no = set(['no', 'n'])

    choice = raw_input("Process image: [y,n]").lower()
    if choice in yes:
        plt.close()
        return True
    elif choice in no:
        plt.close()
        return False
    else:
        sys.stdout.write("Please respond with 'yes' or 'no'")


def read_modis(local_filename):

    mod_data = SD(local_filename, SDC.READ)
    mod_params_ref = mod_data.select("EV_1KM_RefSB").attributes()
    ref = mod_data.select("EV_1KM_RefSB").get()

    ref0 = 0
    ref1 = 13
    ref2 = 14

    r = (ref[ref0, :, :] - mod_params_ref['reflectance_offsets'][ref0]) * mod_params_ref['reflectance_scales'][ref0]
    g = (ref[ref1, :, :] - mod_params_ref['reflectance_offsets'][ref1]) * mod_params_ref['reflectance_scales'][ref1]
    b = (ref[ref2, :, :] - mod_params_ref['reflectance_offsets'][ref2]) * mod_params_ref['reflectance_scales'][ref2]

    b_eq = exposure.equalize_hist(b)
    g_eq = exposure.equalize_hist(g)
    r_eq = exposure.equalize_hist(r)

    rb_eq = r_eq / b_eq
    rb_eq = (rb_eq * (255 / np.max(rb_eq))).astype('uint8')

    b_eq = np.round((b_eq * (255 / np.max(b_eq))) * 0.35).astype('uint8')
    g_eq = np.round((g_eq * (255 / np.max(g_eq))) * 0.35).astype('uint8')
    r_eq = np.round((r_eq * (255 / np.max(r_eq))) * 1.0).astype('uint8')

    rgb = np.dstack((r_eq, g_eq, b_eq))
    return rgb


def digitise(img):

    def draw_poly(event, i, j, flags, param):
        # grab references to the global variables

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed (also add the image shifts)
        if event == cv2.EVENT_LBUTTONDOWN:
            current_pt.append((i, j))

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished

            cv2.circle(image, (i, j), 1, (0, 255, 0), 2)
            cv2.imshow("image", image)

    # make a list to hold the final set of points
    current_pt = []
    image_pt = []

    # lets make an outloop to iterate over the various parts of the image
    y_start = [0, img.shape[0] / 2, img.shape[0]]
    x_start = [0, img.shape[1] / 2, img.shape[1]]

    for iy, y in enumerate(y_start[:-1]):
        for ix, x in enumerate(x_start[:-1]):

            img_sub = img[y: y_start[iy + 1], x:x_start[ix + 1]]

            # load the image, clone it, and setup the mouse callback function
            image = cv2.cvtColor(img_sub, cv2.COLOR_BGR2RGB)
            base_image = image.copy()
            digitised_image = image.copy()

            cv2.namedWindow("image")
            cv2.setMouseCallback("image", draw_poly)

            # create a list to hold all reference points
            quadrant_pt = []

            # keep looping until the 'c' key is pressed
            while True:

                # display the image and wait for a keypress
                cv2.imshow("image", image)
                key = cv2.waitKey(1) & 0xFF

                # if the 'x' ket is pressed, reset all points for quadrant
                if key == ord("x"):
                    image = base_image.copy()
                    digitised_image = base_image.copy()
                    current_pt = []
                    quadrant_pt = []

                # if the 'r' key is pressed, reset the current points and image
                if key == ord("r"):
                    image = digitised_image.copy()
                    current_pt = []

                # if the 'c' key is pressed update quadrant and the digistied plume in image, and reset current_pt
                elif key == ord("c"):
                    image = digitised_image.copy()
                    cv2.fillConvexPoly(image, np.array(current_pt), (255, 0, 0, 120))
                    digitised_image = image.copy()
                    if current_pt:

                        # adjust the current points based on image quadrant shifts
                        m_coords, n_coords = zip(*current_pt)
                        current_pt = zip([m + x for m in m_coords],
                                         [n + y for n in n_coords])

                        quadrant_pt.append(current_pt)
                        current_pt = []

                # if the 'q' key is pressed, break from the loop to stop digitising
                elif key == ord("q"):
                    break

            # append the points from the quadrant to the image list
            if quadrant_pt:
                for item in quadrant_pt:
                    image_pt.append(item)

            print image_pt

            # close all open windows
            cv2.destroyAllWindows()

    return image_pt


def main():
    """ Loads MODIS data files for the specified geographic region,
        timeframe, and time stamps.  The user can then process these
        images and extract smoke plumes.
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    global refPt, cropping
    refPt = []
    cropping = False

    for doy in config.doy_range:

        # connect to ftp and move to correct doy
        ftp = ftp_connect(doy)
        file_list = get_files(ftp, doy)
        ftp.close()

        for f in file_list:

            filename = f.split(None, 8)[-1].lstrip()

            time_stamp = re.search("[.][0-9]{4}[.]", filename).group()
            if (int(time_stamp[1:-1]) < config.min_time) | \
               (int(time_stamp[1:-1]) > config.max_time):
                continue

            mod_url = get_mod_url(doy, time_stamp)
            status = get_image(mod_url)
            if status != 200:  # no access then continue
                continue
            process_flag = display_image()

            if process_flag:

                # download the file
                local_filename = os.path.join(r"../../data/raw/l1b", filename)

                if not os.path.isfile(local_filename):  # if we dont have the file, then dl it
                    ftp = ftp_connect(doy)
                    lf = open(local_filename, "wb")
                    ftp.retrbinary("RETR " + filename, lf.write, 8*1024)
                    lf.close()
                    ftp.close()

                # do the digitising
                img = read_modis(local_filename)
                image_pts = digitise(img)

                matrix = np.zeros((img.shape[0], img.shape[1]))
                for image_pt in image_pts:
                    image_pt_reshape = np.array(image_pt).reshape(-1, 1, 2).squeeze()
                    cv2.drawContours(matrix, [image_pt_reshape], -1, (1), thickness=-1)

                while True:
                    cv2.imshow("image", matrix)
                    key = cv2.waitKey(1) & 0xFF

                    # if the 'c' key is pressed, break from the loop
                    if key == ord("q"):
                        break
                cv2.destroyAllWindows()

                # get the sample indicies
                list_of_points_indices = np.nonzero(matrix)







                # if desiredata features in image then load data
    #ftp://ladsweb.nascom.nasa.gov/allData/6/MYD021KM/2011/267/

    # perform manual feature extraction

    # store in database

    # close off data



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
