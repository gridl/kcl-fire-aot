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
from matplotlib.patches import Circle
import cv2


import config


def ftp_connect_laads(doy, dir):
    ftp = ftplib.FTP("ladsweb.nascom.nasa.gov")
    ftp.login()
    ftp.cwd(dir + str(config.year) + '/')
    ftp.cwd(str(doy))
    return ftp


def get_files(ftp):
    file_list = []
    ftp.retrlines("LIST", file_list.append)
    return file_list


def assess_fire_pixels(doy, local_filename, filename_frp):

    # see if data exists in frp dir, if not then dl it
    if not os.path.isfile(local_filename):
        ftp_laads = ftp_connect_laads(doy, dir='allData/6/MYD14/')
        lf = open(local_filename, "wb")
        ftp_laads.retrbinary("RETR " + filename_frp, lf.write, 8 * 1024)
        lf.close()
        ftp_laads.close()

    frp_data = SD(local_filename, SDC.READ)
    fire_mask = frp_data.select('fire mask').get() >= 7

    # determine whether to process the scene or not
    if frp_data.select('FP_power').checkempty():
        process_flag = False
    else:
        power = frp_data.select('FP_power').get()
        total_fires = len(power)
        total_power = np.sum(power)
        if total_fires > config.min_fires and total_power > config.min_power:
            process_flag = True
        else:
            process_flag = False

    return fire_mask, process_flag



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
        with open('../../data/interim/temp_modis_quicklook.jpg', 'wb') as fname:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, fname)
    return r.status_code


def display_image():
    im = ndimage.imread('../../data/interim/temp_modis_quicklook.jpg', mode="RGB")
    plt.figure(figsize=(16, 16))
    plt.imshow(im)
    plt.draw()
    plt.pause(1)  # <-------

    # raw_input returns the empty string for "enter"
    yes = set(['yes', 'y', 'ye', ''])
    no = set(['no', 'n'])

    while True:
        choice = raw_input("Process image: [y,n]").lower()
        if choice in yes:
            plt.close()
            return True
        elif choice in no:
            plt.close()
            return False
        else:
            sys.stdout.write("Please respond with 'y' or 'n'")


def retrieve_l1(doy, local_filename, filename_l1):
    ftp_laads = ftp_connect_laads(doy, dir='allData/6/MYD021KM/')
    lf = open(local_filename, "wb")
    ftp_laads.retrbinary("RETR " + filename_l1, lf.write, 8 * 1024)
    lf.close()
    ftp_laads.close()


def read_modis(local_filename, fire_mask):

    mod_data = SD(local_filename, SDC.READ)
    mod_params_ref = mod_data.select("EV_1KM_RefSB").attributes()
    mod_params_emm = mod_data.select("EV_1KM_Emissive").attributes()
    ref = mod_data.select("EV_1KM_RefSB").get()
    emm = mod_data.select("EV_1KM_Emissive").get()

    # switch the red and bluse channels, so the we get nice bright red plumes
    ref_chan = 0
    emm_chan = 10
    r = (ref[ref_chan, :, :] - mod_params_ref['radiance_offsets'][ref_chan]) * mod_params_ref['radiance_scales'][
        ref_chan]
    b = (emm[emm_chan, :, :] - mod_params_emm['radiance_offsets'][emm_chan]) * mod_params_emm['radiance_scales'][
        emm_chan]
    g = (r - b) / (r + b)

    r = np.round((r * (255 / np.max(r))) * 1).astype('uint8')
    g = np.round((g * (255 / np.max(g))) * 1).astype('uint8')
    b = np.round((b * (255 / np.max(b))) * 1).astype('uint8')

    mini = 5
    maxi = 95

    r_min, r_max = np.percentile(r, (mini, maxi))
    r = exposure.rescale_intensity(r, in_range=(r_min, r_max))
    r[fire_mask] = 255

    g_min, g_max = np.percentile(g, (mini, maxi))
    g = exposure.rescale_intensity(g, in_range=(g_min, g_max))
    g[fire_mask] = 0

    b_min, b_max = np.percentile(b, (mini, maxi))
    b = exposure.rescale_intensity(b, in_range=(b_min, b_max))
    b[fire_mask] = 0

    rgb = np.dstack((r, g, b))

    return rgb


class Annotate(object):
    def __init__(self, im):
        self.f = plt.figure(figsize=(30, 15))
        self.ax = plt.gca()
        self.im = self.ax.imshow(im, interpolation='none')

        # set up the circle to hold the fires
        self.circ = Circle((0,0))

        # set up the polygon
        self.x = []
        self.y = []
        self.polygons = []

        # set up the events
        self.ax.figure.canvas.mpl_connect('button_press_event', self.click)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.release)


    def click(self, event):
        if event.button == 3:
            self.x.append(int(event.xdata))
            self.y.append(int(event.ydata))
            self.ax.add_patch(Circle((event.xdata, event.ydata), radius=0.25, facecolor='red', edgecolor='black'))
            self.ax.figure.canvas.draw()
        elif event.button == 2:
            pass

    def release(self, event):
        if event.botton == 2:
            pass


def digitise(img):

    img_copy = img.copy()
    smoke_polygons = []
    fire_polygons = []

    while True:

        # first set up the annotator and show the image
        annotator = Annotate(img_copy)
        plt.show()

        # show them what they have digitised, and check if they are OK with that
        # if they are append the polygon, and modify the RGB to reflect the digitised region
        pts = zip(annotator.x, annotator.y)
        if not pts:
            print "you must select some points"
            continue

        digitised_copy = img_copy.copy()

        cv2.fillConvexPoly(digitised_copy, np.array(pts), (255, 255, 255, 255))
        plt.imshow(digitised_copy)
        plt.show()

        happy = raw_input("Are you happy with this plume digitisation? [Y,n]")
        if happy.lower() in ["", "y", "yes", 'ye']:
            smoke_polygons.append(pts)
            img_copy = digitised_copy

        # ask if they want to digitise some more?
        more = raw_input("Do you want to digitise more plumes? [Y,n]")
        if more.lower() not in ["", "y", "yes", 'ye']:
            break

    return polygons


def make_mask(img, image_pts):
    matrix = np.zeros((img.shape[0], img.shape[1]))
    for image_pt in image_pts:
        image_pt_reshape = np.array(image_pt).reshape(-1, 1, 2).squeeze()
        cv2.drawContours(matrix, [image_pt_reshape], -1, (1), thickness=-1)
    return matrix


def main():
    """ Loads MODIS data files for the specified geographic region,
        timeframe, and time stamps.  The user can then process these
        images and extract smoke plumes.
    """
    logger = logging.getLogger(__name__)
    logger.info('making data set from raw data')

    for doy in config.doy_range:

        # connect to ftp and move to correct doy
        ftp_laads = ftp_connect_laads(doy, dir='allData/6/MYD021KM/')
        l1_file_list = get_files(ftp_laads)
        ftp_laads.close()
        ftp_laads = ftp_connect_laads(doy, dir='allData/6/MYD14/')
        frp_file_list = get_files(ftp_laads)
        ftp_laads.close()

        # TODO need to ensure that the filenames are correctly associatied
        for f_l1, f_frp in zip(l1_file_list, frp_file_list):

            filename_l1 = f_l1.split(None, 8)[-1].lstrip()
            filename_frp = f_frp.split(None, 8)[-1].lstrip()

            time_stamp = re.search("[.][0-9]{4}[.]", filename_l1).group()
            if (int(time_stamp[1:-1]) < config.min_time) | \
               (int(time_stamp[1:-1]) > config.max_time):
                continue

            # asses is fire pixels in scene
            local_filename = os.path.join(r"../../data/raw/frp/", filename_frp)
            fire_mask, fires = assess_fire_pixels(doy, local_filename, filename_frp)

            if not fires:
                continue

            mod_url = get_mod_url(doy, time_stamp)
            status = get_image(mod_url)
            if status != 200:  # no access then continue
                continue
            process_flag = display_image()

            if process_flag:

                # download the file
                local_filename = os.path.join(r"../../data/raw/l1b", filename_l1)

                if not os.path.isfile(local_filename):  # if we dont have the file, then dl it
                    retrieve_l1(doy, local_filename, filename_l1)

                # do the digitising
                img = read_modis(local_filename, fire_mask)
                image_pts = digitise(img)
                plume_mask = make_mask(img, image_pts)


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
