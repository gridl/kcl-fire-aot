# -*- coding: utf-8 -*-
import logging
import os
import re
from datetime import datetime
from dateutil.parser import parse

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.misc as misc
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Polygon
from matplotlib.widgets import RadioButtons

import src.config.filepaths as fp


def load_df(path):
    try:
        viirs_plume_df = pd.read_pickle(path)
    except:
        logger.info("viirs dataframe does not exist, creating now")
        viirs_plume_df = pd.DataFrame()
    return viirs_plume_df


def get_timestamp(viirs_sdr_fname):
    try:
        return re.search("[d][0-9]{8}[_][t][0-9]{6}", viirs_sdr_fname).group()
    except Exception, e:
        logger.warning("Could not extract time stamp from: " + viirs_sdr_fname + " with error: " + str(e))
        return ''


def image_seen(viirs_fname):
    if not os.path.exists(fp.processed_filelist_path):
        return False
    else:
        with open(fp.processed_filelist_path, 'r') as txt_file:
            if viirs_fname in txt_file.read():
                logger.info(viirs_fname + " already processed")
                return True
            else:
                return False


def log_digitisation(viirs_fname):
    with open(fp.processed_filelist_path, 'a') as txt_file:
        txt_file.write(viirs_fname + '\n')



def get_viirs_fname(path, timestamp_viirs, viirs_sdr_fname):
    fname = [f for f in os.listdir(path) if timestamp_viirs in f]
    if len(fname) > 1:
        logger.warning("More that one frp granule matched " + viirs_sdr_fname + "selecting 0th option")
        return fname[0]
    elif len(fname) == 1:
        return fname[0]
    else:
        return ''


def get_orac_fname(path, timestamp_viirs, viirs_sdr_fname):
    viirs_dt = datetime.strptime(timestamp_viirs, 'd%Y%m%d_t%H%M%S')
    fname = [f for f in os.listdir(path) if
             abs((viirs_dt -
                  parse(re.search("[_][0-9]{12}[_]", f).group(), fuzzy=True)
                  ).total_seconds()) <= 30]
    if len(fname) > 1:
        logger.warning("More that one frp granule matched " + viirs_sdr_fname + "selecting 0th option")
        return fname[0]
    elif len(fname) == 1:
        return fname[0]
    else:
        return ''


def load_image(f, mode='RGB'):
    return misc.imread(f)


class Annotate(object):
    def __init__(self, tcc, viirs_aod, viirs_flags, orac_aod, orac_cost, ax,
                 plume_polygons, background_polygons):
        self.ax = ax
        self.tcc = tcc
        self.orac_aod = orac_aod
        self.orac_cost = orac_cost
        self.viirs_aod = viirs_aod
        self.viirs_flags = viirs_flags
        self.im = self.ax.imshow(self.orac_aod, interpolation='none', cmap='viridis', vmin=0, vmax=10)
        # if fires is not None:
        #    self.plot = self.ax.plot(fires[1], fires[0], 'r.')
        self.background_polygons = self._add_polygons_to_axis(background_polygons, 'Blues_r')
        self.plume_polygons = self._add_polygons_to_axis(plume_polygons, 'Reds_r')

        # set up the point holders for the plume and background polygons
        self.plume_x = []
        self.plume_y = []
        self.background_x = []
        self.background_y = []

        # set varaible to check if we are using plume tracking on this plume or not
        self.tracking = True

        # set up the digitising patch
        self.plume_p = Circle((1, 1))
        self.background_p = Circle((1, 1))

        # set up the events
        self.ax.figure.canvas.mpl_connect('button_press_event', self.click)

        # set up check to see if we keep annotating
        self.do_annotation = True

        # set up default digitisation type as plume (0 is plume, 1 is background, 2 is vector)
        self.type = 0

        # set up radio buttons
        self.axcolor = 'lightgoldenrodyellow'

        self.rax_peat = plt.axes([0.01, 0.9, 0.1, 0.15], facecolor=self.axcolor)
        self.radio_peat = RadioButtons(self.rax_peat, ('Peat', 'Not'))
        self.radio_peat.on_clicked(self.peat_func)

        self.rax_digitise = plt.axes([0.01, 0.7, 0.1, 0.15], facecolor=self.axcolor)
        self.radio_disitise = RadioButtons(self.rax_digitise, ('Digitise', 'Stop'))
        self.radio_disitise.on_clicked(self.annotation_func)

        self.rax_discard = plt.axes([0.01, 0.5, 0.1, 0.15], facecolor=self.axcolor)
        self.radio_discard = RadioButtons(self.rax_discard, ('Keep', 'Discard'))
        self.radio_discard.on_clicked(self.discard_func)

        self.rax_type = plt.axes([0.01, 0.3, 0.1, 0.15], facecolor=self.axcolor)
        self.radio_type = RadioButtons(self.rax_type, ('Plume', 'Background'))
        self.radio_type.on_clicked(self.type_func)

        self.rax_image = plt.axes([0.01, 0.1, 0.1, 0.15], facecolor=self.axcolor)
        self.radio_image = RadioButtons(self.rax_image, self._radio_labels())
        self.radio_image.on_clicked(self.image_func)

        self.cax = plt.axes([0.925, 0.1, 0.025, 0.8])
        self.cbar = plt.colorbar(self.im, self.cax)

    def _add_polygons_to_axis(self, polygons, cmap):
        patches = [Polygon(verts, True) for verts in polygons]
        colors = [0] * len(patches)
        p = PatchCollection(patches, cmap=cmap, alpha=0.8)
        p.set_array(np.array(colors))
        return self.ax.add_collection(p)

    def _radio_labels(self):
        labels = []
        # FCC and TCC always present
        labels.append('ORAC_AOD')
        labels.append('ORAC_COST')
        labels.append('VIIRS_AOD')
        labels.append('VIIRS_FLAGS')
        labels.append('TCC')
        return tuple(labels)

    def _radio_label_mapping(self):
        label_mapping = {}
        label_mapping['ORAC_AOD'] = self.orac_aod
        label_mapping['ORAC_COST'] = self.orac_cost
        label_mapping['VIIRS_AOD'] = self.viirs_aod
        label_mapping['VIIRS_FLAGS'] = self.viirs_flags
        label_mapping['TCC'] = self.tcc
        return label_mapping

    def peat_func(self, label):
        anno_dict = {'Peat': True, 'Not': False}
        self.tracking = anno_dict[label]

    def annotation_func(self, label):
        anno_dict = {'Digitise': True, 'Stop': False}
        self.do_annotation = anno_dict[label]

    def discard_func(self, label):
        keep_dict = {'Keep': False, 'Discard': True}
        if keep_dict[label] and self.type:
            self.plume_x = []
            self.plume_y = []
        elif keep_dict[label] and not self.type:
            self.background_x = []
            self.background_y = []

    def type_func(self, label):
        type_dict = {'Plume': 0, 'Background': 1}
        self.type = type_dict[label]

    def image_func(self, label):

        image_dict = self._radio_label_mapping()
        im_data = image_dict[label]
        self.im.set_data(im_data)

        if label == "VIIRS_AOD":
            self.im.set_clim(vmax=10, vmin=0)
            self.im.set_cmap('viridis')
        if label == "VIIRS_FLAGS":
            self.im.set_clim(vmax=4, vmin=0)
            cmap = cm.get_cmap('Set1', 4)
            self.im.set_cmap(cmap)
        if label == "ORAC_AOD":
            self.im.set_clim(vmax=10, vmin=0)
            self.im.set_cmap('viridis')
        if label == "ORAC_COST":
            self.im.set_clim(vmax=5, vmin=0)
            self.im.set_cmap('plasma')
        plt.draw()

    def click(self, event):
        if event.button == 3:
            if self.type == 0:
                self.plume_x.append(int(event.xdata))
                self.plume_y.append(int(event.ydata))
                if len(self.plume_x) < 3:
                    self.plume_p = Circle((event.xdata, event.ydata), radius=0.25, facecolor='red', edgecolor='black')
                    self.ax.add_patch(self.plume_p)
                else:
                    self.plume_p.remove()
                    self.plume_p = Polygon(zip(self.plume_x, self.plume_y), color='red', alpha=0.5)
                    plume_p = self.ax.add_patch(self.plume_p)
                self.ax.figure.canvas.draw()
            elif self.type == 1:
                self.background_x.append(int(event.xdata))
                self.background_y.append(int(event.ydata))
                if len(self.background_x) < 3:
                    self.background_p = Circle((event.xdata, event.ydata), radius=0.25, facecolor='blue',
                                               edgecolor='black')
                    self.ax.add_patch(self.background_p)
                else:
                    self.background_p.remove()
                    self.background_p = Polygon(zip(self.background_x, self.background_y), color='blue', alpha=0.5)
                    background_p = self.ax.add_patch(self.background_p)
                self.ax.figure.canvas.draw()


def digitise(tcc, viirs_aod, viirs_flags, orac_aod, orac_cost, viirs_fname):
    plume_polygons = []
    background_polygons = []
    tracking = []

    do_annotation = True
    while do_annotation:

        fig, ax = plt.subplots(1, figsize=(15, 8))
        plt.title(re.search("[d][0-9]{8}[_][t][0-9]{6}", viirs_fname).group())
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # first set up the annotator
        annotator = Annotate(tcc, viirs_aod, viirs_flags, orac_aod, orac_cost, ax,
                             plume_polygons, background_polygons)

        # then show the image
        plt.show()

        # get the polygon points from the closed image, only keep if plume and background polygons
        plume_pts = zip(annotator.plume_x, annotator.plume_y)
        background_pts = zip(annotator.background_x, annotator.background_y)

        if plume_pts and background_pts:
            plume_polygons.append(plume_pts)
            background_polygons.append(background_pts)
            tracking.append(annotator.tracking)

        do_annotation = annotator.do_annotation

    plt.close(fig)

    return plume_polygons, background_polygons, tracking


def append_to_list(plume, background, track, fname, plumes_list):
    row_dict = {}

    row_dict['sensor'] = "VIIRS"
    row_dict['filename'] = fname
    row_dict['plume_extent'] = plume
    row_dict['background_extent'] = background
    row_dict['track_plume'] = track

    # lastly append to the data dictionary
    plumes_list.append(row_dict)


def main():
    """ Loads MODIS data files for the specified geographic region,
        timeframe, and time stamps.  The user can then process these
        images and extract smoke plumes.
    """
    viirs_plume_df = load_df(fp.plume_polygon_path)

    with open(fp.analysis_filelist_path, 'rU') as f:
        for viirs_sdr_fname in f:

            logger.info("Processing viirs file: " + viirs_sdr_fname)

            timestamp_viirs = get_timestamp(viirs_sdr_fname)
            if image_seen(viirs_sdr_fname):
                continue

            # get the filenames
            try:
                tcc_fname = get_viirs_fname(fp.path_to_viirs_sdr_resampled_peat, timestamp_viirs, viirs_sdr_fname)
            except Exception, e:
                logger.warning('Could not load viirs aod file for:' + viirs_sdr_fname + '. Failed with ' + str(e))
                continue

            try:
                aod_fname = get_viirs_fname(fp.path_to_viirs_aod_resampled, timestamp_viirs, viirs_sdr_fname)
            except Exception, e:
                logger.warning('Could not load viirs aod file for:' + viirs_sdr_fname + '. Failed with ' + str(e))
                continue

            try:
                orac_fname = get_orac_fname(fp.path_to_viirs_orac_resampled, timestamp_viirs, viirs_sdr_fname)
            except Exception, e:
                logger.warning('Could not load orac aod file for:' + viirs_sdr_fname + '. Failed with ' + str(e))
                continue

            # load in the data
            try:
                tcc = load_image(os.path.join(fp.path_to_viirs_sdr_resampled_peat, tcc_fname))

            except Exception, e:
                logger.warning('Could not read the input file: ' + viirs_sdr_fname + '. Failed with ' + str(e))
                continue

            # if filenames load in data
            viirs_aod = None
            if aod_fname:
                try:
                    viirs_aod = load_image(os.path.join(fp.path_to_viirs_aod_resampled, aod_fname))

                    # adjust to between 0-2
                    viirs_aod = viirs_aod.astype('float')
                    viirs_aod *= 2.0 / viirs_aod.max()

                    viirs_flags = load_image(os.path.join(fp.path_to_viirs_aod_flags_resampled, aod_fname))
                    viirs_flags = viirs_flags.astype('float')
                    viirs_flags *= 3.0 / viirs_flags.max()

                except Exception, e:
                    logger.warning('Could not read aod file: ' + aod_fname + ' error: ' + str(e))
            if viirs_aod is None:
                logger.info('no viirs aod file for sdr: ' + viirs_sdr_fname + ' continuing')
                continue

            orac_aod = None
            if orac_fname:
                try:
                    orac_aod = load_image(os.path.join(fp.path_to_viirs_orac_resampled, orac_fname))
                    orac_cost = load_image(os.path.join(fp.path_to_viirs_orac_cost_resampled, orac_fname))

                    # adjust to between 0-10
                    orac_aod = orac_aod.astype('float')
                    orac_aod *= 10.0 / orac_aod.max()

                    # adjust to between 0-5
                    orac_cost = orac_cost.astype('float')
                    orac_cost *= 5.0 / orac_cost.max()

                except Exception, e:
                    logger.warning('Could not read aod file: ' + orac_fname + ' error: ' + str(e))
            if orac_aod is None:
                logger.info('no orac aod file for sdr: ' + viirs_sdr_fname + ' continuing')
                continue

            # do the digitising
            plume_polygons, background_polygons, tracking = digitise(tcc,
                                                                     viirs_aod,
                                                                     viirs_flags,
                                                                     orac_aod,
                                                                     orac_cost,
                                                                     viirs_sdr_fname)
            if plume_polygons is None:
                logger.info('no polygons for sdr: ' + viirs_sdr_fname + ' continuing')
                log_digitisation(viirs_sdr_fname)
                continue

            # process plumes and backgrounds
            plumes_list = []
            for pp, bp, t in zip(plume_polygons, background_polygons, tracking):
                append_to_list(pp, bp, t, viirs_sdr_fname, plumes_list)

            # covert pixel/background lists to dataframes and concatenate to main dataframes
            temp_plume_df = pd.DataFrame(plumes_list)
            viirs_plume_df = pd.concat([viirs_plume_df, temp_plume_df])
            viirs_plume_df.to_pickle(fp.plume_polygon_path)

            # if completed log file as digitised
            log_digitisation(viirs_sdr_fname)

    # also write out to csv
    viirs_plume_df.to_csv(fp.plume_polygon_path_csv)


if __name__ == '__main__':
    # plt.ioff()
    matplotlib.pyplot.close("all")

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    main()
