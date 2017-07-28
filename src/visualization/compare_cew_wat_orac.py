'''
Script to compare CEW and WAT LUT AOD, RE and Cost
'''

import glob
import logging
import cPickle as pickle
from datetime import datetime

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from matplotlib.path import Path
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")

import src.config.filepaths as filepaths


def get_primary_time(primary_file):
    tt = datetime.strptime(primary_file.split('_')[-2], '%Y%m%d%H%M').timetuple()
    primary_datestring = str(tt.tm_year) + \
                         str(tt.tm_yday).zfill(3) + \
                         '.' + \
                         str(tt.tm_hour).zfill(2) + \
                         str(tt.tm_min).zfill(2)
    return primary_datestring


def open_primary(primary_file):
    return Dataset(primary_file)


def get_sub_df(primary_time, mask_df):
    return mask_df[mask_df['filename'].str.contains(primary_time)]


def make_mask(primary_data, primary_time, mask_df):

    primary_shape = primary_data.variables['cer'].shape

    # get teh sub dataframe associated with the mask
    sub_df = get_sub_df(primary_time, mask_df)

    # create grid to hold the plume mask
    nx = primary_shape[1]
    ny = primary_shape[0]
    mask = np.zeros((ny, nx))

    # generate the mask for each of the plumes
    for i, plume in sub_df.iterrows():
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        poly_verts = plume['plume_extent']

        # apply mask
        path = Path(poly_verts)
        grid = path.contains_points(points)
        grid = grid.reshape((ny, nx))

        mask += grid
    return mask


def main():

    try:

        wat_dd = pickle.load(open(filepaths.path_to_orac_visuals + 'wat_dd.p', 'rb'))
        cew_dd = pickle.load(open(filepaths.path_to_orac_visuals + 'amz_dd.p', 'rb'))

    except:

        wat_dd = {'cot': [], 'cer': [], 'costjm': []}
        cew_dd = {'cot': [], 'cer': [], 'costjm': []}

        # read in the masks
        mask_df = pd.read_pickle(filepaths.path_to_smoke_plume_masks)

        # get wat and dew files
        wat_files = glob.glob(filepaths.path_to_processed_orac + '*/*/*/*WAT.primary*')
        cew_files = glob.glob(filepaths.path_to_processed_orac + '*/*/*/*AMZ.primary*')

        for wat_f, cew_f in zip(wat_files, cew_files):

            # first get the primary file time
            primary_time = get_primary_time(wat_f)

            # open up the ORAC primary files
            primary_wat_data = open_primary(wat_f)
            primary_cew_data = open_primary(cew_f)

            # make the smoke plume mask
            plume_mask = make_mask(primary_wat_data, primary_time, mask_df)


            for k in wat_dd.keys():
                wat_dd[k].extend(primary_wat_data[k][:][(plume_mask).astype('bool')])
                cew_dd[k].extend(primary_cew_data[k][:][(plume_mask).astype('bool')])

            pickle.dump(wat_dd, open(filepaths.path_to_orac_visuals + 'wat_dd.p', 'wb'))
            pickle.dump(cew_dd, open(filepaths.path_to_orac_visuals + 'cew_dd.p', 'wb'))

    # lets do the plotting

    mask = (np.array(wat_dd['costjm']) > 0) & (np.array(cew_dd['costjm']) > 00)
    masked_wat_costjm = np.array(wat_dd['costjm'])[mask]
    masked_cew_costjm = np.array(cew_dd['costjm'])[mask]

    cost_wat = pd.Series(masked_wat_costjm, name="$costjm_{WAT}$")
    cost_cew = pd.Series(masked_cew_costjm, name="$costjm_{AMZ}$")
    sns_plot = sns.jointplot(cost_wat, cost_cew, kind="kde", size=7, space=0)
    plt.savefig('cost.png')
    plt.close()

    mask = (~(np.isnan(wat_dd['cer'])) & ~(np.isnan(cew_dd['cer'])))
    masked_wat_cer = np.array(wat_dd['cer'])[mask]
    masked_cew_cer = np.array(cew_dd['cer'])[mask]

    re_wat = pd.Series(masked_wat_cer, name="$RE_{WAT}$")
    re_cew = pd.Series(masked_cew_cer, name="$RE_{AMZ}$")
    sns_plot = sns.jointplot(re_wat, re_cew, kind='kde', size=7, space=0)
    plt.savefig('re.png')
    plt.close()

    mask = (np.array(wat_dd['cot']) < 5) & (np.array(cew_dd['cot']) < 5)
    masked_wat_aod = np.array(wat_dd['cot'])[mask]
    masked_cew_aod = np.array(cew_dd['cot'])[mask]

    aod_wat = pd.Series(masked_wat_aod, name="$AOD_{WAT}$")
    aod_cew = pd.Series(masked_cew_aod, name="$AOD_{AMZ}$")
    sns_plot = sns.jointplot(aod_wat, aod_cew, kind="kde", size=7, space=0)
    plt.savefig('aod.png')
    plt.close()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()


