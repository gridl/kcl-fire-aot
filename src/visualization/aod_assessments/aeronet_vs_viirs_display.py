import logging
import os

import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import numpy as np

import src.config.filepaths as fp


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def main():

    df = pd.read_csv(os.path.join(fp.path_to_dataframes, 'aeronet_comp.csv'))

    orac_aod_df = df[(df.orac_aod != -999) & (df.aod550 >= 2)]
    viirs_aod_df = df[(df.viirs_aod != -999) & (df.aod550 < 2)]

    fig = plt.figure(figsize=(10,5))
    ax0 = plt.subplot(121)
    ax1 = plt.subplot(122)

    ax0.set_xlim([0,10])
    ax1.set_xlim([0,2])

    ax0.set_ylim([0,10])
    ax1.set_ylim([0,2])

    ax0.plot([-1000, 7500], [-1000, 7500], '--', color='grey')
    ax1.plot([-1000, 6000], [-1000, 6000], '--', color='grey')

    sns.regplot(orac_aod_df.aod550, orac_aod_df.orac_aod, ax=ax0, scatter=True, color='k')
    sns.regplot(viirs_aod_df.aod550, viirs_aod_df.viirs_aod, ax=ax1, scatter=True, color='k')


    # stats
    slope0, intercept0, r_value0, _, _ = scipy.stats.linregress(orac_aod_df.aod550, orac_aod_df.orac_aod)
    slope1, intercept1, r_value1, _, _ = scipy.stats.linregress(viirs_aod_df.aod550, viirs_aod_df.viirs_aod)

    orac_rmse = rmse(orac_aod_df.aod550, orac_aod_df.orac_aod)
    viirs_rmse = rmse(viirs_aod_df.aod550, viirs_aod_df.viirs_aod)

    orac_mean = np.mean(orac_aod_df.aod550 - orac_aod_df.orac_aod)
    viirs_mean = np.mean(viirs_aod_df.aod550 - viirs_aod_df.viirs_aod)

    orac_sd = np.std(orac_aod_df.aod550 - orac_aod_df.orac_aod)
    viirs_sd = np.std(viirs_aod_df.aod550 - viirs_aod_df.viirs_aod)

    textstr0 = '$R^2=%.3f$\n$RMSE=%.2f$\nMean($\pm$SD)=$%.2f(\pm%.2f)$\n Samples$=%.0f$' % (
    r_value0 ** 2, orac_rmse, orac_mean, orac_sd, orac_aod_df.aod550.shape[0])
    textstr1 = '$R^2=%.3f$\n$RMSE=%.2f$\nMean($\pm$SD)=$%.2f(\pm%.2f)$\n Samples$=%.0f$' % (
    r_value1 ** 2, viirs_rmse, viirs_mean, viirs_sd, viirs_aod_df.aod550.shape[0])

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    ax0.text(0.35, 0.21, textstr0, transform=ax0.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    ax1.text(0.35, 0.21, textstr1, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    ax0.set_xlabel('Aeronet 550 um AOD')
    ax1.set_xlabel('Aeronet 550 um AOD')

    ax0.set_ylabel('ORAC 550 um AOD')
    ax1.set_ylabel('VIIRS SP 550 um AOD')


    plt.show()




if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()