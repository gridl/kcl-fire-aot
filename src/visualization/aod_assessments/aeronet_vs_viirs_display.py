import logging
import os

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

import src.config.filepaths as fp

def main():

    df = pd.read_csv(os.path.join(fp.path_to_dataframes, 'aeronet_comp.csv'))

    orac_aod_df = df[df.orac_aod != -999]
    viirs_aod_df = df[df.viirs_aod != -999]

    fig = plt.figure()
    ax0 = plt.subplot(121)
    ax1 = plt.subplot(122)

    sns.regplot(orac_aod_df.aod550, orac_aod_df.orac_aod, ax=ax0, scatter=True)
    sns.regplot(viirs_aod_df.aod550, viirs_aod_df.viirs_aod, ax=ax1, scatter=True)

    plt.show()




if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()