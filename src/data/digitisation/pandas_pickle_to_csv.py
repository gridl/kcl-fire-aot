'''
Having version issues with Pandas when trying to load pickle file
so going to save the output to a csv to see if it helps.
'''

import pandas as pd
import cPickle as pickle

import src.config.filepaths as filepaths

def load_df(path):
    try:
        myd021km_plume_df = pd.read_pickle(path)
    except:
        myd021km_plume_df = pd.DataFrame()
    return myd021km_plume_df


def main():
    myd021km_plume_df = load_df(filepaths.path_to_smoke_plume_polygons_modis)
    myd021km_plume_df.to_csv(filepaths.path_to_smoke_plume_polygons_modis_csv)
    
    # alternative pickle
    pickle.dump(myd021km_plume_df, open('test_pickle.p', 'wb'))

if __name__ == "__main__":
    main()
