'''
Having version issues with Pandas when trying to load pickle file
so going to save the output to a csv to see if it helps.
'''

import pandas as pd
import cPickle as pickle

import src.config.filepaths as filepaths

def main():
    in_path = '/Users/danielfisher/Projects/kcl-fire-aot/data/Asia/processed/himawari/updated/frp_df.p'
    out_path = '/Users/danielfisher/Projects/kcl-fire-aot/data/Asia/processed/himawari/updated/frp_df.csv'

    df = pd.read_pickle(in_path)
    df.to_csv(out_path)
    
    # # alternative pickle
    # pickle.dump(myd021km_plume_df, open('test_pickle.p', 'wb'))

if __name__ == "__main__":
    main()
