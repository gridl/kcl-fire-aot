# script to get the set of modis times for use in
# downloading the goes data


import pandas as pd
import datetime
import src.config.filepaths as filepaths

myd021km_plume_df = pd.read_pickle(filepaths.path_to_smoke_plume_masks)

dates = []
files = []
for index, row in myd021km_plume_df.iterrows():

    with open(filepaths.path_to_transfer_file, 'a') as the_file:
        the_file.write(filepaths.path_to_modis_l1b + row[0] + '\n')

    doy = row['filename'][14:17]
    dates.append(datetime.datetime.strptime('2014'+doy, '%Y%j').strftime('%d/%m/%Y'))

past_date = ''
for date in dates:
    if date != past_date:
        print date
        past_date = date
