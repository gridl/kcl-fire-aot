# script to get the set of modis times for use in
# downloading the goes data

import pandas as pd
import datetime

myd021km_plume_df = pd.read_pickle(r"../../data/interim/myd021km_plumes_df.pickle")

dates = []
for index, row in myd021km_plume_df.iterrows():
    doy = row['filename'][14:17]
    dates.append(datetime.datetime.strptime('2014'+doy, '%Y%j').strftime('%d/%m/%Y'))

past_date = ''
for date in dates:
    if date != past_date:
        print date
        past_date = date
