# script to get the set of modis times for use in
# downloading the goes data


import pandas as pd
import datetime

myd021km_plume_df = pd.read_pickle(r"../../data/interim/myd021km_plumes_df.pickle")

files_to_transfer = '/Users/dnf/git/kcl-fire-aot/data/raw/rsync_file_list/files_to_transfer.txt'
path_to_files = '/Users/dnf/git/kcl-fire-aot/data/raw/l1b/'

dates = []
files = []

for index, row in myd021km_plume_df.iterrows():

    with open(files_to_transfer, 'a') as the_file:
        the_file.write(path_to_files + row[0] + '\n')

    doy = row['filename'][14:17]
    dates.append(datetime.datetime.strptime('2014'+doy, '%Y%j').strftime('%d/%m/%Y'))

past_date = ''
for date in dates:
    if date != past_date:
        print date
        past_date = date
