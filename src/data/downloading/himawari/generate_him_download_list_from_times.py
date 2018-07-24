'''
Generates a list of files to be used with rsync to download
Himawari files from server.  The list of files is generated
from the same list used to dl mod l1b data.
'''

import os
import numpy as np
from datetime import timedelta

import src.config.filepaths as fp


with open(os.path.join(fp.path_to_filelists, 'indonesia_filepaths.txt'), 'rb') as f:
     plumes_files = f.readlines()


#root = '/group_workspaces/cems2/nceo_generic/users/xuwd/Himawari8/'
band = 'B14'
minutes = ['00', '10', '20', '30', '40', '50']
segments = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

if band in ["B03"]:
    res = '05'
elif band in ["B02"]:
    res = '10'
else:
    res = '20'

# set year month
y = 2015
m = 7
ym = str(y) + str(m).zfill(2)

# set days to iterate over
start_day = 2
stop_day = 12 + 1
days = np.arange(start_day, stop_day, 1)

# set up the hours
hours = np.arange(0,24,1)

with open(os.path.join(fp.path_to_filelists, 'him_file_list_time_based_'+ band +'.txt'), 'wb') as f:

        for day in days:
            day = str(day).zfill(2)

            # iteraete over the times
            for hour in hours:
                # get the day
                hour = str(hour).zfill(2)

                ymdayhour = ym + day + hour + '00'

                # construct himawari base path
                path = os.path.join(ym, day, ymdayhour)

                # now iterate over minutes
                for m in minutes:
                    for seg in segments:

                        # constuct himawair filename
                        fname = 'HS_H08_' + ym + day + '_' + hour + m + '_' + band + '_FLDK_R'+ res +'_S' + seg + '10.DAT.bz2' + '\n'

                        # write to filelist
                        full_path = os.path.join('/', path, m, band, fname)
                        f.write(full_path)

