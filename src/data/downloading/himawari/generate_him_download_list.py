'''
Generates a list of files to be used with rsync to download
Himawari files from server.  The list of files is generated
from the same list used to dl mod l1b data.
'''

import os
from datetime import datetime
from datetime import timedelta

import src.config.filepaths as fp


with open(os.path.join(fp.path_to_filelists, 'indonesia_filepaths.txt'), 'rb') as f:
     plumes_files = f.readlines()


band = 'B03'
minutes = ['00', '10', '20', '30', '40', '50']
segments = ['05', '06', '07']

with open(os.path.join(fp.path_to_filelists, 'him_file_list_B03.txt'), 'wb') as f:

    for plume_fname in plumes_files:

        dt = datetime.strptime(plume_fname[10:22], '%Y%j.%H%M')
        ym = str(dt.year) + str(dt.month).zfill(2)


        # iteraete over the times
        for td in xrange(12):
            # get the day
            day = str(dt.day).zfill(2)
            hour = str(dt.hour).zfill(2)

            ymdayhour = ym + day + hour + '00'

            # construct himawari base path
            path = os.path.join(ym, day, ymdayhour)

            # now iterate over minutes
            for m in minutes:
                for seg in segments:

                    # constuct himawair filename
                    fname = 'HS_H08_' + ym + day + '_' + hour + m + '_' + band + '_FLDK_R05_S' + seg + '10.DAT.bz2' + '\n'

                    # write to filelist
                    full_path = os.path.join('/', path, m, band, fname)
                    f.write(full_path)

            dt -= timedelta(hours=1)
