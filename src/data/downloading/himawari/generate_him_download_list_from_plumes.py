'''
Generates a list of files to be used with rsync to download
Himawari files from server.  The list of files is generated
from the same list used to dl mod l1b data.
'''

import os
from datetime import datetime
from datetime import timedelta

import src.config.filepaths as fp


with open(os.path.join(fp.analysis_filelist_path, 'processed_filelist_viirs.txt'), 'rb') as f:
     plumes_files = f.readlines()


#root = '/group_workspaces/cems2/nceo_generic/users/xuwd/Himawari8/'
band = 'B01'
minutes = ['00', '10', '20', '30', '40', '50']
segments = ['05', '06', '07']

with open(os.path.join(fp.analysis_filelist_path, 'him_file_list_'+ band + '.txt'), 'wb') as f:

    for plume_fname in plumes_files:

        #dt = datetime.strptime(plume_fname[10:22], '%Y%j.%H%M')  # for modis
        dt = datetime.strptime(plume_fname[10:27], 'd%Y%m%d_t%H%M%S')
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
                    fname = 'HS_H08_' + ym + day + '_' + hour + m + '_' + band + '_FLDK_R10_S' + seg + '10.DAT.bz2' + '\n'

                    # write to filelist
                    full_path = os.path.join('/', path, m, band, fname)
                    f.write(full_path)

            dt -= timedelta(hours=1)
