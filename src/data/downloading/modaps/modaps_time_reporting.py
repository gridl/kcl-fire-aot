import glob
import os
import datetime
import time

import numpy as np


def read_log_file(log_file):
    with open(log_file, 'rb') as lf:
        lines = lf.readlines()
    return lines

def extract_log_times(log, time_list):
    for line in log:
        if 'Products time diff' in line:
            time_string = line.split('Products time diff')[-1][2:].split('.')[:-1][0]
            try:
                t = time.strptime(time_string, ' %H:%M:%S')
            except:
                continue
            time_list.append(datetime.timedelta(hours=t.tm_hour, minutes=t.tm_min, seconds=t.tm_sec).total_seconds())



def main():
    log_dir = '/Users/dnf/Projects/kcl-fire-aot/data/nrt_test/modis/2017'
    log_files = glob.glob(os.path.join(log_dir, '*/doy.log'))

    if 'modis' in log_dir:
        hours = 2
    else:
        hours = 3

    time_list = []
    for log_file in log_files:
        log = read_log_file(log_file)
        extract_log_times(log, time_list)

    # convert to array
    time_list = np.array(time_list)

    # set max acceptable time
    max_time = hours * 3600
    time_list = time_list[time_list <= max_time]

    print 'nobs:', time_list.size

    m, s = divmod(np.mean(time_list), 60)
    h, m = divmod(m, 60)
    print 'Mean time diff:', "%d:%02d:%02d" % (h, m, s)

    m, s = divmod(np.std(time_list), 60)
    h, m = divmod(m, 60)
    print 'SD time diff:', "%d:%02d:%02d" % (h, m, s)

    m, s = divmod(np.max(time_list), 60)
    h, m = divmod(m, 60)
    print 'Max time diff:', "%d:%02d:%02d" % (h, m, s)

    m, s = divmod(np.min(time_list), 60)
    h, m = divmod(m, 60)
    print 'Min time diff:', "%d:%02d:%02d" % (h, m, s)



if __name__ == "__main__":
    main()
