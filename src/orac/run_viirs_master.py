#!/usr/bin/python2
import sys
import os
import re
import glob
from datetime import datetime


class ProcParams(object):
    def __init__(self):
        self.sensor = "viirs"
        self.proc_level = 'pre'

        self.data_dir = '/home/users/dnfisher/nceo_aerosolfire/data/orac_proc/viirs/sdr/'
        self.geo_dir = '/home/users/dnfisher/nceo_aerosolfire/data/orac_proc/viirs/geo/'
        self.output_dir = '/group_workspaces/cems/nceo_aerosolfire/data/orac_proc/viirs/'

        self.cldsaddir = '/group_workspaces/cems2/nceo_generic/cloud_ecv/data_in/sad_dir/viirs-npp_WAT'
        self.cldphs = ['WAT']


def run_pre(pp):

    # iterate over viirs files in data dir

    viirs_files = glob.glob(pp.data_dir + 'SVM01_npp_d20150908_t0601144_e0602386_b20016_c20171125110259159115_noaa_ops.h5')

    for input_file_path in viirs_files:

        output_file_path = os.path.join(pp.output_dir, 'pre')

        print output_file_path

        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)

        # call ORAC preproc
        pre_cmd = input_file_path \
                  + ' -o ' + output_file_path \
                  + ' -g ' + pp.geo_dir \
                  + ' -c 1 2 3 4 5 7 8 11 ' \
                  + ' --skip_ecmwf_hr ' \
                  + ' --keep_driver '
        os.system('./orac_preproc.py ' + pre_cmd)


def run_pro(proc_params):

    # get list of files which we are processing
    with open(os.path.join(proc_params.processing_filelist_dir, proc_params.filelist_name), 'rb') as f:
        file_list = f.readlines()

    with open(os.path.join(proc_params.transfer_filelist_dir, proc_params.filelist_name), 'wb') as transfer_filelist:

        # iterate over mod files in data dir
        for root, dirs, files in os.walk(proc_params.output_dir):

            if 'pre' not in root:  # we only want the pre proc dirs
                continue

            # find preproc naming in current root dir
            try:
                msi_roots = glob.glob(root + '/*.msi.nc')
            except:
                continue

            for msi_root in msi_roots:
                msi_root = os.path.basename(msi_root)[:-7]

                # check if msi_root is one of the files to be processed in the file list
                msi_time = datetime.strptime(msi_root.split('_')[-2], '%Y%m%d%H%M')
                msi_str_time = datetime.strftime(msi_time, '%Y%j.%H%M')
                if not any(msi_str_time in f for f in file_list):
                    continue

                pro_dir = root.replace('pre', 'main')

                # Set up and call ORAC for the defined phases --ret_class ClsAerOx
                proc_cmd = '-i ' + root \
                           + ' -o ' + pro_dir \
                           + ' --sad_dir ' + proc_params.aersaddir \
                           + ' --use_channel 1 1 0 1 1 0 0 0 -a AppCld1L --ret_class ClsAerOx ' \
                           + ' --keep_driver ' \
                           + ' --phase '

                for phs in proc_params.aerphs:
                    # write out processed filenames for transfer
                    transfer_filelist.write(os.path.join(pro_dir, msi_root + phs + ".primary.nc") + '\n')                  
                    # call orac
                    os.system('./orac_main.py ' + proc_cmd + phs + ' ' + msi_root)


def main():
    # get proc class
    proc_params = ProcParams()

    # Call the pre-processing
    if proc_params.proc_level == 'pre':
        run_pre(proc_params)
    elif proc_params.proc_level == 'pro':
        run_pro(proc_params)


if __name__ == "__main__":
    main()
