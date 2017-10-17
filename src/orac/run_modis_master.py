#!/usr/bin/python2
import sys
import os
import re
import glob
from datetime import datetime


class ProcParams(object):
    def __init__(self):
        self.filelist_name = 'indonesia_filepaths.txt'
        self.processing_filelist_dir = '/home/users/dnfisher/nceo_aerosolfire/data/filelists/processing/'
        self.transfer_filelist_dir = '/home/users/dnfisher/nceo_aerosolfire/data/filelists/transfer/'
        self.data_dir = '/group_workspaces/cems2/nceo_generic/satellite_data/modis_c6/myd021km/'
        self.geo_dir = '/group_workspaces/cems2/nceo_generic/satellite_data/modis_c6/myd03/'
        self.output_dir = '/group_workspaces/cems/nceo_aerosolfire/data/orac_proc/myd/'
        self.proc_level = 'pro'

        self.cldsaddir = '/group_workspaces/cems2/nceo_generic/cloud_ecv/data_in/sad_dir/modis_WAT'
        self.cldphs = ['WAT']
        self.aersaddir = '/group_workspaces/cems/nceo_aerosolfire/luts/sad'
        self.aerphs = ['AMW']


def run_pre(proc_params):
    # get list of files which we are processing
    with open(os.path.join(proc_params.processing_filelist_dir, proc_params.filelist_name), 'rb') as f:
        file_list = f.readlines()
    file_list = [f.rstrip() for f in file_list]
    print file_list

    # iterate over mod files in proc filelisr
    for root, dirs, files in os.walk(proc_params.data_dir):
        for f in files:
            if f in file_list:
                input_file_path = os.path.join(root, f)
            # elif f:  # check if not empty
            #    input_file_path = os.path.join(root, f)
            else:
                continue
            print f            

            split_root = root.split('/')
            year = split_root[-2]
            doy = split_root[-1]
            output_file_path = os.path.join(proc_params.output_dir, year, doy, 'pre')
            geo_file_path = os.path.join(proc_params.geo_dir, year, doy)

            if not os.path.exists(output_file_path):
                os.makedirs(output_file_path)

            # call ORAC preproc
            pre_cmd = input_file_path \
                      + ' -o ' + output_file_path \
                      + ' -g ' + geo_file_path \
                      + ' -c 1 2 3 4 7 20 31 32 ' \
                      + ' --skip_ecmwf_hr ' \
                      + ' --batch '
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
                           + ' --batch ' \
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
