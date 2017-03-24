#!/usr/bin/python2
import sys
import os
import re
import glob


class ProcParams(object):
    def __init__(self):
        self.data_dir = '/group_workspaces/cems2/nceo_generic/satellite_data/modis_c6/myd021km/2014/'
        self.output_dir = '/group_workspaces/cems/nceo_aerosolfire/data/orac_proc/myd/'
        self.proc_level = 'pre'  # main

        self.cldsaddir = '/group_workspaces/cems/cloud_ecv/orac/sad_dir'
        self.cldphs = ['WAT', 'ICE']
        self.aersaddir = '/group_workspaces/cems/nceo_aerosolfire/luts/sad'
        self.aerphs = ['AFR']


def run_pre(proc_params):
    # iterate over mod files in data dir
    for root, dirs, files in os.walk(proc_params.data_dir):
        for f in files:
            if f:  # check if not empty
                input_file_path = os.path.join(root, f)
            else:
                continue

            output_file_path = os.path.join(proc_params.output_dir, root.split('/')[-1], 'pre')
            geo_file_path = os.path.join(proc_params.geo_dir, root.split('/')[-1])

            if not os.path.exists(output_file_path):
                os.makedirs(output_file_path)

            # call ORAC preproc
            pre_cmd = input_file_path \
                      + ' -o ' + output_file_path \
                      + ' -g ' + geo_file_path
                      + ' --batch '
            os.system('./orac_preproc.py ' + pre_cmd)


def run_pro(proc_params):
    # iterate over mod files in data dir
    for root, dirs, files in os.walk(proc_params.output_dir):

        if 'pre' not in root:  # we only want the pre proc dirs
            continue

        # find preproc naming in current root dir
        msi_root = glob.glob(root + '/*.msi.nc')[0]
        msi_root = os.path.basename(msi_root)[:-7]

        pro_dir = root.replace('pre', 'main')

        # Set up and call ORAC for the defined phases
        proc_cmd = '-i ' + root \
                   + ' -o ' + pro_dir \
                   + ' --sad_dir ' + proc_params.aersaddir \
                   + ' --use_channel 1 1 1 0 1 1  -a AppCld1L --ret_class ClsAerOx' \
                   + ' --keep_driver ' \
                   + ' -V ' \
                   + ' --phase ' \
                   + ' --batch '

        for phs in proc_params.aerphs:
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
