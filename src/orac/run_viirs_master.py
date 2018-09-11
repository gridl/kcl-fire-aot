import os
import glob


class ProcParams(object):
    def __init__(self):
        self.sensor = "viirs"
        self.proc_level = 'pre'

        self.data_dir = '/group_workspaces/jasmin2/nceo_aerosolfire/data/orac_proc/viirs/sdr/'
        self.geo_dir = '/group_workspaces/jasmin2/nceo_aerosolfire/data/orac_proc/viirs/geo/'
        self.output_dir = '/group_workspaces/jasmin2/nceo_aerosolfire/data/orac_proc/viirs/outputs/'

        self.cldsaddir = '/group_workspaces/cems2/nceo_generic/cloud_ecv/data_in/sad_dir/viirs-npp_WAT'
        self.cldphs = ['WAT']
        self.aersaddir = '/group_workspaces/jasmin2/nceo_aerosolfire/data/orac_proc/viirs/luts'
        self.aerphs = ['AMW']


def run_pre(pp):

    # iterate over viirs files in data dir

    viirs_files = glob.glob(pp.data_dir + 'SVM01*')

    for input_file_path in viirs_files:
	print input_file_path
        output_file_path = os.path.join(pp.output_dir, 'pre')

        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)

        # call ORAC preproc
        pre_cmd = '-i ' + input_file_path
        os.system('./single_process.py ' + pre_cmd)


def run_pro(pp):

    # iterate over mod files in data dir
    for root, dirs, files in os.walk(pp.output_dir):

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
            #msi_time = datetime.strptime(msi_root.split('_')[-2], '%Y%m%d%H%M')
            #msi_str_time = datetime.strftime(msi_time, '%Y%j.%H%M')

            pro_dir = root.replace('pre', 'main')

            # Set up and call ORAC for the defined phases --ret_class ClsAerOx
            proc_cmd = '-i ' + root \
                       + ' -o ' + pro_dir \
                       + ' --sad_dir ' + pp.cldsaddir \
                       + ' --use_channel 0 0 0 1 1 0 1 0 0 0 1 0 0 0 0 0 -a AppCld1L --ret_class ClsAerOx ' \
                       + '  --batch ' \
                       + ' --phase '

            #for phs in pp.aerphs:
            for phs in pp.cldphs:
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
