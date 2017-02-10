#!/usr/bin/python2
import sys
import os
import re
import glob

# TODO set up iterator to loop over a list of modis files

# Set-up the paths and extract the date-time from the input file name
l1path = sys.argv[1]
l1name = os.path.basename(l1path)
l1dir = os.path.dirname(l1path)

nrt_ws = '/group_workspaces/cems/nrt_ecmwf_metop'
rsg_ws = '/group_workspaces/cems/rsgnceo'
nceo_ws = '/group_workspaces/cems2/nceo_generic'
orac_dir = '/home/users/gethomas/orac_code'

cldsaddir = '/group_workspaces/cems/cloud_ecv/orac/sad_dir'
cldphs = ['WAT', 'ICE']
aersaddir = '/group_workspaces/cems/nceo_aerosolfire/gethomas/luts'
aerphs = ['AMZ', 'BOR', 'CER', 'AFR']  # Amazon, Boreal, Cerrado, Africa
pixel_limit = "800 900 1000 1100"

# Ensure ORAC libaries are able to be found
os.putenv('LD_LIBRARY_PATH', os.getenv('LD_LIBRARY_PATH') + ':'
          + orac_dir + '/libraries/lib')

# Call the pre-processing
os.chdir(orac_dir + '/trunk/tools')
print " **** Calling ORAC preprocessing on file: ****"
print "   " + mod_dir + '/' + l1name
pre_cmd = mod_dir + '/' + l1name \
          + ' -o ' + preoutdir \
          + ' --batch ' \
          + ' --limit ' + pixel_limit

os.system('./orac_preproc.py ' + pre_cmd)

# Call the processing for the desired phases/types
# Path to the preprocessed files
msi_root = glob.glob(preoutdir + '/*_' + yr + mth + day + hr + mn + '_*.msi.nc')
msi_root = os.path.basename(msi_root=msi_root[:len(msi_root) - 7])

# Call main first time  TODO Set channels to the right ones
proc_cmd = '-i ' + preoutdir \
           + ' -o ' + prooutdir \
           + ' --orac_dir ' + orac_dir + '/trunk' \
           + ' --sad_dir ' + cldsaddir \
           + ' --use_channel 1 1 1 0 0 1 1' \
           + ' --phase '

for phs in cldphs:
    print' **** Calling ORAC for type: ' + phs + ' ****'
    os.system('./orac_main.py ' + proc_cmd + phs + ' '
              + msi_root)

# Call main for aerosol phase, does this need to be done for MODIS?  Also set channels correctly
proc_cmd = '-i ' + preoutdir \
           + ' -o ' + prooutdir \
           + ' --orac_dir ' + orac_dir + '/trunk' \
           + ' --sad_dir ' + aersaddir \
           + ' --use_channel 1 1 1 0 0 0 0' \
           + ' -a AppCld1L' \
           + ' --ret_class ClsAerOx' \
           + ' --phase '
#        +' --extra_lines /home/users/gethomas/orac_code/aerosol_scripts_py/xtra_driver_lines.txt' \

for phs in aerphs:
    print' **** Calling ORAC for type: ' + phs + ' ****'
    os.system('./orac_main.py ' + proc_cmd + phs + ' '
              + msi_root)

# Call the post-processor
post_cmd = '-i ' + prooutdir + \
           ' -o ' + postoutdir \
           + ' --orac_dir ' + orac_dir + '/trunk' \
           + '--verbose' \
           + ' --compress ' + msi_root \
           + ' --phases ' + ' '.join(aerphs) + ' ' + ' '.join(cldphs)
os.system('./orac_postproc.py ' + post_cmd)

# Clean-up
