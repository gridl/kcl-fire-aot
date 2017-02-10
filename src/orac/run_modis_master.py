#!/usr/bin/python2
import sys
import os
import re
import glob

# TODO set up iterator to loop over a list of modis files
# Hard code in the paths and the files for now
data_dir = '/home/users/dnfisher/soft/orac/data'
mod_dir = data_dir + '/testinput'
l1name = 'MYD021KM.A2008172.0405.005.2009317014309.hdf'
pixel_limit = "800 900 1000 1100"
yr = '2008'
doy = '172'
hr = '.04'
mn = '05'

out_dir = data_dir + '/testoutput/modis_testing'
predir = out_dir + 'pre'
maindir = out_dir + 'main'
postdir = out_dir

cldsaddir = '/group_workspaces/cems/cloud_ecv/orac/sad_dir'
cldphs = ['WAT', 'ICE']
aersaddir = '/group_workspaces/cems/nceo_aerosolfire/gethomas/luts'
aerphs = ['AMZ', 'BOR', 'CER', 'AFR']  # Amazon, Boreal, Cerrado, Africa

# Call the pre-processing
print " **** Calling ORAC preprocessing on file: ****"
print "   " + mod_dir + '/' + l1name
pre_cmd = mod_dir + '/' + l1name \
          + ' -o ' + predir \
          + ' --batch ' \
          + ' --limit ' + pixel_limit

os.system('./orac_preproc.py ' + pre_cmd)

# Call the processing for the desired phases/types
# Path to the preprocessed files
msi_root = glob.glob(predir + '/*_' + yr + doy + hr + mn + '_*.msi.nc')
#msi_root = os.path.basename(msi_root[:len(msi_root) - 7])
print msi_root

# Call main first time  TODO Set channels to the right ones
proc_cmd = '-i ' + predir \
           + ' -o ' + maindir \
           + ' --sad_dir ' + cldsaddir \
           + ' --phase '

for phs in cldphs:
    print' **** Calling ORAC for type: ' + phs + ' ****'
    os.system('./orac_main.py ' + proc_cmd + phs) #+ ' ' + msi_root)

# Call main for aerosol phase, does this need to be done for MODIS?  Also set channels correctly
proc_cmd = '-i ' + predir \
           + ' -o ' + maindir \
           + ' --sad_dir ' + aersaddir \
           + ' --phase '
#        +' --extra_lines /home/users/gethomas/orac_code/aerosol_scripts_py/xtra_driver_lines.txt' \

for phs in aerphs:
    print' **** Calling ORAC for type: ' + phs + ' ****'
    os.system('./orac_main.py ' + proc_cmd + phs) #+ ' '+ msi_root)

# Call the post-processor
post_cmd = '-i ' + maindir + \
           ' -o ' + postdir \
           + '--verbose' \
           + ' --compress ' + msi_root \
           + ' --phases ' + ' '.join(aerphs) + ' ' + ' '.join(cldphs)
os.system('./orac_postproc.py ' + post_cmd)

# Clean-up
