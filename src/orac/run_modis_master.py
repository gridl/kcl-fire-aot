#!/usr/bin/python2
import sys
import os
import re
import glob

# Set-up the paths and extract the date-time from the input file name
l1path=sys.argv[1]
l1name=os.path.basename(l1path)
l1dir=os.path.dirname(l1path)
# Extract date - this starts at character 46
yr=l1name[46:50]
mth=l1name[50:52]
day=l1name[52:54]
hr=l1name[54:56]
mn=l1name[56:58]

nrt_ws='/group_workspaces/cems/nrt_ecmwf_metop'
rsg_ws='/group_workspaces/cems/rsgnceo'
nceo_ws='/group_workspaces/cems2/nceo_generic'
orac_dir='/home/users/gethomas/orac_code'
hritscdir=rsg_ws+'/scratch/seviri_nrt/hrit'
preoutdir=rsg_ws+'/scratch/seviri_nrt/preproc_out/'+yr+'/'+mth+'/'+day
prooutdir=rsg_ws+'/scratch/seviri_nrt/proc_out/'+yr+'/'+mth+'/'+day
postoutdir=rsg_ws+'/scratch/seviri_nrt/post_out/'+yr+'/'+mth+'/'+day
cldsaddir='/group_workspaces/cems/cloud_ecv/orac/sad_dir'
cldphs=['WAT','ICE']
aersaddir='/group_workspaces/cems/nceo_aerosolfire/gethomas/luts'
aerphs=['A70','A71','A72','A73','A74','A74','A75','A76','A77','A78','A79']
#pixellimit='1800 2100 1809 2100' # test region near sub-satellite point
pixellimit='1701 2100 3201 3600' # Region centred on UK?
#posoutdir=rsg_ws+'/public/nrt_rolling/nrt_cloud_seviri_msg3'
if not(os.access(preoutdir, os.F_OK)):
    os.makedirs(preoutdir)
if not(os.access(prooutdir, os.F_OK)):
    os.makedirs(prooutdir)
if not(os.access(postoutdir, os.F_OK)):
    os.makedirs(postoutdir)

# Ensure ORAC libaries are able to be found
os.putenv('LD_LIBRARY_PATH', os.getenv('LD_LIBRARY_PATH')+':'
          + orac_dir +'/libraries/lib')

# Preprocessor script arguments
ecmwf_flag=4
data_in=nceo_ws+'/cloud_ecv/data_in'
# UoWis emissivity data 
emis_dir=data_in+'/emissivity'
# RTTOV emissivity atlas
atlas_dir=data_in+'/rttov/emis_data'
# MODIS land surface albedo/BRDF data
mcd43c3_dir=data_in+'/modis/MCD43C3_MODIS_Albedo_neu'
mcd43c1_dir=data_in+'/modis/MCD43C1_MODIS_BRDF_neu'
# RTTOV coefficient files
coef_dir=data_in+'/coeffs'
# ECMWF EMOS regridding library auxilary files
emos_dir=orac_dir+'/libraries/emos'
# Ice/snow coverage data
nise_dir=data_in+'/ice_snow'
# Full path+name to USGS DEM
usgs_file=data_in+'/dem/Aux_file_CM_SAF_AVHRR_GAC_ori_0.05deg.nc'

# Check for updated ECMWF forecast data for the current timeslot
# Set-up ECMWF paths (both for raw forecast files and netcdf versions)
ecmwf_in_1=nrt_ws+'/ecmwf/C3/'+yr+'/'+mth+'/'+day
ecmwf_in_2=nrt_ws+'/ecmwf/C4/'+yr+'/'+mth+'/'+day
ecmwf_out=rsg_ws+'/scratch/seviri_nrt/ecmwf/'+yr+'/'+mth+'/'+day
if not(os.access(ecmwf_out, os.F_OK)):
    os.makedirs(ecmwf_out)


# If we have new ECMWF data, do the conversion into netcdf
# Check what processed ECMWF files we already have
proecm=os.listdir(ecmwf_out)
# What data is available in the NRT archive
rawecm=os.listdir(ecmwf_in_1)
# We only want to keep the latest forecast file for each time slot. 
# The production date/time is stored in characters 3-10 in the ECMWF
# filename; characters 11-18 contain the target date-time, which are
# at three hour intervals...
tartimes=[mth+day+'000', mth+day+'030', mth+day+'060', mth+day+'090',
          mth+day+'120', mth+day+'150', mth+day+'180', mth+day+'210']
for tartime in tartimes:
    rawecm1 = [x for x in rawecm if re.match(r'\w{11,11}'+tartime+'\w',x)]
    proecm1 = [x for x in proecm if re.match(r'\w{11,11}'+tartime+'\w*',x)]
    process_ecmwf = ''
    delete_ecmwf = ''
    if len(rawecm1) > 0 and len(proecm1) > 0 and \
            (max(rawecm1))[3:11] > (max(proecm1))[3:11]:
        process_ecmwf=max(rawecm1)
        delete_ecmwf=max(proecm1)
    elif len(rawecm1) > 0 and len(proecm1) == 0:
        process_ecmwf=max(rawecm1)
    if len(process_ecmwf) > 0:
        os.chdir('/home/users/gethomas/python')
        os.system('./Convert_ECM_GRB2NC.py '+ecmwf_in_1+' '+ecmwf_out+' '
                  +process_ecmwf)
        if len(delete_ecmwf) > 0:
            os.remove(ecmwf_out+'/'+delete_ecmwf)

# Uncompress the SEVIRI HRIT data into a scratch dir
# Call xRITDecompress to uncompress the compressed hrit files. Note
# that xRITDecompress simply uncompresses into the CWD, so we again
# change working directory 
compressed_l1=glob.glob(l1dir+'/H-000-MSG*-'+yr+mth+day+hr+mn+'-C_')
os.chdir(hritscdir)
for f in compressed_l1:
    # Check if we already have the uncompressed files...
    tmp = os.path.basename(f)
    if not(os.access(tmp[0:len(tmp)-2]+'__',os.F_OK)):
        os.system('/home/users/gethomas/bin/xRITDecompress '+f)
# Now create links to the uncompressed files in the NRT archive directory
uncompressed_l1=glob.glob(l1dir+'/H-000-MSG*-'+yr+mth+day+hr+mn+'-__')
for f in uncompressed_l1:
    os.system('ln -s '+f+' '+hritscdir)

# Call the pre-processing
os.chdir(orac_dir+'/trunk/tools')
print " **** Calling ORAC preprocessing on file: ****"
print "   "+hritscdir+'/'+l1name
preproc_cmd=hritscdir+'/'+l1name +' -o '+ preoutdir \
    +' --batch --orac_dir '+ orac_dir+'/trunk' \
    +' --emis_dir '+ emis_dir +' --mcd43c3_dir '+ mcd43c3_dir \
    +' --mcd43c1_dir '+ mcd43c1_dir +' --atlas_dir '+ atlas_dir \
    +' --coef_dir '+ coef_dir +' --ecmwf_dir '+ ecmwf_out \
    +' --emos_dir '+ emos_dir +' --nise_dir '+ nise_dir \
    +' --usgs_file '+ usgs_file +' --day_flag 1 --ecmwf_nlevels 137' \
    +' --use_ecmwf_snow --ecmwf_flag 4 --skip_ecmwf_hr' \
    +' --channel_ids 1 2 3 4 7 9 10 --limit '+ pixellimit
os.system('./orac_preproc.py '+preproc_cmd)

# Call the processing for the desired phases/types
# Path to the preprocessed files
msi_root=glob.glob(preoutdir+'/*_'+yr+mth+day+hr+mn+'_*.msi.nc')
msi_root=os.path.basename(msi_root[_root=msi_root[:len(msi_root)-7]

proc_cmd='-i '+ preoutdir +' -o '+ prooutdir \
    +' --orac_dir '+ orac_dir+'/trunk --sad_dir '+ cldsaddir \
    +' --use_channel 1 1 1 0 0 1 1 --phase '
for phs in cldphs:
    print' **** Calling ORAC for type: '+phs+' ****'
    os.system('./orac_main.py '+ proc_cmd + phs +' '
              + msi_root)

proc_cmd='-i '+ preoutdir +' -o '+ prooutdir \
    +' --orac_dir '+ orac_dir+'/trunk --sad_dir '+ aersaddir \
    +' --use_channel 1 1 1 0 0 0 0 -a AppCld1L --ret_class ClsAerOx' \
    +' --phase '
#    +' --extra_lines /home/users/gethomas/orac_code/aerosol_scripts_py/xtra_driver_lines.txt' \
for phs in aerphs:
    print' **** Calling ORAC for type: '+phs+' ****'
    os.system('./orac_main.py '+ proc_cmd + phs +' '
              +msi_root)


# Call the post-processor
post_cmd='-i '+ prooutdir +' -o '+postoutdir \
    +' --orac_dir '+ orac_dir +'/trunk --verbose' \
    +' --compress '+ msi_root \
    +' --phases '+' '.join(aerphs)+' '+' '.join(cldphs)  
os.system('./orac_postproc.py '+ post_cmd)

# Clean-up
