# Definitions of default values/paths for your local system

from os import environ

# Control flags
ecmwf_flag  = '2'

# Paths
aer_sad_dir = '/group_workspaces/cems/nceo_aerosolfire/luts'
sad_dir     = '/group_workspaces/cems2/nceo_generic/cloud_ecv/data_in/sad_dir/modis_WAT/'
atlas_dir   = '/group_workspaces/cems2/nceo_generic/cloud_ecv/data_in/rttov/emis_data'
calib_file  = '/group_workspaces/cems/aerosol_cci/aux/aatsr_calibration/AATSR_VIS_DRIFT_V03-00.DAT'
coef_dir    = '/group_workspaces/cems2/nceo_generic/cloud_ecv/data_in/coeffs'
emis_dir    = '/group_workspaces/cems2/nceo_generic/cloud_ecv/data_in/emissivity'
emos_dir    = '/home/users/dnfisher/soft/orac/libraries/emos_data'
ggam_dir    = '/badc/ecmwf-era-interim/data/gg/am'
ggas_dir    = '/badc/ecmwf-era-interim/data/gg/as'
hr_dir      = '/group_workspaces/cems2/nceo_generic/cloud_ecv/data_in/ecmwf'
mcd43c3_dir = '/group_workspaces/cems2/nceo_generic/cloud_ecv/data_in/modis/MCD43C3_MODIS_Albedo_neu/'
mcd43c1_dir = '/group_workspaces/cems2/nceo_generic/cloud_ecv/data_in/modis/MCD43C1_MODIS_BRDF_neu/'
nise_dir    = '/group_workspaces/cems2/nceo_generic/cloud_ecv/data_in/ice_snow'
occci_dir   = '/group_workspaces/cems/aerosol_cci/aux/oceancolour_cci/geographic_iop_v2.0'
orac_lib    = environ['ORAC_LIB']
orac_trunk  = '/home/users/dnfisher/soft/orac/trunk'
spam_dir    = '/badc/ecmwf-era-interim/data/sp/am'
usgs_file   = '/group_workspaces/cems2/nceo_generic/cloud_ecv/data_in/dem/Aux_file_CM_SAF_AVHRR_GAC_ori_0.05deg.nc'

# Header details
email       = 'daniel.fisher@kcl.ac.uk'
institute   = 'Kings College London'
project     = 'KCL-NCEO'


# ======= Folder Names Preferences =======

pre_dir = 'pre'
main_dir = 'main'
land_dir = 'land'
sea_dir = 'sea'
log_dir = 'log'

# ======= Regression Test Settings =======

# Fields to ignore during regression testing
atts_to_ignore = ('L2_Processor_Version', 'Production_Time', 'File_Name')
vars_to_accept = ('costjm', 'costja', 'niter')

# Tollerances in regression testing
rtol = 1e-7 # Maximum permissable relative difference
atol = 1e-8 # Maximum permissable absolute difference

# Filters to apply regression test warnings
# (see https://docs.python.org/2/library/warnings.html#the-warnings-filter)
warn_filt = {}
warn_filt['FieldMissing']    = 'once'
warn_filt['InconsistentDim'] = 'error'
warn_filt['RoundingError']   = 'once'
warn_filt['Acceptable']      = 'ignore'

# Queuing system setup...
from orac_batch import bsub as batch

# Arguments that should be included in every call
batch_values = {'email'    : 'daniel.fisher@kcl.ac.uk'}

# Initialisation of script file
batch_script = """#!/bin/bash --noprofile
"""
