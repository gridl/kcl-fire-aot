# Definitions of default values/paths for your local system

# ===== PATHS =====
stack_size = 250000

# Location of source code trunk from ORAC repository
orac_dir  = '/home/users/dnfisher/soft/orac_test/orac'
# Location of data directory from ORAC repository
data_dir  = '/group_workspaces/jasmin2/nceo_aerosolfire/data/orac_proc/viirs/sdr/'
# Path to library file used to compile ORAC
orac_lib  = '/home/users/dnfisher/soft/orac_test/orac/config/lib.ifort.cems'

# Directory of look-up tables
sad_dirs = ['/group_workspaces/jasmin2/nceo_aerosolfire/data/orac_proc/viirs/luts', ]

# Use ECMWF data from the BADC/CEMS archive
ecmwf_flag  = 2

# If any of these directories contain subdirectories named by date, please use
# the syntax of datatime.strftime() to denote the folder structure.
auxiliaries = {
    # Directory containing RTTOV emissivity atlas
    'atlas_dir'   : '/group_workspaces/cems2/nceo_generic/cloud_ecv/data_in/rttov/emis_data',
    # Directory of RTTOV instrument coefficient files
    'coef_dir'    : '/group_workspaces/cems2/nceo_generic/cloud_ecv/data_in/rttov121/coeffs',
    # Directory of MODIS emisivity retrievals
    'emis_dir'    : '/group_workspaces/cems2/nceo_generic/cloud_ecv/data_in/emissivity',
    # Directories of MODIS surface BRDF retrievals
    'mcd43c1_dir': '/group_workspaces/cems2/nceo_generic/cloud_ecv/data_in/modis/MCD43C1_MODIS_BRDF_neu/V006/',
    'mcd43c3_dir' : '/group_workspaces/cems2/nceo_generic/cloud_ecv/data_in/modis/MCD43C3_MODIS_Albedo_neu/V006',
    # To use ECMWF data from the BADC/CEMS archive (ecmwf_flag == 2), specifiy
    # where each of the three types of file are stored
    'ggam_dir'    : '/badc/ecmwf-era-interim/data/gg/am',
    'ggas_dir'    : '/badc/ecmwf-era-interim/data/gg/as',
    'spam_dir'    : '/badc/ecmwf-era-interim/data/sp/am',
    # To use a single ECMWF file (ecmwf_flag == 0), specify their location
    #'ecmwf_dir'   : '/local/storage/ecmwf/era_interim/%Y/%m/%d',
    # Directory of high-resolution ECMWF files
    'hr_dir'      : '/group_workspaces/cems2/nceo_generic/cloud_ecv/data_in/ecmwf',
    # Directory to store the EMOS interpolation file (CF_T0255_R0036000, which
    # will be generated if not already present)
    'emos_dir'    : '/home/users/dnfisher/data/emos_data#',
    # Directory of NSIDC ice/snow extent maps
    'nise_dir'    : '/group_workspaces/cems2/nceo_generic/cloud_ecv/data_in/ice_snow',
    # Directory of Ocean Colour CCI retrievals
    'occci_dir'   : '/group_workspaces/jasmin2/aerosol_cci/aux/oceancolour_cci/geographic_iop_v2.0',
    # File containing AATSR calibration coefficients
    'calib_file'  : '/group_workspaces/jasmin2/aerosol_cci/aux/aatsr_calibration/AATSR_VIS_DRIFT_V03-00.DAT',
    # File containing the USGS land-use map
    'usgs_file'   : '/group_workspaces/cems2/nceo_generic/cloud_ecv/data_in/dem/Aux_file_CM_SAF_AVHRR_GAC_ori_0.05deg.nc',
    # Pre-defined geometry for geostationary imager
    'prelsm_file' : '',
    'pregeo_file' : '',
    # Climatology of Swansea s and p parameters
    'swansea_dir' : '',
}


# ===== FILE HEADER INFORMATION =====

global_attributes = {
    "cfconvention" : "CF-1.4",
    "comments"     : "n/a",
    "email"        : 'daniel.fisher@kcl.ac.uk',
    "history"      : "n/a",
    "institute"    : "Uni",
    "keywords"     : "aerosol; cloud; optimal estimation",
    "license"      : "https://github.com/ORAC-CC/orac/blob/master/COPYING",
    "processor"    : "ORAC",
    "product_name" : "L2-AEROSOL-AER",
    "project"      : "ODA-WP3",
    "references"   : "doi:10.5194/amt-5-1889-2012",
    "summary"      : "n/a",
    "url"          : "http://github.com/ORAC-CC/orac",
}


# ===== FOLDER NAME PREFERENCES =====

log_dir = "log"
pre_dir = "pre"

extra_lines = {
    "land_extra" : "",
    "sea_extra"  : "",
    "cld_extra"  : "",
}


# ===== DEFAULT RETRIEVAL SETTINGS =====

retrieval_settings = {}
#
# # Permute a set of standard cloud retrievals over each sensor
# cld_retrievals = {
#     "wat_cld" : "--phase WAT --ret_class ClsCldWat --approach AppCld1L "
#                 "--sub_dir cld",
#     "ice_cld" : "--phase ICE --ret_class ClsCldIce --approach AppCld1L "
#                 "--sub_dir cld",
#     "wat_ice" : "--phase WAT --ret_class ClsCldWat --approach AppCld2L "
#                 "--multilayer ICE ClsCldIce --sub_dir cld",
# #    "ice_ice" : "--phase ICE --ret_class ClsCldIce --approach AppCld2L "
# #                "--multilayer ICE ClsCldIce --sub_dir cld"
# }
# cld_channels = {
#     "AATSR" : "--use_channels 2 3 4 5 6 7",
#     "ATSR2" : "--use_channels 2 3 4 5 6 7",
#     "AVHRR" : "--use_channels 1 2 3 4 5 6",
#     "MODIS" : "--use_channels 1 2 6 20 31 32",
#     "SEVIRI" : "--use_channels 1 2 3 4 9 10",
# }
# for sensor, channels in cld_channels.items():
#     retrieval_settings[sensor + "_C"] = [
#         channels + " " + ret for ret in cld_retrievals.values()
#     ]
#
# # Permute dual-view aerosol retrievals
aer_phases = range(70, 80)
# aer_dual_retrievals = {
#     "aer_ox" : "--phase A{:02d} --ret_class ClsAerOx --approach AppAerOx "
#                "--no_land --sub_dir sea",
#     "aer_sw" : "--phase A{:02d} --ret_class ClsAerSw --approach AppAerSw "
#                "--no_sea --sub_dir land",
# }
# aer_dual_channels = {
#     "AATSR" : "--use_channels 1 2 3 4 8 9 10 11",
#     "ATSR2" : "--use_channels 1 2 3 4 8 9 10 11",
# }
# for sensor, channels in aer_dual_channels.items():
#     retrieval_settings[sensor + "_A"] = [
#         channels + " " + ret.format(phs)
#         for ret in aer_dual_retrievals.values() for phs in aer_phases
#     ]
#
# Permute single-view aerosol retrievals
aer_single_retrievals = {
    "aer_o1" : "--phase A{:02d} --ret_class ClsAerOx --approach AppAerO1 "
               "--no_land --sub_dir sea",
}
aer_single_channels = {
    "AVHRR" : "--use channels 1 2 3",
    "MODIS" : "--use_channels 4 1 2 6",
    "SEVIRI" : "--use_channels 1 2 3",
}
for sensor, channels in aer_single_channels.items():
    retrieval_settings[sensor + "_A"] = [
        channels + " " + ret.format(phs)
        for ret in aer_single_retrievals.values() for phs in aer_phases
    ]
#
# # Joint aerosol-cloud retrievals
# for sensor, channels in cld_channels.items():
#     retrieval_settings[sensor + "_J"] = [
#         channels + " " + ret for ret in cld_retrievals.values()
#     ]
# for sensor, channels in aer_dual_channels.items():
#     retrieval_settings[sensor + "_J"].extend([
#         channels + " " + ret.format(phs)
#         for ret in aer_dual_retrievals.values() for phs in aer_phases
#     ])
# for sensor, channels in aer_single_channels.items():
#     retrieval_settings[sensor + "_J"].extend([
#         channels + " " + ret.format(phs)
#         for ret in aer_single_retrievals.values() for phs in aer_phases
#     ])
#
# # Default to cloud retrievals
# for sensor in cld_channels.keys():
#     retrieval_settings[sensor] = retrieval_settings[sensor + "_C"]
#
#
# ===== DEFAULT CHANNELS FOR EACH SENSOR =====

channels = {
    "AATSR" : (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    "ATSR2" : (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    "AVHRR" : (1, 2, 3, 4, 5, 6),
    "MODIS" : (1, 2, 6, 20, 31, 32),
    "SEVIRI" : (1, 2, 3, 4, 9, 10),
}


# ===== REGRESSION TEST SETTINGS =====

# Fields to ignore during regression testing
atts_to_ignore = ('L2_Processor_Version', 'Production_Time', 'File_Name')
vars_to_accept = ('costjm', 'costja', 'niter')

# Tollerances in regression testing
rtol = 1e-7 # Maximum permissable relative difference
atol = 1e-7 # Maximum permissable absolute difference

# Filters to apply regression test warnings
# (see https://docs.python.org/2/library/warnings.html#the-warnings-filter)
warn_filt = {
    'FieldMissing'    : 'once',
    'InconsistentDim' : 'error',
    'RoundingError'   : 'once',
    'Acceptable'      : 'once',
}


# ===== BATCH PROCESSING SETTINGS =====

# The Oxford Yau cluster uses QSUB
from pyorac.batch import bsub as batch

# Arguments that should be included in every call
batch_values = {
    'email'    : global_attributes["email"],
    'priority' : 10000,
    'queue'    : 'priority-eodg',
}

# Initialisation of script file
batch_script = """#!/bin/bash --noprofile
"""