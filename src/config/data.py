'''
Contains the various parameters for the data extraction
'''

import numpy as np

# ----------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------

# MYD data download settings
myd_year = 2016
myd_doy_range = np.arange(92, 153, 1)  # 267-300
myd_min_time = 1600
myd_max_time = 2200
myd_min_fires = 10  # minimum number of fires in the scene
myd_max_fires = 1000 # minimum power of the fires in the scene

