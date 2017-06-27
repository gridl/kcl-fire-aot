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

# Geostationary sensor flag
geo_sensor = 'GOES'   # 'Himawari

if geo_sensor == 'GOES':
    pass
elif
    geo_sensor == 'Himawari':
    pass

# GOES order id's from CLASS; update each time running a new order
class_order_ids = ['2720282193', '2720283243', '2720283253', '2720285893',
                   '2720285903', '2720285923', '2720285913', '2720285973',
                   '2720285983', '2720286013', '2720283233', '2720283263',
                   '2720283273', '2720284513', '2720285883', '2720285933',
                   '2720285943', '2720285953', '2720285963', '2720285993',
                   '2720286003', '2720284213', '2720286023']
