'''
Contains the various parameters for the data extraction
'''

import numpy as np
import sensor

# ----------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------

# MYD data download settings
myd_year = '2016'  # needs to be a string as used to acces the FTP server
myd_doy_range = np.arange(213, 273, 1)  # 213-273 92-153
myd_min_time = 1600
myd_max_time = 2200
myd_min_fires = 10  # minimum number of fires in the scene
myd_min_power = 1000 # minimum power of the fires in the scene
myd_min_szn = 85  # minimum solar zenith angle to ensure daylight obs

# lon0 for check if MODIS data intersects with geostationary footprint
if sensor.sensor == 'goes':
    lon_0 = -75.0  # GOES E lon_0

elif sensor.sensor == 'himawari':
    lon_0 = 140.7

# assumed geostationary sensor footprint size
footprint_radius = 5500000  # metres

# earth radius for distance calculations
earth_rad = 6371000  # in metres

# GOES order id's from CLASS; update each time running a new order
class_order_ids = ['2800309375', '2800310225', '2800310235', '2800310245', '2800310255', '2800310265', '2800310665',
                   '2800311125', '2800311135', '2800311145', '2800311155', '2800311165', '2800311175', '2800311185',
                   '2800311195', '2800311205', '2800311215', '2800311225', '2800311235', '2800311245', '2800311255',
                   '2800311265', '2800311275', '2800311295', '2800311295', '2800311315', '2800311345', '2800311355',
                   '2800311365', '2800311375', '2800311385', '2800312315', '2800312835', '2800312845', '2800312855',
                   '2800312865', '2800314415', '2800319285', '2800320745', '2800320755', '2800323995', '2800328005',
                   '2800328015', '2800328025', '2800328035', '2800328045', '2800328055', '2800328065', '2800328075',
                   '2800328085', '2800328095']



