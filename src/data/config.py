import numpy as np
import countries


myd = {'year': 2012,
       'doy_range': np.arange(267, 300, 1),  # 267-300
       'min_time': 1600,
       'max_time': 2200,
       'min_fires': 10,  # minimum number of fires in the scene
       'min_power': 1000,  # minimum power of the fires in the scene
       }

cc = countries.CountryChecker('TM_WORLD_BORDERS-0.3.shp')
