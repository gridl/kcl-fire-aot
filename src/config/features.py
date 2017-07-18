'''
Contain the various parameters for the feature generation
'''

# resampling parameters
padding = 3  # number of pixels around plume to additionally extract
res = 0.01  # grid resolution in degrees
radius_of_influence = 9000
fill_value = 0

# review features flag
reduce_features = True
reduce_percentile = 20

# GLCM Texture measure settings
glcm_window_radius = 2