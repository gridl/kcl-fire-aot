'''
Contain the various parameters for the feature generation
'''

# resampling parameters
padding = 3  # number of pixels around plume to additionally extract
res = 0.01  # grid resolution in degrees
radius_of_influence = 9000
fill_value = 0

# plume masking for smoke plume feature extraction
percentile = 50  # only pixels in the mask with band 8 DNs above this value are used in smoke pixel classification
