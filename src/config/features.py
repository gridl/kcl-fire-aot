'''
Contain the various parameters for the feature generation
'''
import numpy as np

min_number_tracks = 2  # if features to track less than this, use alternative value
angular_limit = np.pi/4  # if angle of feature differs from current plume by more than this then do not use it