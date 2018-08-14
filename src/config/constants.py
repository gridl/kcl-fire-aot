import numpy as np


# FEATURES


# UTM grid size
utm_grid_size = 750

# Fast feature detector threshold
fast_threshold = 10

min_number_tracks = 2  # if features to track less than this, use alternative value
angular_limit = np.pi/4  # if angle of feature differs from the principle axis by more than this do not use