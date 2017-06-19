'''
Resamples satellite data from one geographic
projection to another.
'''

import pyresample


def roi_bounds(roi):
    min_x = 99999
    max_x = 0
    min_y = 99999
    max_y = 0
    for x, y in roi.extent:
        if x > max_x:
            max_x = x
        if x < min_x:
            min_x = x
        if y > max_y:
            max_y = y
        if y < min_y:
            min_y = y
    return [max_x, min_x, max_y, min_y]



def resampler(orac_data, roi):

    # extract roi and geographic grids.  So the algorithm to do this
    # is as follows: find the bounding coordinates of the roi, extract
    # all the required data using the bounding coordinates (e.g. Lat, Lon
    # AOD).  Make it such that the plume coordinates start from zero (subtract
    # min_x from all x coords, and min_y from all y coords).  Now draw the mask
    # in the bounding box using these normalised coordinates.  So with this, we
    # will have extracted the data for the subset and drawn a mask around it.


    # build resampling grid

    # perform resampling

    # return resampled data

