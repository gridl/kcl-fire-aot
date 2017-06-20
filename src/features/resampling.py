'''
Resamples satellite data from one geographic
projection to another.
'''

import pyresample as pr
from matplotlib.path import Path
import numpy as np

import matplotlib.pyplot as plt


def get_roi_bounds(roi):
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
    return {'max_x': max_x,
            'min_x': min_x,
            'max_y': max_y,
            'min_y': min_y}


def get_mask(roi, rb):
    extent = [[x - rb['min_x'], y - rb['min_y']] for x, y in roi.extent]

    nx = rb['max_x'] - rb['min_x']
    ny = rb['max_y'] - rb['min_y']

    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T

    poly_verts = extent

    # apply mask
    path = Path(poly_verts)
    mask = path.contains_points(points)
    mask = mask.reshape((ny, nx))

    return mask


def resampler(orac_data, roi):


    # extract roi and geographic grids.  So the algorithm to do this
    # is as follows: find the bounding coordinates of the roi, extract
    # all the required data using the bounding coordinates (e.g. Lat, Lon
    # AOD).  Make it such that the plume coordinates start from zero (subtract
    # min_x from all x coords, and min_y from all y coords).  Now draw the mask
    # in the bounding box using these normalised coordinates.  So with this, we
    # will have extracted the data for the subset and drawn a mask around it.
    rb = get_roi_bounds(roi)
    lat = orac_data.variables['lat'][rb['min_y']:rb['max_y'],
                                     rb['min_x']:rb['max_x']]
    lon = orac_data.variables['lon'][rb['min_y']:rb['max_y'],
                                     rb['min_x']:rb['max_x']]
    aod = orac_data.variables['cot'][rb['min_y']:rb['max_y'],
                                     rb['min_x']:rb['max_x']]
    mask = get_mask(roi, rb)

    # get lats and lons for resampling grid
    lat_r = np.arange(np.min(lat), np.max(lat), 0.01)
    lon_r = np.arange(np.min(lon), np.max(lon), 0.01)
    lon_r, lat_r = np.meshgrid(lon_r, lat_r)
    lon_r = np.fliplr(lon_r)


    # build resampling swatch definitions
    def_a = pr.geometry.SwathDefinition(lons=lon, lats=lat)
    def_b = pr.geometry.SwathDefinition(lons=lon_r, lats=lat_r)

    # perform resampling
    resampled_aod = pr.kd_tree.resample_nearest(def_a,
                                                aod,
                                                def_b,
                                                radius_of_influence=9000,
                                                fill_value=0)
    resampled_mask = pr.kd_tree.resample_nearest(def_a,
                                                mask,
                                                def_b,
                                                radius_of_influence=9000,
                                                fill_value=0)

    # return resampled data
    plt.imshow(resampled_aod, interpolation='none')
    plt.show()

    plt.imshow(resampled_mask, interpolation='none', cmap='gray')
    plt.show()

