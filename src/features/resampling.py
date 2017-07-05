'''
Resamples satellite data from one geographic
projection to another.
'''

import numpy as np
import pyresample as pr
from matplotlib.path import Path

import src.config.features as feature_settings

def get_roi_bounds(roi):

    if type(roi['plume_extent']) is list:

        min_x = 99999
        max_x = 0
        min_y = 99999
        max_y = 0
        for x, y in roi.plume_extent:
            if x > max_x:
                max_x = x
            if x < min_x:
                min_x = x
            if y > max_y:
                max_y = y
            if y < min_y:
                min_y = y
        return {'max_x': max_x + feature_settings.padding,
                'min_x': min_x - feature_settings.padding,
                'max_y': max_y + feature_settings.padding,
                'min_y': min_y - feature_settings.padding}

    else:
        extent = roi['plume_extent'].values[0]
        return {'max_x': extent[1],
                'min_x': extent[0],
                'max_y': extent[3],
                'min_y': extent[2]}


def get_plume_mask(roi, rb):
    extent = [[x - rb['min_x'], y - rb['min_y']] for x, y in roi.plume_extent]

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


def resample_aod(orac_data, roi):


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
    if type(roi['extent']) is list:
        mask = get_plume_mask(roi, rb)

    # get lats and lons for resampling grid
    lat_r = np.arange(np.min(lat), np.max(lat), feature_settings.res)
    lon_r = np.arange(np.min(lon), np.max(lon), feature_settings.res)
    lon_r, lat_r = np.meshgrid(lon_r, lat_r)

    # build resampling swatch definitions
    def_a = pr.geometry.SwathDefinition(lons=lon, lats=lat)
    def_b = pr.geometry.SwathDefinition(lons=lon_r, lats=lat_r)

    # TODO perhaps change resampling from nearest to bilinear
    # perform resampling
    resampled_aod = pr.kd_tree.resample_nearest(def_a,
                                                aod,
                                                def_b,
                                                radius_of_influence=feature_settings.radius_of_influence,
                                                fill_value=feature_settings.fill_value)
    if type(roi['extent']) is list:
        resampled_mask = pr.kd_tree.resample_nearest(def_a,
                                                     mask,
                                                     def_b,
                                                     radius_of_influence=feature_settings.radius_of_influence,
                                                     fill_value=feature_settings.fill_value)

    # return resampled data0
    if type(roi['extent']) is list:
        return resampled_aod * resampled_mask, lon_r, lat_r, aod*mask, lat, lon
    else:
        return resampled_aod, lon_r, lat_r, aod, lat, lon