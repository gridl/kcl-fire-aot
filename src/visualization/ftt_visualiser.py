import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import shapely.geometry as sgeom
import numpy as np
from matplotlib.patheffects import Stroke
import cv2

import os

import src.config.filepaths as fp


def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


def display_map(f1_radiances_subset_reproj, utm_resampler, fname):

    lons, lats = utm_resampler.area_def.get_lonlats()
    crs = ccrs.PlateCarree()
    extent = [np.min(lons), np.max(lons), np.min(lats), np.max(lats)]

    u_padding = 0.25
    l_padding = 1
    padded_extent = [np.min(lons) - u_padding, np.max(lons) + u_padding,
                     np.min(lats) - u_padding, np.max(lats) + l_padding]

    ax = plt.axes(projection=crs)
    ax.set_extent(padded_extent)

    ax.coastlines(resolution='50m', color='black', linewidth=1)

    gridlines = ax.gridlines(draw_labels=True)
    ax.imshow(f1_radiances_subset_reproj, transform=crs, extent=extent, origin='upper', cmap='gray')


    # Create an inset GeoAxes showing the location
    sub_ax = plt.axes([0.5, 0.66, 0.2, 0.2], projection=ccrs.PlateCarree())
    sub_ax.set_extent([95, 145, -20, 10])

    # Make a nice border around the inset axes.
    effect = Stroke(linewidth=4, foreground='wheat', alpha=0.5)
    sub_ax.outline_patch.set_path_effects([effect])

    # Add the land, coastlines and the extent of the Solomon Islands.
    sub_ax.add_feature(cartopy.feature.LAND)
    sub_ax.coastlines()
    extent_box = sgeom.box(extent[0], extent[2], extent[1], extent[3])
    sub_ax.add_geometries([extent_box], ccrs.PlateCarree(), color='none',
                          edgecolor='blue', linewidth=2)

    plt.savefig(os.path.join(fp.path_to_him_visualisations, 'maps', fname), bbox_inches='tight', dpi=300)
    plt.close()


def display_masked_map(f1_radiances_subset_reproj, mask, utm_resampler,
                       plume_head, plume_tail,
                       flow_vector, projected_flow_vector,
                       fname):

    f1_radiances_subset_reproj_masked = np.ma.masked_array(f1_radiances_subset_reproj, ~mask)

    plume_head = utm_resampler.resample_point_to_geo(plume_head[1], plume_head[0])
    plume_tail = utm_resampler.resample_point_to_geo(plume_tail[1], plume_tail[0])

    lons, lats = utm_resampler.area_def.get_lonlats()
    crs = ccrs.PlateCarree()
    extent = [np.min(lons), np.max(lons), np.min(lats), np.max(lats)]

    u_padding = -0.1
    l_padding = -0.1
    padded_extent = [np.min(lons) - u_padding, np.max(lons) + u_padding,
                     np.min(lats) - u_padding, np.max(lats) + l_padding]

    ax = plt.axes(projection=crs)
    ax.set_extent(padded_extent)

    ax.coastlines(resolution='50m', color='black', linewidth=1)

    #gridlines = ax.gridlines(draw_labels=True)
    plt.imshow(f1_radiances_subset_reproj_masked, transform=crs, extent=extent, origin='upper', cmap='gray')
    plt.plot(plume_head[0], plume_head[1], 'r>', markersize=2)
    plt.plot(plume_tail[0], plume_tail[1], 'ro', markersize=2)
    plt.plot([plume_head[0], plume_tail[0]], [plume_head[1], plume_tail[1]], 'r--', linewidth=0.5)

    # now plot the flow vector progression
    for i, fv in enumerate(flow_vector):

        pv_tail = projected_flow_vector[i]
        pv_head = projected_flow_vector[i+1]

        # convert flow vector into lat lon
        fv = utm_resampler.resample_point_to_geo(fv[1], fv[0])

        # get the positions along the plume vector
        pv_tail = utm_resampler.resample_point_to_geo(pv_tail[1], pv_tail[0])
        pv_head = utm_resampler.resample_point_to_geo(pv_head[1], pv_head[0])

        #plt.plot(fv[0], fv[1], 'r>')
        #plt.plot(pv_tail[0], pv_tail[1], 'ro')

        plt.plot(pv_head[0], pv_head[1], 'r>', markersize=1)
        #plt.plot(pv_tail[0], pv_tail[1], 'ro')

        plt.plot([pv_tail[0], fv[0]], [pv_tail[1], fv[1]], 'r--', linewidth=0.5)
        #plt.plot([pv_tail[0], pv_head[0]], [pv_tail[1], pv_head[1]], 'r--', linewidth=0.5)


    #plt.show()
    plt.savefig(os.path.join(fp.path_to_him_visualisations, 'plumes', fname), bbox_inches='tight', dpi=300)
    plt.close()


def draw_flow(img, flow, step=8):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 0, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 0, 0), -1)
    return vis


def display_flow(flow, f1_radiances_subset_reproj, utm_resampler, fname):

    # vis = f1_radiances_subset_reproj.copy()
    # cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
    # cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
    # draw_str(vis, (20, 20), 'track count: %d' % len(tracks))
    # plt.imshow(vis, cmap='gray')
    # plt.show()

    lons, lats = utm_resampler.area_def.get_lonlats()
    crs = ccrs.PlateCarree()
    extent = [np.min(lons), np.max(lons), np.min(lats), np.max(lats)]

    u_padding = 0.25
    l_padding = 1
    padded_extent = [np.min(lons) - u_padding, np.max(lons) + u_padding,
                     np.min(lats) - u_padding, np.max(lats) + l_padding]

    ax = plt.axes(projection=crs)
    ax.set_extent(padded_extent)

    ax.coastlines(resolution='50m', color='black', linewidth=1)

    #gridlines = ax.gridlines(draw_labels=True)

    # Create an inset GeoAxes showing the location
    sub_ax = plt.axes([0.5, 0.66, 0.2, 0.2], projection=ccrs.PlateCarree())
    sub_ax.set_extent([95, 145, -20, 10])

    # Make a nice border around the inset axes.
    effect = Stroke(linewidth=4, foreground='wheat', alpha=0.5)
    sub_ax.outline_patch.set_path_effects([effect])

    # Add the land, coastlines and the extent of the Solomon Islands.
    sub_ax.add_feature(cartopy.feature.LAND)
    sub_ax.coastlines()
    extent_box = sgeom.box(extent[0], extent[2], extent[1], extent[3])
    sub_ax.add_geometries([extent_box], ccrs.PlateCarree(), color='none',
                          edgecolor='blue', linewidth=2)


    ax.imshow(draw_flow(f1_radiances_subset_reproj, flow), transform=crs, extent=extent, origin='upper', cmap='gray')

    plt.savefig(os.path.join(fp.path_to_him_visualisations, 'flows', fname), bbox_inches='tight', dpi=600)
    plt.close()

