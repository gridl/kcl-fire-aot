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


def display_masked_map(img, plume_points, utm_resampler,
                       plume_head, plume_tail,
                       flow_vector, projected_flow_vector,
                       fname):

    x, y = plume_points.minimum_rotated_rectangle.exterior.xy
    verts = [utm_resampler.resample_point_to_geo(y, x) for (x, y) in zip(x, y)]

    plume_head = utm_resampler.resample_point_to_geo(plume_head[1], plume_head[0])
    plume_tail = utm_resampler.resample_point_to_geo(plume_tail[1], plume_tail[0])

    lons, lats = utm_resampler.area_def.get_lonlats()
    crs = ccrs.PlateCarree()
    extent = [np.min(lons), np.max(lons), np.min(lats), np.max(lats)]

    u_padding = -0
    l_padding = -0
    padded_extent = [np.min(lons) - u_padding, np.max(lons) + u_padding,
                     np.min(lats) - u_padding, np.max(lats) + l_padding]

    ax = plt.axes(projection=crs)
    ax.set_extent(padded_extent)

    ax.coastlines(resolution='50m', color='black', linewidth=1)

    #gridlines = ax.gridlines(draw_labels=True)
    plt.imshow(img, transform=crs, extent=extent, origin='upper', cmap='gray')
    plt.plot([plume_head[0], plume_tail[0]], [plume_head[1], plume_tail[1]], 'k-', linewidth=1)


    for v1, v2 in zip(verts[:-1], verts[1:]):
        plt.plot([v1[0], v2[0]], [v1[1], v2[1]], 'k-', linewidth=1)

    plt.plot(plume_head[0], plume_head[1], 'r.', markersize=2)
    #plt.plot(plume_tail[0], plume_tail[1], 'r>', markersize=2)

    # now plot the flow vector progression
    for i, fv in enumerate(flow_vector):

        pv_tail = projected_flow_vector[i]

        fv = utm_resampler.resample_point_to_geo(fv[1], fv[0])
        pv_tail = utm_resampler.resample_point_to_geo(pv_tail[1], pv_tail[0])

        plt.plot([pv_tail[0], fv[0]], [pv_tail[1], fv[1]], 'k-', linewidth=1)
        plt.plot(fv[0], fv[1], 'r.', markersize=2)

    #plt.show()
    plt.savefig(os.path.join(fp.path_to_him_visualisations, 'plumes', fname), bbox_inches='tight', dpi=300)
    plt.close()



def display_flow(x_flow, y_flow, f1_radiances, utm_resampler, fname):

    x_flow[np.abs(x_flow) < 1] = 0
    y_flow[np.abs(y_flow) < 1] = 0

    lons, lats = utm_resampler.area_def.get_lonlats()
    crs = ccrs.PlateCarree()
    extent = [np.min(lons), np.max(lons), np.min(lats), np.max(lats)]

    u_padding = 0.25
    l_padding = 1
    padded_extent = [np.min(lons) - u_padding, np.max(lons) + u_padding,
                     np.min(lats) - u_padding, np.max(lats) + l_padding]

    ax1 = plt.subplot(1, 3, 1, projection=ccrs.PlateCarree())
    ax1.coastlines('50m')
    ax1.set_extent(padded_extent, ccrs.PlateCarree())
    ax1.imshow(f1_radiances, transform=crs, extent=extent, origin='upper', cmap='gray')

    ax2 = plt.subplot(1, 3, 2, projection=ccrs.PlateCarree())
    ax2.coastlines('50m')
    ax2.set_extent(padded_extent, ccrs.PlateCarree())
    im2 = ax2.imshow(x_flow, transform=crs, extent=extent, origin='upper', cmap='PuOr', vmin=-2, vmax=2)
    plt.colorbar(im2, ax=ax2)

    ax3 = plt.subplot(1, 3, 3, projection=ccrs.PlateCarree())
    ax3.coastlines('50m')
    ax3.set_extent(padded_extent, ccrs.PlateCarree())
    im3 = ax3.imshow(y_flow, transform=crs, extent=extent, origin='upper', cmap='PuOr', vmin=-2, vmax=2)
    plt.colorbar(im3, ax=ax3)

    #plt.show()
    plt.savefig(os.path.join(fp.path_to_him_visualisations, 'flows', fname), bbox_inches='tight', dpi=600)
    plt.close()

