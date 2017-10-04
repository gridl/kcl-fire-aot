import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely.geometry as sgeom
import numpy as np
from matplotlib.patheffects import Stroke



def display_map(f1_radiances_subset_reproj, utm_resampler, fname):

    lons, lats = utm_resampler.area_def.get_lonlats()
    crs = ccrs.PlateCarree()
    extent = [np.min(lons), np.max(lons), np.min(lats), np.max(lats)]

    padding = 0.5
    padded_extent = [np.min(lons) - padding, np.max(lons) + padding, np.min(lats) - padding, np.max(lats) + padding]

    ax = plt.axes(projection=crs)
    ax.set_extent(padded_extent)

    gridlines = ax.gridlines(draw_labels=True)
    plt.imshow(f1_radiances_subset_reproj, transform=crs, extent=extent, origin='upper', cmap='gray')


    # Create an inset GeoAxes showing the location of the Solomon Islands.
    sub_ax = plt.axes([0.7, 0.625, 0.2, 0.2], projection=ccrs.PlateCarree())
    sub_ax.set_extent([110, 180, -50, 10])

    # Make a nice border around the inset axes.
    effect = Stroke(linewidth=4, foreground='wheat', alpha=0.5)
    sub_ax.outline_patch.set_path_effects([effect])

    # Add the land, coastlines and the extent of the Solomon Islands.
    sub_ax.add_feature(cartopy.feature.LAND)
    sub_ax.coastlines()
    extent_box = sgeom.box(extent[0], extent[2], extent[1], extent[3])
    sub_ax.add_geometries([extent_box], ccrs.PlateCarree(), color='none',
                          edgecolor='blue', linewidth=2)

    plt.show()


# def display_map(f1_radiances_subset_reproj, utm_lats, utm_lons, fname):
#     pass
#
#
# def display_flow(flow, f1_radiances_subset_reproj, utm_lats, utm_lons, fname):
#
#     h, w = img.shape[:2]
#     y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1)
#     fx, fy = flow[y, x].T
#     lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
#     lines = np.int32(lines + 0.5)
#     for (x1, y1), (x2, y2) in lines:
#         cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
#
#     plt.imshow(vis)
