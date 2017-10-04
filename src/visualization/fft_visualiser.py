import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def display_map(f1_radiances_subset_reproj, utm_lats, utm_lons, fname):

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([80, 170, -45, 30])
    ax.stock_img()

    gridlines = ax.gridlines(draw_labels=True)

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
