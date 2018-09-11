import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy
import cartopy.crs as ccrs
import shapely.geometry as sgeom
import numpy as np
from matplotlib.patheffects import Stroke
import cv2
import scipy.interpolate as interpolate
from datetime import datetime
import re

import os

import src.data.readers.load_hrit as load_hrit
import src.config.filepaths as fp
import src.features.fre_to_tpm.viirs.ftt_fre as ff


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

    plt.savefig(os.path.join(fp.path_to_plume_tracking_visualisations, 'maps', fname), bbox_inches='tight', dpi=300)
    plt.close()


def display_masked_map(img, fires, plume_points, utm_resampler,
                       plume_head, plume_tail,
                       flow_vector, projected_flow_vector,
                       path, fname):

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

    if fires is not None:
        for f in fires:
            fx, fy = f.xy
            plt.plot(fx, fy, 'ro', markersize=2)

    for v1, v2 in zip(verts[:-1], verts[1:]):
        plt.plot([v1[0], v2[0]], [v1[1], v2[1]], 'k-', linewidth=1)

    plt.plot(plume_head[0], plume_head[1], 'b.', markersize=2)
    #plt.plot(plume_tail[0], plume_tail[1], 'r>', markersize=2)

    # now plot the flow vector progression
    for i, fv in enumerate(flow_vector):

        pv_tail = projected_flow_vector[i]

        fv = utm_resampler.resample_point_to_geo(fv[1], fv[0])
        pv_tail = utm_resampler.resample_point_to_geo(pv_tail[1], pv_tail[0])

        plt.plot([pv_tail[0], fv[0]], [pv_tail[1], fv[1]], 'k-', linewidth=1)
        plt.plot(fv[0], fv[1], 'b.', markersize=2)

    #plt.show()
    plt.savefig(os.path.join(path, fname), bbox_inches='tight', dpi=300)
    plt.close()


def display_masked_map_first(img, fires, plume_points, utm_resampler,
                             plume_head, plume_tail, path, fname):

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

    if fires is not None:
        for f in fires:
            fx, fy = f.xy
            plt.plot(fx, fy, 'ro', markersize=2)

    for v1, v2 in zip(verts[:-1], verts[1:]):
        plt.plot([v1[0], v2[0]], [v1[1], v2[1]], 'k-', linewidth=1)

    plt.plot(plume_head[0], plume_head[1], 'b.', markersize=2)
    #plt.plot(plume_tail[0], plume_tail[1], 'r>', markersize=2)

    #plt.show()
    plt.savefig(os.path.join(path, fname), bbox_inches='tight', dpi=300)
    plt.close()


def save_im(im, path, fname):
    ax = plt.axes()
    ax.imshow(im, cmap='gray')

    output_fname = fname +'.jpg'
    plt.savefig(os.path.join(path, output_fname), bbox_inches='tight', dpi=300)
    plt.close()




def draw_flow_map(img, utm_resampler, plume_points, plume_head, plume_tail, flow, path, fname, stage_name, step=2):

    x, y = plume_points.minimum_rotated_rectangle.exterior.xy
    verts = [utm_resampler.resample_point_to_geo(y, x) for (x, y) in zip(x, y)]

    plume_tail = np.array(plume_tail['tail'].coords)[0]
    plume_head = np.array(plume_head['head'].coords)[0]
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

    # gridlines = ax.gridlines(draw_labels=True)
    plt.imshow(img, transform=crs, extent=extent, origin='upper', cmap='gray')
    plt.plot([plume_head[0], plume_tail[0]], [plume_head[1], plume_tail[1]], 'k-', linewidth=1)
    plt.plot(plume_head[0], plume_head[1], 'b.', markersize=2)

    for v1, v2 in zip(verts[:-1], verts[1:]):
        plt.plot([v1[0], v2[0]], [v1[1], v2[1]], 'k-', linewidth=1)


    # now plot the flow vectors
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    dx, dy = flow[y,x].T

    lons1 = lons[y,x]
    lats1 = lats[y,x]

    # interpolate sub_pixel lats
    h = np.arange(h)
    w = np.arange(w)
    f_lat = interpolate.interp2d(h, w, lats.flatten())
    f_lon = interpolate.interp2d(h, w, lons.flatten())
    lons2 = np.array([f_lon(y[i]+dy[i], x[i]+dx[i])[0] for i in xrange(x.size)])
    lats2 = np.array([f_lat(y[i]+dy[i], x[i]+dx[i])[0] for i in xrange(x.size)])

    plt.plot([lons1, lons2], [lats1, lats2], 'r-', linewidth=0.1)
    plt.plot(lons1, lats1, 'r.', markersize=0.1)

    #plt.show()
    output_fname = stage_name + fname.split('/')[-1].split('.')[0] + '.jpg'
    plt.savefig(os.path.join(path, output_fname), bbox_inches='tight', dpi=300)
    plt.close()


def draw_flow(img, flow, path, fname, stage_name, step=2):

    plt.close('all')
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    x_shift = x+fx
    y_shift = y+fy

    mask = (x_shift > 0) & (x_shift < w) & (y_shift > 0) & (y_shift < h)
    x = x[mask]
    y = y[mask]
    x_shift = x_shift[mask]
    y_shift = y_shift[mask]

    ax = plt.axes()
    ax.imshow(img, cmap='gray')
    #ax.quiver(x, y, fx, fy, scale=200, color='red')
    ax.plot((x,x_shift), (y, y_shift), 'r-', linewidth=0.25)
    ax.plot(x,y, 'r.', markersize=0.25)

    output_fname = stage_name + fname.split('/')[-1].split('.')[0] + '.jpg'
    plt.savefig(os.path.join(path, output_fname), bbox_inches='tight', dpi=300)
    plt.close()


def extract_observation(f, bb, segment):
    # load geostationary files for the segment
    rad_segment_1, _ = load_hrit.H8_file_read(os.path.join(fp.path_to_himawari_imagery, f))

    # load for the next segment
    f_new = f.replace('S' + str(segment).zfill(2), 'S' + str(segment + 1).zfill(2))
    rad_segment_2, _ = load_hrit.H8_file_read(os.path.join(fp.path_to_himawari_imagery, f_new))

    # concat the himawari files
    rad = np.vstack((rad_segment_1, rad_segment_2))

    # extract geostationary image subset using adjusted bb and rescale to 8bit
    rad_bb = rad[bb['min_y']:bb['max_y'], bb['min_x']:bb['max_x']]

    return rad_bb


def load_image(geostationary_fname, bbox, min_geo_segment):
    return extract_observation(geostationary_fname, bbox, min_geo_segment)


def reproject_image(im, geo_dict, plume_geom_utm):
        return plume_geom_utm['utm_resampler_plume'].resample_image(im, geo_dict['geostationary_lats_subset'],
                                                                        geo_dict['geostationary_lons_subset'])


def run_plot(flow, geostationary_fnames, plume_geom_geo, pp, bbox, him_segment, him_geo_dict, plume_geom_utm,
             plume_head, plume_tail, plume_points, utm_resampler,
             plume_logging_path, n):

    plume_tail = np.array(plume_tail['tail'].coords)[0]
    plume_head = np.array(plume_head['head'].coords)[0]

    utm_flow_vectors = []
    utm_plume_projected_flow_vectors = [plume_tail]

    for i in xrange(n):

        fname = geostationary_fnames[i]
        t = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", geostationary_fnames[i]).group(), '%Y%m%d_%H%M')
        fires = ff.fire_locations_for_plume_roi(plume_geom_geo, pp['frp_df'], t)
        im_subset = load_image(geostationary_fnames[i], bbox, him_segment)
        im_subset_reproj = reproject_image(im_subset, him_geo_dict, plume_geom_utm)

        if i == 0 :
            display_masked_map_first(im_subset_reproj,
                                     fires,
                                     plume_points,
                                     utm_resampler,
                                     plume_head,
                                     plume_tail,
                                     plume_logging_path,
                                     'subset_' + fname.split('/')[-1].split('.')[0] + '.jpg')

        else:
            utm_flow_vectors += [utm_plume_projected_flow_vectors[-1] + flow]
            utm_plume_projected_flow_vectors += [utm_plume_projected_flow_vectors[-1] + flow]
            display_masked_map(im_subset_reproj,
                               fires,
                               plume_points,
                               utm_resampler,
                               plume_head,
                               plume_tail,
                               utm_flow_vectors,
                               utm_plume_projected_flow_vectors,
                               plume_logging_path,
                               'subset_' + fname.split('/')[-1].split('.')[0] + '.jpg')

        # set a max numbner of plot and then break
        max_plots = 6
        if i == max_plots:
            break


def plot_plume_data(sat_data_utm, plume_data_utm, plume_bounding_box, plume_logging_path):
    plt.imshow(sat_data_utm['viirs_png_utm'][plume_bounding_box['min_y']:plume_bounding_box['max_y'],
               plume_bounding_box['min_x']:plume_bounding_box['max_x'], :])
    plt.savefig(os.path.join(plume_logging_path, 'viirs_plume_tcc.png'), bbox_inches='tight')
    plt.close()
    plt.imshow(plume_data_utm['viirs_aod_utm_plume'], vmin=0, vmax=2)
    cb = plt.colorbar()
    cb.set_label('VIIRS IP AOD')
    plt.savefig(os.path.join(plume_logging_path, 'viirs_plume_aod.png'), bbox_inches='tight')
    plt.close()

    ax = plt.imshow(plume_data_utm['viirs_flag_utm_plume'])
    cmap = cm.get_cmap('Set1', 4)
    ax.set_cmap(cmap)
    cb = plt.colorbar()
    cb.set_label('VIIRS IP AOD Flag')
    plt.savefig(os.path.join(plume_logging_path, 'viirs_plume_flag.png'), bbox_inches='tight')
    plt.close()

    plt.imshow(plume_data_utm['orac_aod_utm_plume'], vmin=0, vmax=2)
    cb = plt.colorbar()
    cb.set_label('VIIRS ORAC AOD')
    plt.savefig(os.path.join(plume_logging_path, 'viirs_plume_orac.png'), bbox_inches='tight')
    plt.close()

    plt.imshow(plume_data_utm['orac_cost_utm_plume'], vmax=10, cmap='plasma')
    cb = plt.colorbar()
    cb.set_label('VIIRS ORAC AOD COST')
    plt.savefig(os.path.join(plume_logging_path, 'viirs_plume_orac_cost.png'), bbox_inches='tight')
    plt.close()