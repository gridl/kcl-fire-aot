import matplotlib.pyplot as plt
import pyresample as pr
from netCDF4 import Dataset
from pyhdf.SD import SD, SDC
from datetime import datetime
import glob
import numpy as np
from mpl_toolkits.basemap import Basemap

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def get_primary_time(primary_file):
    tt = datetime.strptime(primary_file.split('_')[-2], '%Y%m%d%H%M').timetuple()
    primary_datestring = str(tt.tm_year) + \
                         str(tt.tm_yday).zfill(3) + \
                         '.' + \
                         str(tt.tm_hour).zfill(2) + \
                         str(tt.tm_min).zfill(2)
    return primary_datestring


def read_aod_orac(primary_file):
    ds = Dataset(primary_file)
    aod = ds.variables['cot']
    lat = ds.variables['lat']
    lon = ds.variables['lon']
    return aod, lat, lon


def read_aod_mod(aod_file):
    ds = SD(aod_file, SDC.READ)
    lat = ds.select("Latitude").get()
    lon = ds.select("Longitude").get()
    aod = ds.select("Deep_Blue_Aerosol_Optical_Depth_550_Land")
    aod = aod.get() * aod.attributes()['scale_factor']
    return aod, lat, lon


def read_myd021km(local_filename):
    return SD(local_filename, SDC.READ)


def fcc_myd021km_250(mod_data):
    mod_params_500 = mod_data.select("EV_500_Aggr1km_RefSB").attributes()
    ref_500 = mod_data.select("EV_500_Aggr1km_RefSB").get()

    mod_params_250 = mod_data.select("EV_250_Aggr1km_RefSB").attributes()
    ref_250 = mod_data.select("EV_250_Aggr1km_RefSB").get()

    r = (ref_250[0, :, :] - mod_params_250['radiance_offsets'][0]) * mod_params_250['radiance_scales'][
        0]  # 2.1 microns
    g = (ref_500[1, :, :] - mod_params_500['radiance_offsets'][1]) * mod_params_500['radiance_scales'][
        1]  # 0.8 microns
    b = (ref_500[0, :, :] - mod_params_500['radiance_offsets'][0]) * mod_params_500['radiance_scales'][
        0]  # 0.6 microns

    r = image_histogram_equalization(r)
    g = image_histogram_equalization(g)
    b = image_histogram_equalization(b)

    r = np.round((r * (255 / np.max(r))) * 1).astype('uint8')
    g = np.round((g * (255 / np.max(g))) * 1).astype('uint8')
    b = np.round((b * (255 / np.max(b))) * 1).astype('uint8')



    rgb = np.dstack((r, g, b))
    return rgb


def main():

    f_names = ['KCL-NCEO-L2-CLOUD-CLD-MODIS_ORAC_AQUA_201405201905_R4591WAT.primary.nc',
               'KCL-NCEO-L2-CLOUD-CLD-MODIS_ORAC_AQUA_201408022125_R4591WAT.primary.nc',
               'KCL-NCEO-L2-CLOUD-CLD-MODIS_ORAC_AQUA_201408062100_R4591WAT.primary.nc',
               'KCL-NCEO-L2-CLOUD-CLD-MODIS_ORAC_AQUA_201408101700_R4591WAT.primary.nc',
               'KCL-NCEO-L2-CLOUD-CLD-MODIS_ORAC_AQUA_201408141635_R4591WAT.primary.nc']

    plume_extent_number = [0,0,0,8,0]  # this plume extent index for agiven file with multiple plumes

    path_to_orac = '/Users/dnf/Projects/kcl-fire-aot/data/processed/orac_proc/2014'
    path_to_mod04 = '/Users/dnf/Projects/kcl-fire-aot/data/external/mod_aod'
    path_to_extent = '/Users/dnf/Projects/kcl-fire-aot/data/processed/plume_masks/plume_extents.txt'
    path_to_l1b = '/Users/dnf/Projects/kcl-fire-aot/data/raw/l1b'
    path_to_output = '/Users/dnf/Projects/kcl-fire-aot/data/processed/mod_comp/'

    # open plume extent text file
    with open(path_to_extent, 'r') as f:
        file_extents = f.readlines()

    # iterate over files to visualise
    for f, ind in zip(f_names, plume_extent_number):

        # get plume indexes
        extent_list = []
        for l in file_extents:
            if f in l:
                extent_list.append(l)
        pc = []  # plume coordinate
        for pix_pos in extent_list[ind].split()[1:]:
            pc.append(int(pix_pos))

        # load orac
        orac_file = glob.glob(path_to_orac + '/*/*/' + f)[0]
        orac_aod, orac_lat, orac_lon = read_aod_orac(orac_file)

        # load modis AOD
        primary_time = get_primary_time(f)
        aod_file = glob.glob(path_to_mod04 + '/*' + primary_time + '*')[0]
        mod_aod, mod_lat, mod_lon = read_aod_mod(aod_file)

        # load MODIS L1B
        l1b_file = glob.glob(path_to_l1b + '/*' + primary_time + '*')[0]
        mod_data = read_myd021km(l1b_file)
        rgb = fcc_myd021km_250(mod_data)

        # get ORAC plume extent
        e = 20
        rgb_sub = rgb[pc[0]-e:pc[1]+e, pc[2]-e:pc[3]+e, :]
        orac_aod_sub = orac_aod[pc[0]-e:pc[1]+e, pc[2]-e:pc[3]+e]
        orac_lat_sub = orac_lat[pc[0]-e:pc[1]+e, pc[2]-e:pc[3]+e]
        orac_lon_sub = orac_lon[pc[0]-e:pc[1]+e, pc[2]-e:pc[3]+e]


        # resample modis AOD to ORAC grid
        orac_aod_def = pr.geometry.SwathDefinition(lons=orac_lon_sub, lats=orac_lat_sub)
        mod_aod_def = pr.geometry.SwathDefinition(lons=mod_lon, lats=mod_lat)

        resampled_mod_aod = pr.kd_tree.resample_nearest(mod_aod_def,
                                                        mod_aod,
                                                        orac_aod_def,
                                                        radius_of_influence=100000)

        resampled_mod_aod = np.ma.masked_array(resampled_mod_aod, resampled_mod_aod < 0)

        # lets plot in on a map fiurst set up axes
        fig, axes = plt.subplots(1, 3, figsize=(12,4))



        # now set up map
        m = Basemap(projection='merc', llcrnrlon=np.min(orac_lon_sub), urcrnrlon=np.max(orac_lon_sub),
                    llcrnrlat=np.min(orac_lat_sub), urcrnrlat=np.max(orac_lat_sub), resolution='c')

        parallels = np.arange(-90., 90., 0.5)
        meridians = np.arange(-130., 45., 0.5)
        m.drawparallels(parallels, labels=[True, False, True, False], ax=axes[0])
        mer = m.drawmeridians(meridians, labels=[False, True, False, True], ax=axes[0])

        for me in mer:
            try:
                mer[me][1][0].set_rotation(45)
            except:
                pass

        m.drawparallels(parallels, ax=axes[1])
        mer = m.drawmeridians(meridians, labels=[False, True, False, True], ax=axes[1])

        for me in mer:
            try:
                mer[me][1][0].set_rotation(45)
            except:
                pass

        m.drawparallels(parallels, ax=axes[2])
        mer = m.drawmeridians(meridians, labels=[False, True, False, True], ax=axes[2])

        for me in mer:
            try:
                mer[me][1][0].set_rotation(45)
            except:
                pass

        img = rgb_sub[:,:,2]
        im0 = m.pcolormesh(orac_lon_sub, orac_lat_sub, img, ax=axes[0], latlon=True, cmap='gray')
        im1 = m.pcolormesh(orac_lon_sub, orac_lat_sub, orac_aod_sub, ax=axes[1], latlon=True)
        im2 = m.pcolormesh(orac_lon_sub, orac_lat_sub, resampled_mod_aod, ax=axes[2], latlon=True)


        divider0 = make_axes_locatable(axes[0])
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        cbar0 = plt.colorbar(im0, cax=cax0)

        divider1 = make_axes_locatable(axes[1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cbar1 = plt.colorbar(im1, cax=cax1)


        divider2 = make_axes_locatable(axes[2])
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        cbar2 = plt.colorbar(im2, cax=cax2)

        for ax, title in zip(axes, ['MYD DNs', 'ORAC AOD', 'DB AOD']):
            ax.set_title(title)


        plt.savefig(path_to_output + f.split('.')[0] + 'comp.png', bbox_inches='tight')
        plt.close()




if __name__ == "__main__":
    main()
