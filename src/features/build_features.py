import glob
import os

import pandas as pd
import numpy as np

import src.data.readers as readers
import config
import resampling

import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import collections


def lc_subset():

    gt = ds.GetGeoTransform()

    #TODO move this outside of this function, just open file in here.
    # we dont want to load the whole image, just the part of interest
    lon_start = -2  #  run from W to E
    lon_stop = 2

    lat_start = 52  # runs from N to S
    lat_stop = 50

    x_start = (lon_start - gt[0]) / gt[1]
    x_stop = (lon_stop - gt[0]) / gt[1]
    x_range = int(x_stop - x_start)

    y_start = (lat_start - gt[3]) / gt[5]
    y_stop = (lat_stop - gt[3]) / gt[5]
    y_range = int(y_stop - y_start)

    x = np.arange(0, x_range, 1)
    y = np.arange(0, y_range, 1)
    grids = np.meshgrid(x, y)

    lons = lon_start + grids[0] * gt[1]
    lats = lat_start + grids[1] * gt[5]

    lc_data = ds.GetRasterBand(1).ReadAsArray(int(round(x_stop)),
                                              int(round(y_stop)),
                                              x_range,
                                              y_range)


def get_orac_fname(orac_file_path, plume):
    y = plume.filename[10:14]
    doy = plume.filename[14:17]
    time = plume.filename[18:22]
    return glob.glob(os.path.join(orac_file_path, y, doy, 'main', '*' + time + '*.primary.nc'))[0]


def collocate_fires(lats, lons, resampled_plume, orac_filename, frp_data):
    fname = orac_filename.split('/')[-1]
    y = fname[38:42]
    m = fname[42:44]
    d = fname[44:46]
    frp_data_subset = frp_data[(frp_data.year == int(y)) &
                               (frp_data.month == int(m)) &
                               (frp_data.day == int(d))]

    frp_data_subset = frp_data_subset[(frp_data_subset['latitude'] > np.min(lats)) &
                                      (frp_data_subset['latitude'] < np.max(lats)) &
                                      (frp_data_subset['lontitude'] > np.min(lons)) &
                                      (frp_data_subset['lontitude'] > np.max(lons))]
    return frp_data_subset


def compute_fre(plume_causing_fires):
    # we approximate FRE on sin function?
    pass


def plot_plume(resampled_plume, lons, lats):

    masked_plume = np.ma.masked_array(resampled_plume, resampled_plume <=0)
    m = Basemap(llcrnrlon=np.min(lons), llcrnrlat=np.min(lats),
                urcrnrlon=np.max(lons), urcrnrlat=np.max(lats))
    m.pcolormesh(lons, lats, masked_plume)

    m.drawparallels(np.arange(-90., 120., 0.05))
    m.drawmeridians(np.arange(0., 420., 0.05))

    plt.show()


def plot_fires_spatial(fire_df):

    df_subset = fire_df[(fire_df['latitude'] > -90) &
                        (fire_df['latitude'] < 90) &
                        (fire_df['lontitude'] > -180) &
                        (fire_df['lontitude'] < 180)]


    # count fires on a 0.5 degree grid or similar, the plot the bin counts
    lons = np.arange(-180,180,0.5)
    lats = np.arange(-90,90,0.5)*-1
    lons, lats = np.meshgrid(lons, lats)

    fires = np.zeros(lons.shape)
    for i, row in fire_df.iterrows():

        try:
            iy = int((row['latitude'] - 90) * -2)
            ix = int((row['lontitude'] + 180) * 2)
            fires[iy, ix] += 1
        except Exception, e:
            print e, row['latitude'], row['lontitude']
            continue
    fires = np.ma.masked_array(fires, fires == 0)

    m = Basemap(projection='geos', lon_0=-75, resolution='i')
    m.drawcoastlines()
    # draw parallels and meridians.
    m.drawparallels(np.arange(-90., 120., 30.))
    m.drawmeridians(np.arange(0., 420., 30.))
    m.pcolormesh(lons, lats, fires,
                 norm=colors.LogNorm(),
                 latlon=True)
    cbar = plt.colorbar()
    cbar.set_label('No. Fire Obs.')


    plt.show()


def plot_fires_mean_frp(fire_df):

    df_subset = fire_df[(fire_df['latitude'] > -90) &
                        (fire_df['latitude'] < 90) &
                        (fire_df['lontitude'] > -180) &
                        (fire_df['lontitude'] < 180)]


    # count fires on a 0.5 degree grid or similar, the plot the bin counts
    lons = np.arange(-180,180,0.5)
    lats = np.arange(-90,90,0.5)*-1
    lons, lats = np.meshgrid(lons, lats)

    frp = np.zeros(lons.shape)
    count = np.zeros(lons.shape)
    for i, row in fire_df.iterrows():

        try:
            iy = int((row['latitude'] - 90) * -2)
            ix = int((row['lontitude'] + 180) * 2)
            count[iy, ix] += 1
            if row['FRP'] > 100000000:
                print 'false frp', row['FRP']
            else:
                frp[iy, ix] += row['FRP']
        except Exception, e:
            print e, row['latitude'], row['lontitude']
            continue


    count_zero = count == 0
    count[count_zero] += 1
    frp /= count
    count[count_zero] -= 1

    frp = np.ma.masked_array(frp, count == 0)
    frp = np.ma.masked_array(frp, frp < 1)
    frp[frp > 1000] = 1000

    m = Basemap(projection='geos', lon_0=-75, resolution='i')
    m.drawcoastlines()
    # draw parallels and meridians.
    m.drawparallels(np.arange(-90., 120., 30.))
    m.drawmeridians(np.arange(0., 420., 30.))
    m.pcolormesh(lons, lats, frp,
                 norm=colors.LogNorm(),
                 latlon=True)
    cbar = plt.colorbar()
    cbar.set_label('Mean FRP (MW)')


    plt.show()


def plot_fires_temporal(fire_df):
    fig, ax = plt.subplots()
    hhmm = fire_df['hhmm'].values.astype('int')
    hhmm_counts = collections.Counter(hhmm)
    hhmm_counts = collections.OrderedDict(sorted(hhmm_counts.items()))
    df = pd.DataFrame.from_dict(hhmm_counts, orient='index')
    df.plot(kind='bar', legend=False, grid='off')
    plt.xlabel('Obs. Time')
    plt.ylabel('Sample Count')
    plt.show()



def main():

    # create df to hold the outputs
    output_df = pd.DataFrame()

    # read in non-plume specific files
    frp_data = readers.read_goes_frp(config.goes_frp_file_path)
    lc_data = readers.read_lc(config.lc_file_path)

    # plot fires
    #plot_fires_spatial(frp_data)
    #plot_fires_mean_frp(frp_data)
    #plot_fires_temporal(frp_data)


    # read in plume dataframes
    plume_masks = readers.read_plume_data(config.plume_mask_file_path)
    plume_backgrounds = readers.read_plume_data(config.plume_background_file_path)

    # rename the extents for easier processing
    plume_masks.rename(columns={'plume_extent': 'extent'}, inplace=True)
    plume_backgrounds.rename(columns={'bg_extent': 'extent'}, inplace=True)

    # set up plot for fires coincident with plumes
    glons = np.arange(-180, 180, 0.5)
    glats = np.arange(-90, 90, 0.5) * -1
    glons, glats = np.meshgrid(glons, glats)
    cleaned_fires = np.zeros(glons.shape)

    # set up list to hold times of fires
    fire_times = []


    # iterate over each plume in the plume mask dataframe
    modis_filename = ''
    for index, plume in plume_masks.iterrows():

        # load in orac file.  If the plumes is from
        # a different modis file, then load in the
        # correct ORAC processed file
        if plume.filename != modis_filename:
            modis_filename = plume.filename
            try:

                orac_filename = get_orac_fname(config.orac_file_path, plume)
                orac_data = readers.read_orac(orac_filename)

                # extract background data for plume
                background = plume_backgrounds[plume_backgrounds.plume_id == plume.plume_id]

                # resample plume AOD to specified grid resolution
                resampled_plume, r_lons, r_lats, aod, lats, lons = resampling.resampler(orac_data, plume)

                # resample background AOD to specified grid resolution
                resampled_background, _, _, _, _, _ = resampling.resampler(orac_data, background)

                # get fires contained within plume (using geo coords and date time, if none then continue)
                plume_causing_fires = collocate_fires(r_lats, r_lons, resampled_plume, orac_filename, frp_data)

                if plume_causing_fires.empty:
                    continue

                for i, row in plume_causing_fires.iterrows():
                    try:
                        iy = int((row['latitude'] - 90) * -2)
                        ix = int((row['lontitude'] + 180) * 2)
                        cleaned_fires[iy, ix] += 1

                        fire_times.append(int(row['hhmm']))

                    except Exception, e:
                        print e, row['latitude'], row['lontitude']
                        continue

                # plot the plume
                #plot_plume(aod, lons, lats)

                # for the fires in the plume attempt to compute the FRE

                # compute tpm for the plume


                # get fire landsurface type


                # insert all data into dataframe

            except Exception, e:
                print e
                continue

    # now plot the fires after collocated with digitised plumes
    cleaned_fires = np.ma.masked_array(cleaned_fires, cleaned_fires == 0)
    m = Basemap(projection='geos', lon_0=-75, resolution='i')
    m.drawcoastlines()
    # draw parallels and meridians.
    m.drawparallels(np.arange(-90., 120., 30.))
    m.drawmeridians(np.arange(0., 420., 30.))
    m.pcolormesh(glons, glats, cleaned_fires,
                 norm=colors.LogNorm(),
                 latlon=True)
    cbar = plt.colorbar()
    cbar.set_label('No. Colloc. Fire Obs.')
    plt.show()

    hhmm_counts = collections.Counter(fire_times)
    hhmm_counts = collections.OrderedDict(sorted(hhmm_counts.items()))
    df = pd.DataFrame.from_dict(hhmm_counts, orient='index')
    df.plot(kind='bar', legend=False, grid='off')
    plt.xlabel('Obs. Time')
    plt.ylabel('Sample Count')
    plt.show()



if __name__ == "__main__":
    main()
