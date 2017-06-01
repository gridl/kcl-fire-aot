import matplotlib.pyplot as plt
import pyresample as pr
from netCDF4 import Dataset
from pyhdf.SD import SD, SDC
from datetime import datetime
import glob


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
    aod = ds.select("Deep_Blue_Aerosol_Optical_Depth_550_Land").get()
    return aod, lat, lon


def main():

    f_names = ['KCL-NCEO-L2-CLOUD-CLD-MODIS_ORAC_AQUA_201404051740_R4591WAT.primary.nc',
               'KCL-NCEO-L2-CLOUD-CLD-MODIS_ORAC_AQUA_201404051940_R4591WAT.primary.nc',
               'KCL-NCEO-L2-CLOUD-CLD-MODIS_ORAC_AQUA_201404101815_R4591WAT.primary.nc',
               'KCL-NCEO-L2-CLOUD-CLD-MODIS_ORAC_AQUA_201405041905_R4591WAT.primary.nc']

    plume_extent_number = [0,1,8,2]  # this plume extent index for agiven file with multiple plumes

    path_to_orac = '/Users/dnf/Projects/kcl-fire-aot/data/processed/orac_proc/2014'
    path_to_mod04 = '/Users/dnf/Projects/kcl-fire-aot/data/external/mod_aod'
    path_to_extent = '/Users/dnf/Projects/kcl-fire-aot/data/processed/plume_masks/plume_extents.txt'

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

        # load modis
        primary_time = get_primary_time(f)
        aod_file = glob.glob(path_to_mod04 + '/*' + primary_time + '*')[0]
        mod_aod, mod_lat, mod_lon = read_aod_mod(aod_file)

        # get ORAC plume extent
        orac_aod_sub = orac_aod[pc[0]:pc[1], pc[2]:pc[3]]
        orac_lat_sub = orac_lat[pc[0]:pc[1], pc[2]:pc[3]]
        orac_lon_sub = orac_lon[pc[0]:pc[1], pc[2]:pc[3]]

        plt.imshow(orac_aod_sub)
        plt.show()

        # resample modis AOD to ORAC grid
        grid_def = pr.geometry.GridDefinition(lons=orac_lon_sub, lats=(orac_lat_sub))
        aod_swath_def = pr.geometry.SwathDefinition(lons=mod_lon, lats=mod_lat)

        resampled_mod_aod = pr.kd_tree.resample_nearest(aod_swath_def,
                                                        mod_aod,
                                                        grid_def,
                                                        radius_of_influence=50000)

        # display
        plt.imshow(resampled_mod_aod)
        plt.show()


if __name__ == "__main__":
    main()
