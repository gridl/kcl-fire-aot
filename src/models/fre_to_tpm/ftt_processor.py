
import src.models.fre_to_tpm.ftt_utils as ut
import src.config.filepaths as fp

import matplotlib.pyplot as plt

def main():

    # load in static data
    #frp_df = ut.read_frp_df(fp.path_to_himawari_frp)
    plume_df = ut.read_plume_polygons(fp.path_to_smoke_plume_polygons_csv)
    lc_data = []
    geostationary_lats, geostationary_lons = [], [] # the geostationary lats and lons

    # set up arrays to hold data
    fre = []
    tpm = []
    lc = []

    # itereate over the plumes
    for i, plume in plume_df.iterrows():

        # load plume datasets
        orac_aod = []
        myd04_aod = []

        # construct plume coordinate data
        try:
            plume_bounding_box = ut.construct_bounding_box(plume)
            plume_lats, plume_lons = ut.read_modis_geo_subset(fp.path_to_modis_l1b, plume, plume_bounding_box)
            plume_polygon = ut.construct_polygon(plume, plume_bounding_box, plume_lats, plume_lons)
            plume_mask = ut.construct_plume_mask(plume, plume_bounding_box)
        except Exception, e:
            print e
            continue

        # subset ORAC and MYD04 datasets
        orac_aod_subset = []
        myd04_aod_subset = []

        # set up utm resampler that we use to resample all data to utm
        utm_resampler = ut.utm_resampler(plume_lats, plume_lons, 1000)

        # resample the datasets (mask, orac_aod, MYD04)
        utm_plume_polygon = ut.reproject_polygon(plume_polygon, utm_resampler)
        utm_plume_mask = utm_resampler.resample(plume_mask, plume_lats, plume_lons)
        utm_orac_aod_subset = []
        utm_modis_aod_subset = []

        # get FRP integration start and stop times
        start_time, stop_time = ut.find_integration_start_stop_times(plume_polygon)

        # get the variables of interest
        fre.append(ut.compute_fre(plume_polygon, frp_df, start_time, stop_time))
        tpm.append(ut.compute_aod(orac_aod, myd04_aod, plume_bounding_box, plume_mask, plume_lats, plume_lons))
        lc.append(0)

    # split data based on lc type

    # compute models





if __name__=="__main__":
    main()
