
import src.models.fre_to_tpm.ftt_utils as ut
import src.config.filepaths as fp

def main():

    # load in static data
    frp_df = ut.read_frp_df(fp.path_to_himawari_frp)
    plume_df = ut.read_plume_polygons(fp.path_to_smoke_plume_polygons_csv)
    lc_data = []

    # set up arrays to hold data
    fre = []
    tpm = []
    lc = []

    # itereate over the plumes
    for i, plume in plume_df.iterrows():

        # load plume datasets
        lats, lons = ut.read_geo(fp.path_to_modis_l1b, plume)
        orac_aod = []
        myd04_aod = []

        # construct plume coordinate data
        plume_polygon = ut.construct_polygon(plume, lats, lons)
        plume_bounding_box = ut.construct_bounding_box(plume)
        plume_mask = ut.construct_plume_mask(plume, plume_bounding_box)

        # get FRP integration start and stop times
        start_time, stop_time = []

        # get the variables of interest
        fre.append(ut.compute_fre(plume_polygon, frp_df, start_time, stop_time))
        tpm.append(ut.compute_aod(orac_aod, myd04_aod, plume_bounding_box, plume_mask, lats, lons))
        lc.append(0)

    # split data based on lc type

    # compute models





if __name__=="__main__":
    main()
