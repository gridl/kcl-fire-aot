import logging
import os

import numpy as np
import pandas as pd
import scipy.misc as misc
import scipy.interpolate as interpolate
from sklearn.gaussian_process import GaussianProcess
import seaborn as sns

import src.features.fre_to_tpm.viirs.ftt_utils as ut

import src.config.filepaths as fp
import src.config.constants as constants
import src.data.readers.load_hrit as load_hrit

import matplotlib.pyplot as plt


def proc_params():
    d = {}

    d['plot'] = True
    d['resampled_pix_size'] = 750  # size of UTM grid in meters
    d['plume_df'] = ut.read_plume_polygons(fp.path_to_smoke_plume_polygons_viirs_csv)
    return d


def resample_satellite_datasets(plume, current_timestamp, pp):

    d = {}

    try:
        viirs_aod_data = ut.load_viirs(fp.path_to_viirs_aod, current_timestamp, plume.filename)
        orac_aod_data = ut.load_orac(fp.path_to_viirs_orac, current_timestamp)
        if pp['plot']:
            d['viirs_png_utm'] = misc.imread(os.path.join(fp.path_to_viirs_sdr_resampled, plume.filename))
    except Exception, e:
        logger.info('Could not load AOD data with error: ' + str(e))
        return None

    # set up resampler
    utm_rs = ut.utm_resampler(orac_aod_data.variables['lat'][:],
                                           orac_aod_data.variables['lon'][:],
                                           constants.utm_grid_size)

    # get the mask for the lats and lons and apply
    orac_aod = ut.orac_aod(orac_aod_data)
    viirs_null_mask = np.ma.getmask(orac_aod)
    masked_lats = np.ma.masked_array(utm_rs.lats, viirs_null_mask)
    masked_lons = np.ma.masked_array(utm_rs.lons, viirs_null_mask)

    # resample all the datasets to UTM
    d['viirs_aod_utm'] = utm_rs.resample_image(ut.viirs_aod(viirs_aod_data), masked_lats, masked_lons, fill_value=0)
    d['viirs_flag_utm'] = utm_rs.resample_image(ut.viirs_flags(viirs_aod_data), masked_lats, masked_lons, fill_value=0)
    d['orac_aod_utm'] = utm_rs.resample_image(orac_aod, masked_lats, masked_lons, fill_value=0)
    d['orac_cost_utm'] = utm_rs.resample_image(ut.orac_cost(orac_aod_data), masked_lats, masked_lons, fill_value=0)
    d['lats'] = utm_rs.resample_image(utm_rs.lats, masked_lats, masked_lons, fill_value=0)
    d['lons'] = utm_rs.resample_image(utm_rs.lons, masked_lats, masked_lons, fill_value=0)
    return d


def setup_plume_data(plume, ds_utm):
    d = {}
    try:
        # get plume extent geographic data (bounding box in in UTM as plume extent is UTM)
        d['plume_bounding_box'] = ut.construct_bounding_box(plume.plume_extent)
        d['plume_lats'] = ut.subset_data(ds_utm['lats'], d['plume_bounding_box'])
        d['plume_lons'] = ut.subset_data(ds_utm['lons'], d['plume_bounding_box'])

        # get plume vector geographic data
        vector_lats, vector_lons = ut.extract_subset_geo_bounds(plume.plume_vector, d['plume_bounding_box'],
                                                                d['plume_lats'], d['plume_lons'])
        # get plume polygon geographic data
        poly_lats, poly_lons = ut.extract_subset_geo_bounds(plume.plume_extent, d['plume_bounding_box'],
                                                            d['plume_lats'], d['plume_lons'])


        # get plume mask
        d['plume_mask'] = ut.construct_mask(plume.plume_extent, d['plume_bounding_box'])

        # setup shapely objects for plume geo data
        d['plume_vector'] = ut.construct_shapely_vector(vector_lats, vector_lons)
        d['plume_points'] = ut.construct_shapely_points(poly_lats, poly_lons)
        d['plume_polygon'] = ut.construct_shapely_polygon(poly_lats, poly_lons)

        d['background_bounding_box'] = ut.construct_bounding_box(plume.background_extent)
        d['background_mask'] = ut.construct_mask(plume.background_extent, d['background_bounding_box'])

        return d
    except Exception, e:
        logger.error(str(e))
        return None


def subset_sat_data_to_plume(sat_data_utm, plume_geom_geo):
    d = {}
    d['viirs_aod_utm_plume'] = ut.subset_data(sat_data_utm['viirs_aod_utm'], plume_geom_geo['plume_bounding_box'])
    d['viirs_flag_utm_plume'] = ut.subset_data(sat_data_utm['viirs_flag_utm'], plume_geom_geo['plume_bounding_box'])
    d['orac_aod_utm_plume'] = ut.subset_data(sat_data_utm['orac_aod_utm'], plume_geom_geo['plume_bounding_box'])
    d['orac_cost_utm_plume'] = ut.subset_data(sat_data_utm['orac_cost_utm'], plume_geom_geo['plume_bounding_box'])

    d['viirs_aod_utm_background'] = ut.subset_data(sat_data_utm['viirs_aod_utm'],
                                                   plume_geom_geo['background_bounding_box'])
    d['viirs_flag_utm_background'] = ut.subset_data(sat_data_utm['viirs_flag_utm'],
                                                    plume_geom_geo['background_bounding_box'])
    d['orac_aod_utm_background'] = ut.subset_data(sat_data_utm['orac_aod_utm'],
                                                  plume_geom_geo['background_bounding_box'])
    d['orac_cost_utm_background'] = ut.subset_data(sat_data_utm['orac_cost_utm'],
                                                   plume_geom_geo['background_bounding_box'])
    return d


def resample_plume_geom_to_utm(plume_geom_geo):
    d = {}
    d['utm_resampler_plume'] = ut.utm_resampler(plume_geom_geo['plume_lats'],
                                                plume_geom_geo['plume_lons'],
                                                constants.utm_grid_size)
    d['utm_plume_points'] = ut.reproject_shapely(plume_geom_geo['plume_points'], d['utm_resampler_plume'])
    d['utm_plume_polygon'] = ut.reproject_shapely(plume_geom_geo['plume_polygon'], d['utm_resampler_plume'])
    d['utm_plume_vector'] = ut.reproject_shapely(plume_geom_geo['plume_vector'], d['utm_resampler_plume'])
    return d


def extract_combined_aod_full_plume(plume_data_utm,
                                    plume_mask):

    # combine plume mask with VIIRS good and ORAC good
    viirs_good = plume_data_utm['viirs_flag_utm_plume'] <= 1
    orac_good = plume_data_utm['orac_cost_utm_plume'] <= 3
    viirs_plume_mask = plume_mask & viirs_good  # viirs contribtuion
    orac_plume_mask = plume_mask & (orac_good & ~viirs_good)  # ORAC contribution

    # extract the aod data
    combined_aod = np.zeros(viirs_good.shape) -999
    combined_aod[viirs_plume_mask] = plume_data_utm['viirs_aod_utm_plume'][viirs_plume_mask]
    combined_aod[orac_plume_mask] = plume_data_utm['orac_aod_utm_plume'][orac_plume_mask]
    return combined_aod


def eval_interpolation_methods(plumes, masks):
    pc = 0.25  # % of points to set to null (i.e. excluded from sampling)
    error_rbf = []
    error_gp = []
    error_mean = []

    for p, m in zip(plumes, masks):

        # build the interpolation grid
        y = np.linspace(0, 1, p.shape[0])
        x = np.linspace(0, 1, p.shape[1])
        xx, yy = np.meshgrid(x, y)

        # remove a subset of samples
        r = np.arange(0, p.shape[0])
        c = np.arange(0, p.shape[1])
        cc, rr = np.meshgrid(c, r)
        null_locs = np.random.randint(p.size, size=p.size)[:int(p.size*pc)]
        rr_subset = rr.flatten()[null_locs]
        cc_subset = cc.flatten()[null_locs]
        t = p.copy()

        # get null mask
        t[rr_subset, cc_subset] = -999
        vals = t != -999

        # create interpolated grid (can extend to various methods)
        rbf = interpolate.Rbf(xx[vals], yy[vals], t[vals], function='linear')
        interp_rbf = rbf(xx, yy)

        gp = GaussianProcess(theta0=0.1, thetaL=.001, thetaU=1., nugget=0.01)
        gp.fit(X=np.column_stack([xx[vals], yy[vals]]), y=t[vals])
        rr_cc_as_cols = np.column_stack([xx.flatten(), yy.flatten()])
        interp_gp = gp.predict(rr_cc_as_cols).reshape(t.shape)

        # extract points associated with plume using the mask
        p_valid = p != -999
        observed = p[p_valid & m]
        predicted_mean = np.mean(observed)
        predicted_rbf = interp_rbf[p_valid & m]
        predicted_gp = interp_gp[p_valid & m]

        # for non-null samples compute percentage error predicted / observed
        ratio_rbf = np.abs(1 - (predicted_rbf / observed))
        ratio_gp = np.abs(1 - (predicted_gp / observed))
        ratio_mean = np.abs(1 - (predicted_mean / observed))

        # append to error
        error_rbf.append(ratio_rbf)
        error_gp.append(ratio_gp)
        error_mean.append(ratio_mean)

    # plot the cumulative histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    n_bins = 50
    lab=True
    for e0, e1, e2 in zip(error_rbf, error_gp, error_mean):
        if lab:
            ax.hist(e0, n_bins, normed=1, histtype='step', cumulative=True, label='rbf', color='m', alpha=0.5)
            ax.hist(e1, n_bins, normed=1, histtype='step', cumulative=True, label='gp', color='c', alpha=0.5)
            ax.hist(e2, n_bins, normed=1, histtype='step', cumulative=True, label='mean', color='y', alpha=0.5)
            lab=False
        else:
            ax.hist(e0, n_bins, normed=1, histtype='step', cumulative=True, color='m', alpha=0.5)
            ax.hist(e1, n_bins, normed=1, histtype='step', cumulative=True, color='c', alpha=0.5)
            ax.hist(e2, n_bins, normed=1, histtype='step', cumulative=True, color='y', alpha=0.5)
    plt.ylabel('Cumulative Dist')
    plt.xlabel('Abs. Ratio (|1-pred/obs|)')
    plt.xlim((-0.1, 1.1))
    plt.legend(loc=4)
    plt.show()





def main():

    # setup the data dict to hold all data
    pp = proc_params()
    previous_timestamp = ''

    plumes = []
    masks = []

    # itereate over the plumes and setup the plumes
    for p_number, plume in pp['plume_df'].iterrows():

        # if p_number not in [0,1,2,3,4]:
        #     continue

        # get plume time stamp
        current_timestamp = ut.get_timestamp(plume.filename)

        # read in satellite data
        if current_timestamp != previous_timestamp:
            sat_data_utm = resample_satellite_datasets(plume, current_timestamp, pp)
            previous_timestamp = current_timestamp
            if sat_data_utm is None:
                continue

        # construct plume and background coordinate data
        plume_geom_geo = setup_plume_data(plume, sat_data_utm)

        # subset the satellite AOD data to the plume
        plume_data_utm = subset_sat_data_to_plume(sat_data_utm, plume_geom_geo)

        # plume aod
        plume_aod = extract_combined_aod_full_plume(plume_data_utm, plume_geom_geo['plume_mask'])

        plumes.append(plume_aod)
        masks.append(plume_geom_geo['plume_mask'])

    # evaluate interpolation methods
    eval_interpolation_methods(plumes, masks)




if __name__ == '__main__':


    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()