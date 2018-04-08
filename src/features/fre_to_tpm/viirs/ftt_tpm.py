import os

import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import src.config.constants as constants


def build_output_dict():
    d = {}
    key_list = ['n_plume_pixels', 'n_viirs_good_pixels', 'n_orac_good_pixels', 'n_both_good_pixels',
                'n_either_good_pixels', 'mad_aod_on_quality', 'mad_aod_on_range', 'coverage_plume',
                'coverage_viirs', 'coverage_orac', 'plume_area_total', 'plume_area_used',
                'median_plume_aod', 'mean_plume_aod', 'std_plume_aod', 'coverage_both', 'mean_bg_aod',
                'std_bg_aod', 'adjusted_mean_plume_aod', 'tpm', 'pm_factor',
                'median_plume_viirs_aod', 'mean_plume_viirs_aod', 'std_plume_viirs_aod',
                'median_plume_orac_aod', 'mean_plume_orac_aod', 'std_plume_orac_aod',]
    for k in key_list:
        d[k] = np.nan
    return d


def extract_best_mean_aod_subplume(d,
                          plume_data_utm,
                          plume_mask,
                          bg_aod_dict,
                          logging_path,
                          plot=True):

    # first redifine the plume mask to be those pixels that are: (not background | failed
    # retrievals (here assumed  due to high AOD and not cloud) ) & are plume_mask
    flag_mask = plume_data_utm['viirs_flag_utm_plume'] > 1
    aod_mask = plume_data_utm['viirs_aod_utm_plume'] > bg_aod_dict['mean_bg_aod'] + 3*bg_aod_dict['std_bg_aod']
    updated_plume_mask = plume_mask & (flag_mask | aod_mask)

    if plot:
        plt.imshow(np.ma.masked_array(plume_data_utm['viirs_aod_utm_plume'], ~updated_plume_mask), vmin=0, vmax=2)
        plt.colorbar()
        plt.savefig(os.path.join(logging_path, 'viirs_sub_plume_aod.png'))
        plt.close()

        ax = plt.imshow(np.ma.masked_array(plume_data_utm['viirs_flag_utm_plume'], ~updated_plume_mask))
        cmap = cm.get_cmap('Set1', 4)
        ax.set_cmap(cmap)
        cb = plt.colorbar()
        plt.savefig(os.path.join(logging_path, 'viirs_sub_plume_flag.png'))
        plt.close()

        plt.imshow(np.ma.masked_array(plume_data_utm['orac_aod_utm_plume'], ~updated_plume_mask), vmin=0, vmax=2)
        plt.colorbar()
        plt.savefig(os.path.join(logging_path, 'viirs_sub_plume_orac.png'))
        plt.close()

        plt.imshow(np.ma.masked_array(plume_data_utm['orac_cost_utm_plume'], ~updated_plume_mask), vmax=10, cmap='plasma')
        plt.colorbar()
        plt.savefig(os.path.join(logging_path, 'viirs_sub_plume_orac_cost.png'))
        plt.close()

        viirs_mask = updated_plume_mask & (plume_data_utm['viirs_flag_utm_plume'] <= 1)
        orac_mask = updated_plume_mask & (plume_data_utm['viirs_flag_utm_plume'] > 1) & (plume_data_utm['orac_cost_utm_plume'] <= 3)
        combined = np.zeros(viirs_mask.shape)
        combined[orac_mask] = plume_data_utm['orac_aod_utm_plume'][orac_mask]
        combined[viirs_mask] = plume_data_utm['viirs_aod_utm_plume'][viirs_mask]
        mask = combined == 0
        plt.imshow(np.ma.masked_array(combined, mask), vmin=0, vmax=2)
        plt.colorbar()
        plt.savefig(os.path.join(logging_path, 'combined_aod.png'))
        plt.close()

    # now extract the plume data
    viirs_aod = plume_data_utm['viirs_aod_utm_plume'][updated_plume_mask]
    viirs_flag = plume_data_utm['viirs_flag_utm_plume'][updated_plume_mask]
    orac_aod = plume_data_utm['orac_aod_utm_plume'][updated_plume_mask]
    orac_cost = plume_data_utm['orac_cost_utm_plume'][updated_plume_mask]

    # get the good data masks  # TODO Add to config file
    viirs_good = viirs_flag <= 1
    orac_good = orac_cost <= 3
    both_good = viirs_good & orac_good
    either_good = viirs_good | orac_good
    both_aod_gt_1_lt_2 = (viirs_aod >= 1) & (viirs_aod < 2) & (orac_aod >= 1) & (orac_aod < 2)

    # extract the aod data
    viirs_aod_subset = viirs_aod[viirs_good]
    orac_aod_subset = orac_aod[orac_good & ~viirs_good]
    aod = np.concatenate((viirs_aod_subset, orac_aod_subset))

    # stats
    d['n_plume_pixels'] = np.sum(updated_plume_mask)
    d['n_viirs_good_pixels'] = np.sum(viirs_good)
    d['n_orac_good_pixels'] = np.sum(orac_good)
    d['n_both_good_pixels'] = np.sum(both_good)
    d['n_either_good_pixels'] = np.sum(either_good)

    d['mad_aod_on_quality'] = np.mean(np.abs(viirs_aod[both_good] - orac_aod[both_good]))
    d['mad_aod_on_range'] = np.mean(np.abs(viirs_aod[both_aod_gt_1_lt_2] - orac_aod[both_aod_gt_1_lt_2]))
    d['coverage_viirs'] = float(d['n_viirs_good_pixels']) / d['n_plume_pixels']
    d['coverage_orac'] = float(d['n_orac_good_pixels']) / d['n_plume_pixels']
    d['coverage_both'] = float(d['n_both_good_pixels']) / d['n_plume_pixels']
    d['coverage_either'] = float(d['n_either_good_pixels']) / d['n_plume_pixels']

    d['mean_plume_aod'] = np.mean(aod)
    d['median_plume_aod'] = np.median(aod)
    d['std_plume_aod'] = np.std(aod)

    d['mean_plume_viirs_aod'] = np.mean(viirs_aod_subset)
    d['median_plume_viirs_aod'] = np.median(viirs_aod_subset)
    d['std_plume_viirs_aod'] = np.std(viirs_aod_subset)

    d['mean_plume_orac_aod'] = np.mean(orac_aod[orac_good])
    d['median_plume_orac_aod'] = np.median(orac_aod[orac_good])
    d['std_plume_orac_aod'] = np.std(orac_aod[orac_good])

    d['coverage_plume'] = np.sum(updated_plume_mask * 1.) / np.sum(plume_mask)


def extract_combined_aod_full_plume(plume_data_utm,
                                    plume_mask,
                                    logging_path,
                                    plot=True):

    # combine plume mask with VIIRS good and ORAC good
    viirs_good = plume_data_utm['viirs_flag_utm_plume'] <= 1
    orac_good = plume_data_utm['orac_cost_utm_plume'] <= 3
    viirs_plume_mask = plume_mask & viirs_good  # viirs contribtuion
    orac_plume_mask = plume_mask & (orac_good & ~viirs_good)  # ORAC contribution

    if plot:
        combined = np.zeros(viirs_plume_mask.shape) - 999
        combined[orac_plume_mask] = plume_data_utm['orac_aod_utm_plume'][orac_plume_mask]
        combined[viirs_plume_mask] = plume_data_utm['viirs_aod_utm_plume'][viirs_plume_mask]
        mask = combined == -999
        plt.imshow(np.ma.masked_array(combined, mask), vmin=0, vmax=2)
        plt.colorbar()
        plt.savefig(os.path.join(logging_path, 'combined_aod.png'))
        plt.close()

    # extract the aod data
    viirs_aod_subset = plume_data_utm['viirs_aod_utm_plume'][viirs_plume_mask]
    orac_aod_subset = plume_data_utm['orac_aod_utm_plume'][orac_plume_mask]
    plume_aod = np.concatenate((viirs_aod_subset, orac_aod_subset))
    return plume_aod, combined


def extract_bg_aod(plume_data_utm, mask):
    masked_viirs_aod = plume_data_utm['viirs_aod_utm_background'][mask]
    masked_viirs_flag = plume_data_utm['viirs_flag_utm_background'][mask]

    masked_orac_aod = plume_data_utm['orac_aod_utm_background'][mask]
    masked_orac_cost = plume_data_utm['orac_cost_utm_background'][mask]

    if np.min(plume_data_utm['viirs_flag_utm_background'][mask]) <= 1:
        # should we use 1 or 0?
        aod_mean = np.mean(masked_viirs_aod[masked_viirs_flag <= 1])
        aod_std = np.std(masked_viirs_aod[masked_viirs_flag <= 1])
        typ = 'sp'
    elif np.min(masked_orac_cost[mask] < 3):
        aod_mean = np.mean(masked_orac_aod[masked_orac_cost < 3])
        aod_std = np.std(masked_orac_aod[masked_orac_cost < 3])
        typ = 'orac'
    else:
        aod_mean = np.nan
        aod_std = np.nan
        typ = 'failed'

    return {'mean_bg_aod': aod_mean,
            'std_bg_aod': aod_std,
             'bg_type': typ}


def interp_aod(aod, plume_mask, logging_path, plot=True):
    '''
    Interpolate using a radial basis function.  See models
    that shows thta this is the best approach.
    '''

    good_mask = aod != -999

    # build the interpolation grid
    y = np.linspace(0, 1, aod.shape[0])
    x = np.linspace(0, 1, aod.shape[1])
    xx, yy = np.meshgrid(x, y)

    # create interpolated grid (can extend to various methods)
    rbf = interpolate.Rbf(xx[good_mask], yy[good_mask], aod[good_mask], function='linear')
    interpolated_aod = rbf(xx, yy)

    aod[~good_mask] = interpolated_aod[~good_mask]

    if plot:
        plt.imshow(np.ma.masked_array(aod, ~plume_mask), vmin=0, vmax=2)
        plt.colorbar()
        plt.savefig(os.path.join(logging_path, 'combined_aod_interpolated.png'))
        plt.close()

    return aod[plume_mask]


def compute_tpm_subset(plume_data_utm,
                utm_plume_polygon, utm_plume_mask, bg_aod_dict,
                sub_plume_logging_path, pp):

    d = build_output_dict()

    try:
        # put background data into d
        d['mean_bg_aod'] = bg_aod_dict['mean_bg_aod']
        d['std_bg_aod'] = bg_aod_dict['std_bg_aod']
        d['bg_type'] = bg_aod_dict['bg_type']

        d['pm_factor'] = 4.3  # m2/g TODO add this to a config file

        # get plume area
        d['plume_area_total'] = utm_plume_polygon.area

        # extract mean ORAC plume AOD using plume mask
        extract_best_mean_aod_subplume(d, plume_data_utm,
                              utm_plume_mask, bg_aod_dict, sub_plume_logging_path, pp)

        d['plume_area_used'] = d['plume_area_total'] * d['coverage_plume']

        # subtract mean background AOD from mean plume AOD
        d['adjusted_mean_plume_aod'] = d['mean_plume_aod'] - d['mean_bg_aod']

        # convert to PM using conversion factor
        plume_pm = d['adjusted_mean_plume_aod'] / d['pm_factor']

        # multiply by plume area evaluated (i.e. considered not background) to get total tpm
        d['tpm'] = plume_pm * d['plume_area_used']

        # return plume TPM
        return d
    except Exception, e:
        print 'tpm extraction failed with error', str(e), 'filling output dict with NaN'
        return d


def compute_tpm_full(plume_data_utm, plume_geom_utm, plume_geom_geo, bg_aod_dict, plume_logging_path):

    d = {}

    try:
        # put background data into d
        d['mean_bg_aod'] = bg_aod_dict['mean_bg_aod']
        d['std_bg_aod'] = bg_aod_dict['std_bg_aod']
        d['bg_type'] = bg_aod_dict['bg_type']

        d['pm_factor'] = 7.42  # m2/g from 10.1029/2005GL022678 TODO add this to a config file

        # get plume area
        d['plume_area_total'] = plume_geom_utm['utm_plume_polygon'].area

        # extract mean ORAC plume AOD using plume mask
        combined_aod_list, comibined_aod_image = extract_combined_aod_full_plume(plume_data_utm,
                                                                                 plume_geom_geo['plume_mask'],
                                                                                 plume_logging_path)

        # create interpolated aod from image
        interpolated_aod = interp_aod(comibined_aod_image, plume_geom_geo['plume_mask'], plume_logging_path)

        # subtract mean background AOD from plume AODs and then take the mean.
        # Another approach would be to sum the above background AOD.  But we are missing
        # some retrieval pixels, so this will lead to a bias, as not retrieved pixels will
        # likely have raised AOD, but are not observed, so wont be considered in any sum.
        # taking the mean, we can account for these pixels in the next step.
        d['mean_plume_aod_bg_adjusted'] = np.mean(combined_aod_list - d['mean_bg_aod'])
        d['summed_plume_aod_bg_adjusted'] = np.sum(combined_aod_list - d['mean_bg_aod'])
        d['std_plume_aod_bg_adjusted'] = np.std(combined_aod_list - d['mean_bg_aod'])
        d['mean_interpolated_plume_aod_adjusted'] = np.mean(interpolated_aod - d['mean_bg_aod'])
        d['summed_interpolated_plume_aod_adjusted'] = np.sum(interpolated_aod - d['mean_bg_aod'])


        # convert to mean PM using conversion factor
        plume_pm = d['mean_interpolated_plume_aod_adjusted'] / d['pm_factor']  # in g/m^2

        # multiply by plume area to get total tpm
        d['tpm'] = plume_pm * d['plume_area_total']  # g

        # return plume TPM
        return d
    except Exception, e:
        print 'tpm extraction failed with error', str(e)
        return d
