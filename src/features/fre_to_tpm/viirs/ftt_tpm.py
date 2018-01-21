import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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


def extract_best_mean_aod(d,
                          viirs_aod,
                          viirs_flag,
                          orac_aod,
                          orac_cost,
                          plume_mask,
                          bg_aod_dict,
                          sub_plume_logging_path,
                          plot=True):

    # first redifine the plume mask to be those pixels that are: (not background | failed
    # retrievals (here assumed  due to high AOD and not cloud) ) & are plume_mask
    flag_mask = viirs_flag > 1
    aod_mask = viirs_aod > bg_aod_dict['mean_bg_aod'] + 3*bg_aod_dict['std_bg_aod']
    updated_plume_mask = plume_mask & (flag_mask | aod_mask)

    if plot:
        plt.imshow(np.ma.masked_array(viirs_aod, ~updated_plume_mask), vmin=0, vmax=2)
        plt.colorbar()
        plt.savefig(os.path.join(sub_plume_logging_path, 'viirs_sub_plume_aod.png'))
        plt.close()

        ax = plt.imshow(np.ma.masked_array(viirs_flag, ~updated_plume_mask))
        cmap = cm.get_cmap('Set1', 4)
        ax.set_cmap(cmap)
        cb = plt.colorbar()
        plt.savefig(os.path.join(sub_plume_logging_path, 'viirs_sub_plume_flag.png'))
        plt.close()

        plt.imshow(np.ma.masked_array(orac_aod, ~updated_plume_mask), vmin=0, vmax=2)
        plt.colorbar()
        plt.savefig(os.path.join(sub_plume_logging_path, 'viirs_sub_plume_orac.png'))
        plt.close()

        plt.imshow(np.ma.masked_array(orac_cost, ~updated_plume_mask), vmax=100, cmap='plasma')
        plt.colorbar()
        plt.savefig(os.path.join(sub_plume_logging_path, 'viirs_sub_plume_orac_cose.png'))
        plt.close()

        viirs_mask = updated_plume_mask & (viirs_flag <= 1)
        orac_mask = updated_plume_mask & (viirs_flag > 1) & (orac_cost <= 50)
        combined = np.zeros(viirs_mask.shape)
        combined[orac_mask] = orac_aod[orac_mask]
        combined[viirs_mask] = viirs_aod[viirs_mask]
        mask = combined == 0
        plt.imshow(np.ma.masked_array(combined, mask), vmin=0, vmax=2)
        plt.colorbar()
        plt.savefig(os.path.join(sub_plume_logging_path, 'combined_aod.png'))
        plt.close()

    # now extract the plume data
    viirs_aod = viirs_aod[updated_plume_mask]
    viirs_flag = viirs_flag[updated_plume_mask]
    orac_aod = orac_aod[updated_plume_mask]
    orac_cost = orac_cost[updated_plume_mask]

    # get the good data masks
    viirs_good = viirs_flag <= 1
    orac_good = orac_cost <= 50
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


def extract_bg_aod(viirs_aod, viirs_flag, mask):
    viirs_aod = viirs_aod[mask]
    viirs_flag = viirs_flag[mask]
    return {'mean_bg_aod': np.mean(viirs_aod[viirs_flag == 0]),
            'std_bg_aod': np.std(viirs_aod[viirs_flag == 0])}


def compute_tpm(viirs_aod_utm_plume, viirs_flag_utm_plume,
                orac_aod_utm_plume, orac_cost_utm_plume,
                utm_plume_polygon, utm_plume_mask, bg_aod_dict,
                sub_plume_logging_path, plot=True):

    d = build_output_dict()

    try:
        # put background data into d
        d['mean_bg_aod'] = bg_aod_dict['mean_bg_aod']
        d['std_bg_aod'] = bg_aod_dict['std_bg_aod']

        d['pm_factor'] = 4.3  # m2/g TODO add this to a config file

        # get plume area
        d['plume_area_total'] = utm_plume_polygon.area

        # extract mean ORAC plume AOD using plume mask
        extract_best_mean_aod(d, viirs_aod_utm_plume, viirs_flag_utm_plume, orac_aod_utm_plume, orac_cost_utm_plume,
                              utm_plume_mask, bg_aod_dict, sub_plume_logging_path, plot=plot)

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
