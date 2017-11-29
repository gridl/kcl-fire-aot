import numpy as np


def extract_best_mean_aod(viirs_aod,
                          viirs_flag,
                          orac_aod,
                          orac_cost,
                          mask):

    viirs_aod = viirs_aod[mask]
    viirs_flag = viirs_flag[mask]
    orac_aod = orac_aod[mask]
    orac_cost = orac_cost[mask]

    # need to improve the approach here but will do for now
    viirs_aod[viirs_flag > 0] = orac_aod[viirs_flag > 0]

    return np.mean(viirs_aod)


def extract_bg_aod(viirs_aod, viirs_flag, mask):
    viirs_aod = viirs_aod[mask]
    viirs_flag = viirs_flag[mask]
    return np.mean(viirs_aod[viirs_flag == 0])


def compute_tpm(viirs_aod_utm_plume, viirs_flag_utm_plume,
                orac_aod_utm_plume, orac_cost_utm_plume,
                viirs_aod_utm_background, viirs_flag_utm_background,
                utm_plume_polygon, utm_plume_mask, utm_bg_mask):

    try:
        pm_factor = 4.3  # m2/g TODO add this to a config file

        # get plume area
        plume_area = utm_plume_polygon.area

        # extract mean ORAC plume AOD using plume mask
        mean_plume_aod = extract_best_mean_aod(viirs_aod_utm_plume, viirs_flag_utm_plume,
                                               orac_aod_utm_plume, orac_cost_utm_plume,
                                               utm_plume_mask)

        # extract mean MYD04 background AOD using bg_mask
        mean_bg_aod = extract_bg_aod(viirs_aod_utm_background, viirs_flag_utm_background)

        # subtract mean background AOD from mean plume AOD
        print 'mean plume', mean_plume_aod
        print 'mean bg', mean_bg_aod
        mean_plume_aod -= mean_bg_aod

        # convert to PM using conversion factor
        plume_pm = mean_plume_aod / pm_factor

        # multiply by plume area to get total tpm
        tpm = plume_pm * plume_area

        # return plume TPM
        return tpm
    except:
        return None
