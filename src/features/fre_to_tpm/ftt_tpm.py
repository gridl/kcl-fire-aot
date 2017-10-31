import numpy as np


def extract_mean_aod(aod, mask):
    return np.mean(aod[mask])


def compute_tpm(utm_orac_aod_plume,
                utm_orac_aod_background,
                utm_mxd04_aod_background,
                utm_plume_polygon,
                utm_plume_mask,
                utm_bg_mask):

    pm_factor = 4.3  # m2/g TODO add this to a config file

    # get plume area
    plume_area = utm_plume_polygon.area

    # extract mean ORAC plume AOD using plume mask
    mean_plume_aod = extract_mean_aod(utm_orac_aod_plume, utm_plume_mask)

    # extract mean MYD04 background AOD using bg_mask
    mean_bg_aod_orac = extract_mean_aod(utm_orac_aod_background, utm_bg_mask)
    mean_bg_aod_modis = extract_mean_aod(utm_mxd04_aod_background, utm_bg_mask)

    # subtract mean background AOD from mean plume AOD
    mean_plume_aod -= mean_bg_aod_modis

    # convert to PM using conversion factor
    plume_pm = mean_plume_aod / pm_factor

    # multiply by plume area to get total tpm
    tpm = plume_pm * plume_area

    # return plume TPM
    return tpm
