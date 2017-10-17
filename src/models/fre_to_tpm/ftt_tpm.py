import numpy as np


def extract_mean_aod(aod, mask):
    return np.mean(aod[mask])


def compute_tpm(orac_aod, myd04_aod, plume_polygon, plume_mask, bg_mask):

    pm_factor = 4.3  # TODO add this to a config file

    # get plume area
    plume_area = plume_polygon.area

    # extract mean ORAC plume AOD using plume mask
    mean_plume_aod = extract_mean_aod(orac_aod, plume_polygon)

    # extract mean MYD04 background AOD using bg_mask
    mean_bg_aod = extract_mean_aod(myd04_aod, plume_polygon)

    # subtract mean background AOD from mean plume AOD
    mean_plume_aod -= mean_bg_aod

    # convert to PM using conversion factor
    plume_pm = mean_plume_aod*pm_factor

    # multiply by plume area to get total tpm
    tpm = plume_pm * plume_area

    # return plume TPM
    return tpm
