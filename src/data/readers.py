

def read_orac(orac_file_path):
    '''

    :param orac_file_path: path to orac nc file
    :return: opened orac nc file
    '''
    pass


def read_goes_frp(goes_frp_file_path):
    '''

    :param goes_frp_file_path: path to goes frp data
    :return: opened goes FRP data set as a dataframe
    '''
    pass


def read_plume_masks(plume_mask_file_path):
    '''

    :param plume_mask_file_path: path to digited plume mask
    :return: plume mask locations
    '''
    pass

def read_lc(lc_file_path):
    '''

    :param lc_file_path: path to landcover file
    :return: opened landcover file
    '''
    pass


def main():

    root = ''

    orac_file_path = root +
    goes_frp_file_path = root +
    plume_mask_file_path = root +
    lc_file_path = root +

    res = read_orac(orac_file_path)
    res = read_goes_frp(goes_frp_file_path)
    res = read_plume_masks(plume_mask_file_path)
    res = read_lc(lc_file_path)

if __name__ == "__main__":
    main()