# open Himawari-8 standard format
# output data dictionary ready to put in frp_pixel.py
# the struction of the dictionary should be like this
# ['ir39', 'ir12', 'saa', 'ir11', 'cloudfree', 'lat',
# 'ir11rad', 'diff', 'sun_glint', 'ACQTIME', 'vza',
# 'vaa', 'lon', 'cmask', 'CMa_TEST', 'pixsize', 'szen',
# 'tirradratio', 'infos', 'redrad', 'visradratio',
# 'tcwv', 'ir39rad', 'lcov']

import os
import datetime
import struct
import numpy as np
import scipy.ndimage
from subprocess import call
import bz2file as h8_bz2
import sys


def sunglint(vzen, vaz, szen, saz):
    """ all the input in degrees
        calculation from Prins et al. enhanced fired detection 1998
    """
    vzen_r = np.radians(vzen)
    vaz_r = np.radians(vaz)
    szen_r = np.radians(szen)
    saz_r = np.radians(saz)
    raz_r = vaz_r - saz_r
    G = np.cos(vzen_r) * np.cos(szen_r) - np.sin(vzen_r) * np.sin(szen_r) * np.cos(raz_r)
    sun_glint = np.degrees(np.arccos(G))
    return sun_glint


def cloud_mask(data):
    """
    A simple cloud masking for Himawari8
    for fire detection
    """
    # threshold for Albedo
    vis_day = 0.20
    bt10_day = 278.
    bt10_day_vz = 290

    bt4_ni = 272.
    bt10_ni = 268.
    bt11_ni = 268.
    Diffthresh_day = 15.0
    Diffthresh_ni = 10.0
    Diff2thresh = 13.7
    data['cmask'] = np.zeros(data['ir39'].shape, dtype=np.int8) - 1
    # work in the satellite visible area and the land
    mask = ((data['vza'] > 0.0) & (data['lcov'] < 20))
    data['cmask'][mask] = 0.
    # check if day or night - day <70.degrees and night gt 70.degrees
    # work on day time first
    day = ((data['szen'] < 75.) & (data['szen'] > 0.0) & \
           (data['lcov'] < 20) & (data['vza'] > 0.0))
    # visible band
    vis_thresh = ((data['vis'] > vis_day) & (data['ir11'] < bt10_day) & \
                  (data['cmask'] < 1) & (day > 0))
    data['cmask'][vis_thresh] = 1

    # 10mincron threshold
    tir_thresh = ((data['ir11'] < bt10_day) & (data['cmask'] < 1) & (day > 0))
    data['cmask'][tir_thresh] = 2

    # his is the mid infrared temperature threshold  used here for cloud detected
    # bt4_th = -0.35 * data['szen'] + 300
    diff = data['ir39'] - data['ir11']
    # 10mincron and 3.9um difference threshold
    dif_thresh = ((data['vis'] > vis_day / 3) & (diff > Diffthresh_day) & (data['ir11'] < bt10_day_vz) & \
                  (data['cmask'] < 1) & (day > 0))
    data['cmask'][dif_thresh] = 5

    # night time
    night = ((data['szen'] > 90.0) & (data['vza'] > 0.) & (data['lcov'] < 20))
    # MIR
    # mir_thresh = ((data['ir39'] < bt4_ni) & (np.abs(diff) > 2) & (night > 0))
    # data['cmask'][mir_thresh] = 7
    # 10mincron
    tir_thresh = ((data['ir11'] < bt10_ni) & (np.abs(diff) > 4) & (night > 0))
    data['cmask'][tir_thresh] = 8

    # 10mincron and 3.9um difference threshold
    dif_thresh = ((diff > Diffthresh_ni) & (data['ir39'] < 275) & (night > 0))
    data['cmask'][dif_thresh] = 5
    # twilight time
    twilight = ((data['szen'] >= 75.0) & (data['szen'] < 90.0) & (data['vza'] > 0.) & (data['lcov'] < 20))
    vis_thresh = ((data['vis'] > vis_day / 4) & (data['cmask'] < 1) & (twilight > 0))
    data['cmask'][vis_thresh] = 1
    # sun glint affest area
    glint = (data['sun_glint'] < 20.0)
    vis_thresh = ((data['vis'] > vis_day / 4) & (data['cmask'] < 1) & (glint > 0))
    data['cmask'][vis_thresh] = 1

    # this is clear sky
    data['cloudfree'] = (data['cmask'] < 1.0) & (data['cmask'] > -1.0)
    return data


def water_mask(data):
    """
    A simple water masking for Himawari8
    for fire georeference
    land: 1, water: 0, background:-1
    """
    # threshold for Albedo
    sir_day = 0.05
    data['wmask'] = np.zeros(data['ir39'].shape, dtype=np.int8) - 1
    # work in the satellite visible area and the land
    mask = data['vza'] > 0.0
    data['wmask'][mask] = 0.
    # check if day or night - day <70.degrees and night gt 70.degrees
    # work on day time first
    day = ((data['szen'] < 80.) & (data['szen'] > 0.0) & (data['vza'] > 0.0))
    # sir band
    sir_thresh = ((data['sir'] > sir_day) & (data['wmask'] < 1) & \
                  (data['cmask'] < 1) & (day > 0) & (data['diff'] < 20))
    data['wmask'][sir_thresh] = 1

    return data


def geo_read(f, verbose=False):
    """
    read in the static data like view angle, landcover
    put them in a data dictionary
    """
    dim = 5500  # hard coded for Himawari8 possible it is 5500 in which case we need to zoom
    if verbose:
        print 'reading file %s' % f
    dtype = np.float32
    shape = (2, dim, dim)
    data = np.fromfile(f, dtype=dtype).reshape(shape)
    lat = data[0, :, :].astype(dtype)
    lon = data[1, :, :].astype(dtype)
    return lat, lon


def static_read(file_dict, verbose=False):
    """
    read in the static data like view angle, landcover
    put them in a data dictionary
    """
    d = {}
    dim = 5500  # hard coded for Himawari8
    for key in file_dict.keys():
        file = file_dict[key][0][0]
        if verbose:
            print 'file path %s' % key
            print 'reading file %s' % file
        if key == 'landcover_path':
            dtype = np.int8
            shape = (dim, dim)
            data = np.fromfile(file, dtype=dtype).reshape(shape)
            data_key = file_dict[key][1]
            d[data_key] = data.astype(dtype)
        elif key == 'fixed_position_path':
            dtype = np.float32
            shape = (dim, dim)
            data = np.fromfile(file, dtype=dtype).reshape(shape)
            data_key = file_dict[key][1]
            d[data_key] = data.astype(dtype)
        else:
            dtype = np.float32
            shape = (2, dim, dim)
            data = np.fromfile(file, dtype=dtype).reshape(shape)
            data_key = file_dict[key][1]
            d[data_key] = data[0, :, :].astype(dtype)
            data_key = file_dict[key][2]
            d[data_key] = data[1, :, :].astype(dtype)
    # pixel size
    d['pixsize'] = ((2.0 ** 2) * 1000.0 ** 2) * (1 / np.cos(np.radians(d['vza'])))
    return d


def sun_angles(lat, lon, time_key):
    """
    input: lat, np array; lon, np array
           time_key, string
           YYYYMMDDHHMM format like 201501031100
    output:szen, sun zenith angle
           saa, sun azimuth angle
    """
    # Define internal constants used for conversion
    EQTIME1 = 229.18
    EQTIME2 = 0.000075
    EQTIME3 = 0.001868
    EQTIME4 = 0.032077
    EQTIME5 = 0.014615
    EQTIME6 = 0.040849

    DECL1 = 0.006918
    DECL2 = 0.399912
    DECL3 = 0.070257
    DECL4 = 0.006758
    DECL5 = 0.000907
    DECL6 = 0.002697
    DECL7 = 0.00148
    # Evaluate the input lat and lon in radians
    RadLat = np.radians(lat)
    dt = datetime.datetime.strptime(time_key, '%Y%m%d%H%M')
    # get the days in the year, normal year:365; leap year:366
    d1 = datetime.datetime(dt.year, 1, 1)
    d2 = datetime.datetime(dt.year + 1, 1, 1)
    days_in_year = (d2 - d1).days
    # Evaluate the fractional year in radians
    # dt.hour-12 because gamma start from local noon time
    gamma = 2 * np.pi * (dt.timetuple().tm_yday - 1 + \
                         (dt.hour - 12) / 24.0) / days_in_year
    # Evaluate the Equation of time in minutes
    eqtime = EQTIME1 * (EQTIME2 + EQTIME3 * np.cos(gamma) - \
                        EQTIME4 * np.sin(gamma) - EQTIME5 * np.cos(2 * gamma) - \
                        0.040849 * np.sin(2 * gamma))
    # Time offset in minutes
    time_offset = eqtime + 4.0 * lon
    # local solar time in minutes
    true_solar_time = dt.hour * 60 + dt.minute + dt.second / 60 + time_offset
    # Solar hour angle in degrees and in radians
    HaRad = np.radians((true_solar_time / 4.) - 180.)
    # Evaluate the solar declination angle in radians
    Decli = DECL1 - DECL2 * np.cos(gamma) + DECL3 * np.sin(gamma) - \
            DECL4 * np.cos(2 * gamma) + DECL5 * np.sin(2 * gamma) - \
            DECL6 * np.cos(3 * gamma) + DECL7 * np.sin(3 * gamma)
    # Evaluate the Solar local Coordinates
    CosZen = (np.sin(RadLat) * np.sin(Decli) + \
              np.cos(RadLat) * np.cos(Decli) * np.cos(HaRad))

    TmpZenRad = np.arccos(CosZen)

    szen = np.degrees(TmpZenRad)

    CosAzi = -((np.sin(RadLat) * np.cos(TmpZenRad) - np.sin(Decli)) / \
               (np.cos(RadLat) * np.sin(TmpZenRad)))

    saa = 360. - np.degrees(np.arccos(CosAzi))
    # Correct for Time < 12.00 ( -> in range 0 . 180 )
    saa[(true_solar_time < 720)] = 360. - saa[(true_solar_time < 720)]  # in minutes 12 *60

    return (szen, saa)


def rebin(a, newshape):
    '''Rebin an array to a new shape.
    '''
    assert len(a.shape) == len(newshape)

    slices = [slice(0, old, float(old) / new) for old, new in zip(a.shape, newshape)]
    coordinates = np.mgrid[slices]
    indices = coordinates.astype('i')  # choose the biggest smaller integer index
    return a[tuple(indices)]


def H8_file_read(file, verbose=False):
    # type: (object, object) -> object
    '''
    read in a single Himawari8 file.
    '''
    if not os.path.exists(file):
        print 'can not read %s' % file
    fileExtension = os.path.splitext(file)[1]
    if fileExtension in '.bz2':
        fh = h8_bz2.BZ2File(file, 'rb')
    else:
        fh = open(file, 'rb')
        # doit = call(["bunzip2", file])
        # if doit < 1:
        #     file = file[:-4]
        # else:
        #     print 'can not unzip ', file

    # Read in the head blocks

    print "processing %s" % file
    total_len = 0
    # read in the file as binary see python struct for help
    for bb in xrange(11):
        # for Block 1
        fh.seek(total_len)
        Block_no = struct.unpack('b', fh.read(1))[0]
        if verbose:
            print 'Reading block %s' % Block_no
        fh.seek(total_len + 1)
        Block_len = struct.unpack('h', fh.read(2))[0]
        if verbose:
            print 'The length of block %s is %s' % (Block_no, Block_len)
        # from block 2 read in number of samps and lines
        if Block_no == 2:
            fh.seek(total_len + 5)
            samps = struct.unpack('h', fh.read(2))[0]
            fh.seek(total_len + 7)
            lines = struct.unpack('h', fh.read(2))[0]
        # from block 3 read projection information
        elif Block_no == 3:
            fh.seek(total_len + 3)
            sub_lon = struct.unpack('d', fh.read(8))[0]
            print 'central longitude %r' % sub_lon
            fh.seek(total_len + 11)
            CFAC = struct.unpack('I', fh.read(4))[0]
            fh.seek(total_len + 15)
            LFAC = struct.unpack('I', fh.read(4))[0]
            fh.seek(total_len + 19)
            COFF = struct.unpack('f', fh.read(4))[0]
            fh.seek(total_len + 23)
            LOFF = struct.unpack('f', fh.read(4))[0]
            fh.seek(total_len + 27)
            # Information about satellite height, earth equatorial radius
            # more infor can be found on page 16 of
            # Himawari_D_users_guide_en
            Proj_info = struct.unpack('ddddddd', fh.read(8 * 7))[:]
        elif Block_no == 4:
            fh.seek(total_len + 3)
            Nav_info = struct.unpack('dddddddd', fh.read(8 * 8))[:]
        elif Block_no == 5:
            fh.seek(total_len + 3)
            Band_no = struct.unpack('h', fh.read(2))[0]
            fh.seek(total_len + 5)
            central_wave = struct.unpack('d', fh.read(8))[0]
            fh.seek(total_len + 19)
            Cal_info = struct.unpack('ddddddddddd', fh.read(8 * 11))[:]


            # Change the block length for next block
        total_len += Block_len
        if verbose:
            print 'Total header length %s' % total_len
    # Now read in image data
    fh.seek(total_len)
    dtype = 'u2'
    shape = [lines, samps]
    size = np.dtype(dtype).itemsize * samps * lines
    data = fh.read()
    data = np.frombuffer(data[:size], dtype).reshape(shape)
    fh.close()
    fileExtension = os.path.splitext(file)[1]
    # if fileExtension in '.DAT':
    #     call(["bzip2", file])

    if verbose:
        print 'slope: %f, offset: %f for radiance' % (Cal_info[0], Cal_info[1])
    radiance = data * Cal_info[0] + Cal_info[1]
    # for infrared bands
    if Band_no > 6:
        # for Planck temperature
        speed_of_light = Cal_info[8]
        planck_constant = Cal_info[9]
        boltzmann_constant = Cal_info[10]
        # radiance = 2.5
        # central_wave = 4
        c1 = 2.0 * planck_constant * speed_of_light * speed_of_light
        c2 = planck_constant * speed_of_light / boltzmann_constant
        #   -- Derived constant scaling factors for:
        #        c1: W.m2 to W/(m2.um-4) => multiplier of 1.0e+24 is required.
        #        c2: K.m to K.um a=> multiplier of 1.0e+06 is required.
        c1_scale = 1.0e+24
        c2_scale = 1.0e+06
        #   -- Calculate wavelength dependent "constants"
        fk1 = c1_scale * c1 / (central_wave ** 5)
        fk2 = c2_scale * c2 / central_wave
        logarithm = np.log((fk1 / (radiance) + 1.0))
        temperature = fk2 / logarithm
        BT = Cal_info[2] + Cal_info[3] * temperature + Cal_info[4] * \
                                                       temperature * temperature
    else:
        # if samps > 12000:  # hard coded should find a better way later
        #     # Resampled by a factor of 0.25 with bilinear interpolation
        #     # sub = radiance[4300*4:4700*4,2500*4:2900*4]
        #     # d['vis_full'] = sub
        #     radiance = rebin(radiance, (lines / 4, samps / 4))
        # elif samps > 5500:
        #     radiance = rebin(radiance, (lines / 2, samps / 2))

        # for visible band this is Albedo
        BT = radiance * Cal_info[2]
    return (radiance, BT)


def Himawari_read(file_dict, verbose=False):
    """
    Read the Himawari-8 channels
    for fire detection, we only need red, MIR and TIR
    input: file dictionary like
    {'red_path' : 'HS_H08_20150109_0600_B03_FLDK_R20_S0101.DAT',
    'mir_path' : 'HS_H08_20150109_0600_B07_FLDK_R20_S0101.DAT',
    'tir_path' : 'HS_H08_20150109_0600_B14_FLDK_R20_S0101.DAT'}
    output: date dictionary like
    {'mir_BT' : np.array(5500,55500)}
    reference: Himawari_D_users_guide_en from
    http://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/hsd_sample/HS_D_users_guide_en_v11.pdf
    """
    d = {}
    for key in file_dict.keys():
        files = file_dict[key][0]
        files.sort()
        rad_data_list = []
        BT_data_list = []
        for file in files:
            radiance, BT = H8_file_read(file)
            rad_data_list.append(radiance)
            BT_data_list.append(BT)
        radiance = np.vstack(rad_data_list)
        BT = np.vstack(BT_data_list)
        data_key = file_dict[key][1]
        d[data_key] = BT.astype(np.float32)
        data_key = file_dict[key][2]
        d[data_key] = radiance.astype(np.float32)

    return d


def get_path(root, band, time_key=None, path_tree=None):
    """Finds path for given time key and data band
    time_key: 201501081300, YYYYMMDDHHMM
    band: B07,MIR (3.9um), B14,TIR (11um), B03,red (0.6um)
    sometime B03 has both 500m and 2km resolution files
    variable: BT, brightness temperature; Radiance
    for static data return path
    path_tree: HSFD, the original japan FTP path like:#
        201501/09/201501090000/00/B03
    else,weidong own path tree like
        201501090000
    """

    if time_key is not None:  # EO realtime date
        # separate the date and time from the time_key
        dt_time_key = datetime.datetime.strptime(time_key, '%Y%m%d%H%M')
        dt_date = dt_time_key.strftime('%Y%m%d')
        dt_time = dt_time_key.strftime('%H%M')
        # keys = [dt_date, dt_time, band]  # Realtime EO channels
        if path_tree in ['HSFD']:
            root = root + \
                   '/'.join([dt_time_key.strftime('%Y%m'), \
                             dt_time_key.strftime('%d'), \
                             dt_time_key.strftime('%Y%m%d%H') + '00', \
                             dt_time_key.strftime('%M'), band]) + '/'
            # 500m resolution data
            band_vis_05 = band + "_FLDK_R05_S"
            # 2km resolution data
            band = band + "_FLDK_R20_S"
        else:
            root = root + time_key + "/"
            band_vis_05 = band

    else:
        root = root + "lcov/"
        band_vis_05 = band
    print "root: %s" % root
    print "band: %s" % band
    # now iterate over root path
    if os.path.exists(root):
        filepath = []
        filepath1 = []

        for f in os.listdir(root):
            if band in f:
                file_size = os.path.getsize(root + f)
                if file_size > 10000:
                    filepath.append(root + f)
            elif band_vis_05 in f:
                filepath1.append(root + f)
            else:
                continue
        if len(filepath) < 1:
            filepath = filepath1
            # return
        # if filepath1 is not None:
        #     return filepath1
        # else:
        return filepath
    else:
        print root, 'does not exists'
        sys.exit


def paths(root, time_key=None, path_tree=None, mode=0):
    """Constructs a dictionary for
    the file paths
    """

    # path dictionary construted from here
    d = {}
    if time_key is not None:  # EO realtime date
        # for fire detection model
        if mode == 0:
            d["red_path"] = [get_path(root, "B03", \
                                      time_key=time_key, path_tree=path_tree), 'vis', 'redrad']
            # d["nir_path"] = [get_path(root, "B04", \
            #     time_key=time_key,path_tree=path_tree), 'nir', 'nirrad']
            d["sir_path"] = [get_path(root, "B06", \
                                      time_key=time_key, path_tree=path_tree), 'sir', 'sirrad']
        d["mir_path"] = [get_path(root, "B07", \
                                  time_key=time_key, path_tree=path_tree), 'ir39', 'ir39rad']
        d["tir11_path"] = [get_path(root, "B14", \
                                    time_key=time_key, path_tree=path_tree), 'ir11', 'ir11rad']
        d["tir12_path"] = [get_path(root, "B15", \
                                    time_key=time_key, path_tree=path_tree), 'ir12', 'ir12rad']
    else:
        d["latlon_path"] = [get_path(root, "lat_lon.img"), 'lat', 'lon']
        d["sat_view_angle_path"] = [get_path(root, "vza_vaa.img"), 'vza', 'vaa']
        d["landcover_path"] = [get_path(root, "lcov.img"), 'lcov']
        # d["fixed_position_path"] = [get_path(root, "H8_tir_201501090620.img"),'fpos']

    return d


def load_h8(in_root, time_key, path_tree=None, mode=0):
    """
    load all the data and put them in a dictionary
    """
    # firstly setup the path dictionary
    EO_path_dict = paths(in_root, time_key=time_key, path_tree=path_tree, mode=0)
    # readin all the Himawari files here
    EO_data = Himawari_read(EO_path_dict)
    # construt a static data dictionary
    static_path_dict = paths(in_root)
    # readin all static data here
    static_data = static_read(static_path_dict)
    # get the sun angle
    szen, saa = sun_angles(static_data['lat'], static_data['lon'], time_key)
    # get the sun glint angle
    sun_glint = sunglint(static_data['vza'], static_data['vaa'], szen, saa)
    # combine EO and static data together
    EO_data.update(static_data)
    EO_data['szen'] = szen
    EO_data['sun_glint'] = sun_glint
    EO_data['ACQTIME'] = np.zeros(EO_data['ir39'].shape, dtype=np.int8)
    # for fire detection
    EO_data['diff'] = EO_data['ir39'] - EO_data['ir11']
    EO_data['tirradratio'] = EO_data['ir39rad'] / EO_data['ir11rad']
    EO_data['visradratio'] = EO_data['ir39rad'] / EO_data['redrad']
    # d['ndvi'] = (d['nir'] - d['vis']) / (d['nir'] + d['vis'])

    # correct the navigation problem
    dt_time_key = datetime.datetime.strptime(time_key, '%Y%m%d%H%M')
    dt_time = int(dt_time_key.strftime('%H'))
    # if dt_time < 11:
    #     doit = img_move(EO_data)


    # do the cloud masking
    data = cloud_mask(EO_data)
    # data = water_mask(EO_data)
    return data





if __name__ == '__main__':
    user_home = os.environ['HOME']
    in_root = user_home + '/data/Himawari8/'
    # root for the output files
    out_root = in_root + 'fire_result/'
    # slot time YYYYMMDDHHMM
    time_key = "201510040230"
    data = load_h8(in_root, time_key, path_tree="HSFD")

