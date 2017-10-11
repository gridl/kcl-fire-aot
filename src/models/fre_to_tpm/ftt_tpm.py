
# TODO still need to make sure area is being calculated correctly
def compute_plume_area(utm_plume_polygon):
    # # TODO is sinusoidal proj good enough?  Yes it is: https://goo.gl/KE3tuY
    # # get extra accuracy by selecting an appropriate lon_0
    # m = Basemap(projection='sinu', lon_0=140, resolution='c')
    #
    # lons = (lons + 180) - np.floor((lons + 180) / 360) * 360 - 180;
    # zone = stats.mode(np.floor((lons + 180) / 6) + 1, axis=None)[0][0]
    # p = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84', datum='WGS84')
    #
    # # apply to shapely polygon
    # projected_plume_polygon_m = transform(m, plume_polygon)
    # projected_plume_polygon_p = transform(p, plume_polygon)

    # compute projected polygon area in m2
    # return projected_plume_polygon_m.area, projected_plume_polygon_p.area

    # we already have the plume polygon area
    return utm_plume_polygon.area


def compute_aod(plume_bounding_pixels, plume_mask, lats, lons):
    '''
    Resampling of the ORAC AOD data is required to remove the bowtie effect from the data.  We can then
    sum over the AODs contained with the plume.

    Resampling of the MODIS AOD data is required so that it is in the same projection as the ORAC AOD.
    With it being in the same projection we can replace low quality ORAC AODs with those from MXD04 products.
    We also need to get the background AOD data from MXD04 products as ORAC does not do optically thin
    retrievals well.

    Also, we need to check that the pixel area estimates being computed by compute_plume_area are reasonable.
    That can be done in this function, we just produce another area estimate from the resampled mask by getting
    vertices, getting the lat lons of the vertices, creating a shapely polygon from them and then computing the area.
    '''

    # create best AOD map from ORAC and MYD04 AOD

    # extract best AOD using plume mask

    # split into background and plume AODs

    # subtract background from plume

    # return plume AOD
