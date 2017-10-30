"""
See: https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/lws-classic/api.php
for API commands

"""

import urllib
from xml.etree.ElementTree import parse  # py2.7 syntax
from datetime import datetime

import logging as log


def generic_request(method, base_url, **kargs):
    ret_list = []
    url = base_url + method
    if len(kargs) > 0:
        url += "?"

        for key, value in kargs.items():
            url += key + "=" + str(value) + "&"  # Extra & on the end doesn't seem to hurt anything

    log.info("REQUEST: %s" % (url))
    root = parse(urllib.urlopen(url)).getroot()

    for element in root:
        ret_list.append(element.text)

    return ret_list


def search_for_files(base_url,
                     product="MOD021KM",
                     collection="6",
                     start=datetime.utcnow().strftime("%Y-%m-%d"),
                     stop=datetime.utcnow().strftime("%Y-%m-%d"),
                     north="90", south="-90",
                     west="-180", east="180",
                     coordsOrTiles="coords",
                     dayNightBoth="DNB"):
    r_dict = {"product": product,
              "collection": collection,
              "start": start,
              "stop": stop,
              "north": north,
              "south": south,
              "east": east,
              "west": west,
              "coordsOrTiles": coordsOrTiles,
              "dayNightBoth": dayNightBoth}

    return generic_request("searchForFiles", base_url, **r_dict)


def get_file_urls(base_url, files):
    assert (len(files) > 0)
    file_id_str = ",".join(files)
    return generic_request("getFileUrls", base_url, fileIds=file_id_str)


def download_urls(urls):
    for url in urls:
        filename = url.split("/")[-1]
        log.info("Downloading to: %s" % (filename))
        urllib.urlretrieve(url, filename)


def main():

    #base_url = "http://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices/"  # laads
    base_url = "http://lance.modaps.eosdis.nasa.gov/axis2/services/MWSLance/"  # lance

    print generic_request("listProducts", base_url)

    # files_dundee = search_for_files(base_url, start=datetime.utcnow().strftime("%Y-%m-%d"),
    #                          stop=datetime.utcnow().strftime("%Y-%m-%d"),
    #                          north="56.47", south="56.46", west="-3", east="-2.9")
    # urls_dundee = get_file_urls(base_url, files_dundee)


    # files_global = search_for_files(base_url, start=datetime.utcnow().strftime("%Y-%m-%d"),
    #                          stop=datetime.utcnow().strftime("%Y-%m-%d"),
    #                          north="90", south="-90", west="-180", east="180")
    #
    # urls_global = get_file_urls(base_url, files_global)







if __name__ == "__main__":
    log.basicConfig(level=log.DEBUG)
    main()


