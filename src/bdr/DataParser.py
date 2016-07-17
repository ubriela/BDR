__author__ = 'ubriela'

import os
import numpy as np
from Params import Params
from Video import Video
from FOV import FOV
import urllib
import xml.etree.ElementTree as ET
import json

"""
filename,fileID,fovnum,Plat,Plng,Px,Py,prevX,prevY,speed,dir,prevDir,R,alpha,timestamp
videoid 1
fovid 2
lat 3
lon 4
dir 10
r 12
alpha 13
"""

def read_data(file):
    data = np.genfromtxt(file, unpack=True)

    prev_vid = 1
    fovs = []
    idx = 0
    videos = []
    video_id = 0
    for vid in data[1]:
        # print vid, prev_vid
        if vid == prev_vid:
            fov = FOV(data[3][idx],data[4][idx],data[10][idx],data[12][idx],data[13][idx])
            fovs.append(fov)
        else:
            # new video
            v = Video(fovs)
            v.id = video_id
            video_id = video_id + 1
            videos.append(v)
            # print v.to_str()

            # new fovs
            fovs = []
            fov = FOV(data[3][idx],data[4][idx],data[10][idx],data[12][idx],data[13][idx])
            fovs.append(fov)

        idx = idx + 1
        prev_vid = vid

    return videos

"""
lat 0
lon 1
dir 2
"""
def read_image_data(file):
    data = np.genfromtxt(file, unpack=True)

    idx = 0
    fovs = []
    for i in range(0,data.shape[1]):
        fov = FOV(data[0][idx],data[1][idx],data[2][idx], 60, 250)
        fov.id = idx
        idx = idx + 1
        fovs.append(fov)

    return fovs


"""
LON,LAT,PGA,PGV,MMI,PSA03,PSA10,PSA30,STDPGA,URAT,SVEL
LOT 0
LAT 1
MMI 2
"""

def read_shakemap_xml(url='http://earthquake.usgs.gov/earthquakes/shakemap/global/shake/20002926/download/grid.xml'):
    u = urllib.urlopen(url)
    # u is a file-like object
    data = u.read()
    root = ET.fromstring(data)
    for child in root.getchildren():
        if child.tag.endswith('event'):
            print child.attrib
        if child.tag.endswith('grid_specification'):
            print child.attrib["lat_min"], child.attrib["lon_min"], child.attrib["lat_max"], child.attrib["lon_max"]
            print child.attrib["nlat"], child.attrib["nlon"], child.attrib["nominal_lat_spacing"], child.attrib["nominal_lon_spacing"]
        if child.tag.endswith('grid_data'):
            grid_data = child.text
            rows = grid_data.split("\n")
            for row in rows:
                values = row.split(" ")
                print values[0], values[1], values[4]
                if len(values) == 11 and float(values[4]) > 7:
                    pass


if False:

    if False:    # read shakemap file
        read_shakemap_xml()

    if True:   # read video metadata
        param = Params(1000)
        param.select_dataset()
        videos = read_data(os.path.splitext(param.dataset)[0] + ".txt")

        video_locs = np.zeros((2,len(videos)))
        idx = 0
        for v in videos:
            vl = v.location()
            video_locs[0,idx] = vl[0]
            video_locs[1,idx] = vl[1]
            idx = idx + 1

        # print video_locs

        np.savetxt(param.dataset, video_locs.transpose(), fmt='%.4f\t')

        # print sum([v.size for v in videos])

        # for v in videos:
            # print v.area(), "\t", v.sum_fov_area()
            # print v.location()

