__author__ = 'ubriela'


import time
import os
import random
import math
import numpy as np
import sys
import logging
from Exp import Exp
from Params import Params
from Video import Video
from FOV import FOV
from Utils import zipf_pmf

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
    for vid in data[1]:
        if vid == prev_vid:
            fov = FOV(data[3][idx],data[4][idx],data[10][idx],data[12][idx],data[13][idx])
            fovs.append(fov)
        else:
            # new video
            v = Video(fovs)
            v.id = vid
            videos.append(v)
            # print v.to_str()

            # new fovs
            fovs = []

        idx = idx + 1
        prev_vid = vid

    return videos

if True:

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

