__author__ = 'ubriela'

import sys
import numpy as np

from cdi_parser import cdi_parse
from nresp_parser import nresp_parse

sys.path.append('../../src/bdr')
from DataParser import read_data

cdi_arr = cdi_parse()
nresp_arr = nresp_parse()

# print cdi
# print nresp

N = 10000
t_nresp = sum([i[1] for i in cdi_arr])
delta_lng = 0.0114074
delta_lat = 0.0089928

videos = read_data('../../dataset/napa/napa_fov.csv')
print len(videos)

filtered_videos = []
for x in cdi_arr:
    cdi, nresp, lat, lng = x[0], x[1], x[2], x[3]
    nvideo = round(((nresp + 0.0)/t_nresp) * N)
    min_lat, min_lng, max_lat, max_lng = lat - delta_lat/2, lng - delta_lng/2, lat + delta_lat/2, lng + delta_lng/2

    # print nvideo

    # select nvideo videos in the corresponding grid
    nvideo_count = 0
    for v in videos:
        latv, lngv = v.fovs[0].lat, v.fovs[0].lon
        if min_lat <= latv <= max_lat and min_lng <= lngv <= max_lng:
            filtered_videos.append(v)
            nvideo_count = nvideo_count + 1
            # if nvideo_count == nvideo: # enough data point
            #     break

np.random.shuffle(filtered_videos)

t_nresp = sum(nresp_arr)
t = 0
vid = 0
for x in nresp_arr:
    fv = int(round((x + 0.0)/t_nresp * len(filtered_videos)))
    print fv
    # str_videos = ""
    # for i in range(fv):
    #     # print filtered_videos[vid].to_str()
    #     str_videos = str_videos + filtered_videos[vid].to_str()
    #     vid = vid + 1
    #
    #     if vid >= len(filtered_videos):
    #         break
    #
    # # output into a file
    # text_file = open("../../dataset/napa/snapshots/" + str(t) + ".txt", "w")
    # text_file.write(str_videos)
    # text_file.close()
    #
    # t = t + 1


# for v in filtered_videos:
#     print v.fovs[0].lat, '\t', v.fovs[0].lon