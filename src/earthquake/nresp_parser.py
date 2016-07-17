__author__ = 'ubriela'

import sys
import numpy as np

sys.path.append('../../src/bdr')
from Params import Params

def nresp_parse(nresp_file = '../../dataset/napa/numresp.txt'):
    TOTAL_TIME_SNAPSHOT = 80
    data = np.genfromtxt(nresp_file, unpack=True)
    idx = 0
    nresps = np.zeros(Params.TIME_SNAPSHOT)
    min_time = 0.035 # ~ 0 hour
    max_time = 7.998 # ~ 8 hours
    time_duration = (max_time - min_time) / TOTAL_TIME_SNAPSHOT
    for resp in data[1]:
        time = float(data[0][idx])
        index = int((time - min_time) / time_duration)
        if index >= Params.TIME_SNAPSHOT:
            break
        # print time_duration, time, index
        nresps[index] = int(nresps[index] + 1)
        #nresp = int(resp)

        idx = idx + 1
    return nresps