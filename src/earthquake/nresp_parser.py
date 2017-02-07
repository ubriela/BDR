__author__ = 'ubriela'

import sys
import numpy as np

sys.path.append('../../src/bdr')
from Params import Params

# http://earthquake.usgs.gov/archive/product/dyfi/nc72282711/us/1470094942810/nc72282711_plot_numresp.txt
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



def compare_twwet_dyfi(nresp_file = '../../dataset/napa/numresp.txt', tweet_file = '../../dataset/napa/tweet_report_time.txt'):
    data = np.genfromtxt(nresp_file, unpack=True)
    tweet = np.genfromtxt(tweet_file, unpack=True)
    idx, dyfi_count = 0, 0
    tweet_idx, tweet_count = 0, 0
    prev_time = 0
    for time in range(100,3700,100):
        while float(data[0][idx])* 3600 < time:
            dyfi_count = data[1][idx]
            idx = idx + 1

        # count number of tweet report before time t
        while tweet[tweet_idx] < time:
            tweet_count += 1
            tweet_idx += 1

        print time, '\t', dyfi_count, '\t', tweet_count

compare_twwet_dyfi()