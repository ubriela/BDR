__author__ = 'ubriela'

import numpy as np
import re

"""
videoFileName FOVNUM Lat Log ThetaX ThetaY ThetaZ R Alpha Timestamp
"""
def clean_data(fileIn, fileOut):
    str = ""    # all data
    str_locs = "" # locations only
    fo = open(fileOut,"w")
    fo_locs = open(re.sub(r'\.txt$', '', fileOut) + ".dat", "w")
    with open(fileIn) as f:
        for line in f:
            arr = line.split(" ")
            print arr[2], arr[3]
            if 30 < float(arr[2]) < 35 and -120 < float(arr[3]) < -115:
                str += line
                str_locs += arr[2] + "\t" + arr[3] + "\n"

    fo.write(str)
    fo_locs.write(str_locs)

    fo.close()
    fo_locs.close()


if True:
    clean_data("../../dataset/mediaq_pbs/pbs.txt", "../../dataset/mediaq_pbs/pbs_cleaned.txt")
