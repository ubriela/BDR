import logging

import numpy as np

from Grid import Grid
from Params import Params


class Grid_standard(Grid):

    def __init__(self, data, param):
        Grid.__init__(self, data, param)

        self.param.m = self.param.part_size
        self.param.maxHeightHTree = 2

        logging.debug("Grid_standard: size: %d" % self.param.m)


    def getCoordinates(self, curr):
        """ 
        get corrdinates of the point which defines the subnodes: 
        return split_arr: split points
	    n_data_arr: data in each partitions
        """
        _box = curr.n_box
        dimP = curr.n_depth % self.param.NDIM  # split dimension

        split_arr = self.getEqualSplit(self.param.m, _box[0, dimP], _box[1, dimP])
        # print self.param.m

        # get data points in these partitions
        n_data_arr = [None for _ in range(self.param.m)]
        _data = curr.n_data

        if _data is not None and _data.shape[1] >= 1:
            _idx = np.argsort(_data[dimP, :], kind='mergesort')
            _data[:, :] = _data[:, _idx]  # sorted by dimP dimension

            for i in range(self.param.m):
                posP1 = np.searchsorted(_data[dimP, :], split_arr[i])
                posP2 = np.searchsorted(_data[dimP, :], split_arr[i + 1])
                if i == 0:  # the first partition
                    n_data = _data[:, :posP2]
                elif i == len(split_arr) - 2:  # the last partition
                    n_data = _data[:, posP1:]
                else:
                    n_data = _data[:, posP1:posP2]
                n_data_arr[i] = n_data
        print len(split_arr), len(n_data_arr)
        return split_arr, n_data_arr
