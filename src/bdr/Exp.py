import time
import logging

import numpy as np

from Kd_standard import Kd_standard
from Quad_standard import Quad_standard


class Exp(object):
    def __init__(self, data, query_list):
        self.data = data
        self.query_list = query_list
        logging.debug('Getting true query answers...')
        self.trueRes = np.array([self.getTrue(query) for query in query_list])

    def getTrue(self, query):
        """Get true answer by linear search along each dimension"""
        _data = self.data.copy()
        _ndim = _data.shape[0]
        for dim in range(_ndim):
            if _data.shape[1] == 0:
                break
            idx = np.argsort(_data[dim, :], kind='mergesort')
            _data[:, :] = _data[:, idx]
            x = np.searchsorted(_data[dim, :], query[0, dim], side='left')
            y = np.searchsorted(_data[dim, :], query[1, dim], side='right')
            _data = _data[:, x:y + 1]
        return _data.shape[1]

    def run_Kd_standard(self, param):
        logging.debug('building Kd_standard...')
        tree = Kd_standard(self.data, param)
        start = time.clock()
        tree.buildIndex()
        end = time.clock()
        logging.info('[T] Kd-standard building time: %.2f' % (end - start))
        return self.query(tree)


    def run_Quad_standard(self, param):
        logging.debug('building Quad_baseline...')
        tree = Quad_standard(self.data, param)
        start = time.clock()
        tree.buildIndex()
        end = time.clock()
        logging.info('[T] Quad_standard building time: %.2f' % (end - start))
        return self.query(tree)


    def query(self, tree):
        """ wrapper for query answering and computing query error """
        #        i_whole, l_whole, l_part = 0, 0, 0
        #        result = []
        #        for query in self.query_list:
        #            res, iw, lw, lp = tree.rangeCount(query)
        #            result.append(res)
        #            i_whole += iw
        #            l_whole += lw
        #            l_part += lp
        #        Res = np.array(result)
        #        print 'i_whole:', float(i_whole)/len(self.query_list)
        #        print 'l_whole:', float(l_whole)/len(self.query_list)
        #        print 'l_part:', float(l_part)/len(self.query_list)
        result = []
        for query in self.query_list:
            result.append(tree.rangeCount(query))
        Res = np.array(result)
        return self.computeError(Res)

    def computeError(self, Res):
        """ Compute median absolute and relative errors """
        absErr = np.abs(Res - self.trueRes)
        idx_nonzero = np.where(self.trueRes != 0)
        absErr_nonzero = absErr[idx_nonzero]
        true_nonzero = self.trueRes[idx_nonzero]
        relErr = absErr_nonzero / true_nonzero
        absErr = np.sort(absErr)
        relErr = np.sort(relErr)
        print Res, self.trueRes
        n_abs = len(absErr)
        n_rel = len(relErr)
        return absErr[int(n_abs / 2)], relErr[int(n_rel / 2)]
