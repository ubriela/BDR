import time
import os

import numpy as np
import math
import random
import time

from maxcover import max_cover

from sets import Set

# import multiprocessing as mult
import sys
import re
import logging
from Exp import Exp
from Params import Params
from DataParser import read_data, read_image_data
from Grid_standard import Grid_standard
from Kd_standard import Kd_standard
from Quad_standard import Quad_standard
from knapsack import zeroOneKnapsack
from UtilsBDR import cell_coord

from VideoLevelExp import compute_urgency, data_readin

# sys.path.append('../../../../../../_Research/_Crowdsourcing/_Privacy/privategeocrowddynamic/src/common')
sys.path.append('../plot/code')

from Utils import rect_area,zipf_pmf

seed_list = [2172]
# seed_list = [9110, 4064, 6903, 7509, 5342, 3230, 3584, 7019, 3564, 6456]

method_list = None
exp_name = None
dataset_identifier = "_fov_mediaq"


"""
universe = Set([1,2,3,4,5])
weights = {1:1,2:1,3:2,4:3,5:1}
all_sets = {}
all_sets[0] = Set([1,2,3])
all_sets[1] = Set([2,4])
all_sets[2] = Set([3,4])
all_sets[3] = Set([4,5])
budget = 1

print max_cover(universe, all_sets, budget, weights)
"""
def optimization(tree, fov_count, seed, param):
    np.random.seed(seed)
    fovs = read_image_data(param.dataset)

    universe = Set([i for i in range(param.GRID_SIZE*param.GRID_SIZE)])
    all_sets = {}
    for i in range(len(fovs)):
        all_sets[i] = fovs[i].cellids(param)

    weights = {}
    count = 0
    last_weight = 0
    for item in universe:
        coord = cell_coord(item, param)
        # print coord
        leaf_node = tree.leafCover(coord)
        if leaf_node is not None:
            compute_urgency(leaf_node)
            boundary = np.array([[param.x_min, param.y_min],[param.x_max, param.y_max]])
            coverage_ratio = min(1, rect_area(boundary)/(param.GRID_SIZE*param.GRID_SIZE*rect_area(leaf_node.n_box)))
            weights[item] = leaf_node.urgency*coverage_ratio
            last_weight = weights[item]
        else:
            weights[item] = last_weight
            count = count + 1
    # print count
    # print universe
    # print all_sets
    # print fov_count
    # print weights

    start = time.time()

    covered_sets, covered_items, covered_weight = max_cover(universe, all_sets, fov_count, weights)
    print "time ", time.time() - start

    print covered_weight
    return covered_weight


def eval_partition(data, param):
    # tree = Grid_standard(data, param)
    tree = Quad_standard(data, param)
    # tree = Kd_standard(data, param)
    tree.buildIndex()

    seed = 1000
    fov_count = 20

    print optimization(tree, fov_count, seed, param)

def eval_analyst(data, param):
    logging.info("eval_analyst")
    exp_name = "eval_analyst"

    analyst = [4,5,6,7,8]
    fov_count = 200    # fixed
    method_list = ['grid_standard', 'quad_standard', 'kd_standard']

    res_cube_value = np.zeros((len(analyst), len(seed_list), len(method_list)))

    for j in range(len(seed_list)):
        param.seed = seed_list[j]
        for i in range(len(analyst)):
            param.part_size = analyst[i]
            param.ANALYST_COUNT = analyst[i] * analyst[i]
            for k in range(len(method_list)):
                if method_list[k] == 'grid_standard':
                    tree = Grid_standard(data, param)
                elif method_list[k] == 'quad_standard':
                    tree = Quad_standard(data, param)
                elif method_list[k] == 'kd_standard':
                    tree = Kd_standard(data, param)
                else:
                    logging.error('No such index structure!')
                    sys.exit(1)
                tree.buildIndex()

                res_cube_value[i, j, k] = optimization(tree, fov_count, seed_list[j], param)

    res_value_summary = np.average(res_cube_value, axis=1)
    np.savetxt(param.resdir + exp_name + dataset_identifier , res_value_summary, fmt='%.4f\t')

def eval_bandwidth(data, param):
    logging.info("eval_bandwidth")
    exp_name = "eval_bandwidth"

    analyst = 6
    param.part_size = analyst
    param.ANALYST_COUNT = analyst * analyst

    fov_count = [10,15,20,25,30]
    method_list = ['grid_standard', 'quad_standard', 'kd_standard']

    res_cube_value = np.zeros((len(fov_count), len(seed_list), len(method_list)))

    for j in range(len(seed_list)):
        param.seed = seed_list[j]
        for i in range(len(fov_count)):
            for k in range(len(method_list)):
                if method_list[k] == 'grid_standard':
                    tree = Grid_standard(data, param)
                elif method_list[k] == 'quad_standard':
                    tree = Quad_standard(data, param)
                elif method_list[k] == 'kd_standard':
                    tree = Kd_standard(data, param)
                else:
                    logging.error('No such index structure!')
                    sys.exit(1)
                tree.buildIndex()

                res_cube_value[i, j, k] = optimization(tree, fov_count[i], seed_list[j], param)

    res_value_summary = np.average(res_cube_value, axis=1)
    np.savetxt(param.resdir + exp_name + dataset_identifier, res_value_summary, fmt='%.4f\t')


"""
varying skewness of video size
"""
def eval_skewness(data, param):
    logging.info("eval_skewness")
    exp_name = "eval_skewness"

    analyst = 6
    param.part_size = analyst
    param.ANALYST_COUNT = analyst * analyst
    fov_count = 200    # fixed

    skewness = [1.6,1.8,2.0,2.2,2.4]
    method_list = ['grid_standard', 'quad_standard', 'kd_standard']

    res_cube_value = np.zeros((len(skewness), len(seed_list), len(method_list)))

    for j in range(len(seed_list)):
        param.seed = seed_list[j]
        for i in range(len(skewness)):
            param.ZIPFIAN_SKEW = skewness[i]
            for k in range(len(method_list)):
                if method_list[k] == 'grid_standard':
                    tree = Grid_standard(data, param)
                elif method_list[k] == 'quad_standard':
                    tree = Quad_standard(data, param)
                elif method_list[k] == 'kd_standard':
                    tree = Kd_standard(data, param)
                else:
                    logging.error('No such index structure!')
                    sys.exit(1)
                tree.buildIndex()

                res_cube_value[i, j, k] = optimization(tree, fov_count, seed_list[j], param)

    res_value_summary = np.average(res_cube_value, axis=1)
    np.savetxt(param.resdir + exp_name + dataset_identifier , res_value_summary, fmt='%.4f\t')



if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG, filename='../../log/debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")

    param = Params(1000)
    data = data_readin(param)
    param.NDIM, param.NDATA = data.shape[0], data.shape[1]
    param.LOW, param.HIGH = np.amin(data, axis=1), np.amax(data, axis=1)

    eval_partition(data, param)

    # eval_analyst(data, param)
    # eval_bandwidth(data, param)
    # eval_skewness(data, param)


    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  END")