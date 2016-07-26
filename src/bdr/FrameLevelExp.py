import time
import os

import numpy as np
import math
import random
import time
from collections import deque
from maxcover import max_cover

from sets import Set
import heapq

# import multiprocessing as mult
import sys
import re
import logging
from Exp import Exp
from Params import Params
from DataParser import read_data
from Grid_standard import Grid_standard
from Kd_standard import Kd_standard
from Quad_standard import Quad_standard
from knapsack import zeroOneKnapsack
from UtilsBDR import cell_coord

from VideoLevelExp import compute_urgency, data_readin

# sys.path.append('../../../../../../_Research/_Crowdsourcing/_Privacy/privategeocrowddynamic/src/common')
sys.path.append('../plot/code')

from Utils import rect_area,zipf_pmf
from Utils import is_rect_cover

seed_list = [2172]
# seed_list = [9110, 4064, 6903, 7509, 5342, 3230, 3584, 7019, 3564, 6456]

method_list = None
exp_name = None
dataset_identifier = "_fov_gau"

def gen_fovs(param):
    np.random.seed(param.seed)
    fovs_file = re.sub(r'\.dat$', '', param.dataset) + ".txt"
    videos = read_data(fovs_file)

    fovs = []
    for v in videos:
        v.size = np.random.zipf(param.ZIPFIAN_SKEW)
        if v.size > 10:
            v.size = 10
        v.fov_count = int(v.size)
        fovs = fovs + v.get_fovs()
        # else:
        #     print "not a leaf node", v.location()
    return fovs

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
def fovs_info(fovs):
    all_sets = {}
    for i in range(len(fovs)):
        all_sets[i] = Set(fovs[i].cellids(param))

    universe = Set([])
    for s in all_sets.values():
        universe = universe | s

    weights = {}
    boundary = np.array([[param.x_min, param.y_min],[param.x_max, param.y_max]])
    unit_cell_area = rect_area(boundary)/(param.GRID_SIZE*param.GRID_SIZE)
    for item in universe:
        weights[item] = random.randint(0,10) * unit_cell_area

    return all_sets, universe, weights

def eval_partition(data, param):
    # tree = Grid_standard(data, param)
    # tree = Quad_standard(data, param)
    tree = Kd_standard(data, param)
    tree.buildIndex()

    seed = 1000
    fov_count = 200

    print optimization(tree, fov_count, seed, param)

def getLeafNode(tree, type):
    leaf_boxes = []
    if type == 1:
        for l1_child in tree.root.children:
            # print len(tree.root.children)
            if not l1_child.n_isLeaf and l1_child.children is not None:
                # print len(l1_child.children)
                for l2_child in l1_child.children:  # child1 is a first-level cell
                    leaf_boxes.append((l2_child.n_box, l2_child.n_count))
            # leaf_boxes.append((l1_child.n_box, l1_child.n_count))
    elif type == 2:
        queue = deque()
        queue.append(tree.root)
        while len(queue) > 0:
            curr = queue.popleft()
            if curr.n_isLeaf is False:
                queue.append(curr.nw)
                queue.append(curr.ne)
                queue.append(curr.sw)
                queue.append(curr.se)
            else:
                leaf_boxes.append((curr.n_box, curr.n_count))

    return leaf_boxes

def heapsort(iterable):
    h = []
    for value in iterable:
        heapq.heappush(h, value)
    return [heapq.heappop(h) for i in range(len(h))]

# varying the number of analysts, measure the total visual awareness
def eval_analyst(param):
    logging.info("eval_analyst")
    exp_name = "eval_analyst"

    analyst = [4,5,6,7,8]
    analyst_capacity = 40
    method_list = ['grid_standard', 'quad_standard', 'kd_standard']

    res_cube_value = np.zeros((len(analyst), len(seed_list), len(method_list)))

    for j in range(len(seed_list)):
        param.seed = seed_list[j]
        fovs = gen_fovs(param)  # generate videos given a seed
        for k in range(len(method_list)):
            # all fovs' locations
            locs = np.zeros((2, len(fovs)))
            for l in range(len(fovs)):
                locs[0][l], locs[1][l] = fovs[l].lat, fovs[l].lon

            for i in range(len(analyst)):
                param.part_size = analyst[i]
                param.ANALYST_COUNT = analyst[i] * analyst[i]
                bandwidth = param.ANALYST_COUNT * analyst_capacity

                if method_list[k] == 'grid_standard': # if grid --> partition first, select later
                    tree = Grid_standard(locs, param)
                elif method_list[k] == 'quad_standard':
                    tree = Quad_standard(locs, param)
                elif method_list[k] == 'kd_standard':
                    # upload best fovs
                    # print len(fovs)
                    all_sets, universe, weights = fovs_info(fovs)
                    # print len(all_sets), len(universe), len(weights)
                    covered_sets, covered_items, covered_weight = max_cover(universe, all_sets, bandwidth, weights)

                    # locations of the uploaded fovs
                    uploaded_fovs = [fovs[idx] for idx in covered_sets]

                    uploaded_locs = np.zeros((2, len(uploaded_fovs)))
                    for l in range(len(uploaded_fovs)):
                        uploaded_locs[0][l], uploaded_locs[1][l] = uploaded_fovs[l].lat, \
                                                                   uploaded_fovs[l].lon

                    tree = Kd_standard(uploaded_locs, param)
                else:
                    logging.error('No such index structure!')
                    sys.exit(1)

                tree.buildIndex()
                all_values = []

                # get leaf nodes (work cells)
                if method_list[k] == 'quad_standard' or method_list[k] == 'kd_standard':
                    leaf_nodes = getLeafNode(tree, 2)
                else:
                    leaf_nodes = getLeafNode(tree, 1)

                # each analyst chooses the best fovs in their assigned work cells
                for (n_box, count) in leaf_nodes:
                    if count > 0:
                        leaf_fovs = []
                        for fov in fovs:
                            if is_rect_cover(n_box, [fov.lat, fov.lon]):
                                leaf_fovs.append(fov)
                        if len(leaf_fovs) > 0:
                            leaf_all_sets, leaf_universe, leaf_weights = fovs_info(fovs)
                            leaf_covered_sets, leaf_covered_items, leaf_covered_weight = max_cover(leaf_universe, leaf_all_sets, bandwidth, leaf_weights)
                            all_values.append(leaf_covered_weight)

                heap = heapsort(all_values)
                for h in range(len(all_values) - param.ANALYST_COUNT):
                    heapq.heappop(heap)

                res_cube_value[i, j, k] = sum(heap)

    res_value_summary = np.average(res_cube_value, axis=1)
    np.savetxt(param.resdir + exp_name + dataset_identifier , res_value_summary, fmt='%.4f\t')


# varying the bandwidth constraint
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

    # print data
    # eval_partition(data, param)

    eval_analyst(param)
    # eval_bandwidth(data, param)
    # eval_skewness(data, param)


    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  END")