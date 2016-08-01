import time
import os

import numpy as np
import math
import random
import time
import heapq
from collections import deque
from maxcover import max_cover

from sets import Set

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

from Utils import rect_area

from VideoLevelExp import compute_urgency, data_readin

# sys.path.append('../../../../../../_Research/_Crowdsourcing/_Privacy/privategeocrowddynamic/src/common')
sys.path.append('../plot/code')

from Utils import rect_area,zipf_pmf

seed_list = [9110, 4064, 6903]
# seed_list = [9110, 4064, 6903, 7509, 5342, 3230, 3584, 7019, 3564, 6456]

analyst = [4, 5, 6, 7, 8]
analyst_count = 8

# each analyst can handle an amount of work
capacity = [2, 3, 4, 5, 6]
analyst_capacity = 2


method_list = None
exp_name = None


def data_readin2(p):
    """Read in spatial data and initialize global variables."""
    data = np.genfromtxt(p.dataset, unpack=True)
    # data = sample_data(data, p)
    p.NDIM, p.NDATA = data.shape[0], data.shape[1]
    p.LOW, p.HIGH = np.amin(data, axis=1), np.amax(data, axis=1)
    logging.debug(data.shape)
    logging.debug(p.LOW)
    logging.debug(p.HIGH)
    return data[0:2]

def parse_urgency_map(file='../../dataset/napa/urgency_total.csv'):
    data = np.genfromtxt(file, unpack=True, delimiter=',')
    # print data.shape[0], data.shape[1]
    # print np.average(data), np.amax(data), np.std(data)
    return data

urgency_map= parse_urgency_map()
SHAKEMAP_LAT_SIZE = 201
SHAKEMAP_LON_SIZE = 301
min_lat, min_lon, max_lat, max_lon = 37.3822, -123.5617, 39.0488, -121.0617
delta_lat = (max_lat - min_lat)/SHAKEMAP_LAT_SIZE
delta_lon = (max_lon - min_lon)/SHAKEMAP_LON_SIZE

"""
given a location, determine its urgency value
"""
def urgency_value(lat, lon):
    lat_index = int(round((lat - min_lat)/delta_lat))
    lon_index = int(round((lon - min_lon)/delta_lon))
    return urgency_map[lon_index][lat_index]


"""
this set <unit cell id, visual awareness> contains the selected unit cells
and their updated values of visual awareness.
"""
updated_unit_cells = {}
updated_unit_cells_count = {}


def getLeafNode(tree, type):
    leaf_nodes = []
    if type == 1:
        for l1_child in tree.root.children:
            # print len(tree.root.children)
            if not l1_child.n_isLeaf and l1_child.children is not None:
                # print len(l1_child.children)
                for l2_child in l1_child.children:  # child1 is a first-level cell
                    leaf_nodes.append(l2_child)
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
                leaf_nodes.append(curr)

    return leaf_nodes

def heapsort(iterable):
    h = []
    for value in iterable:
        heapq.heappush(h, value)
    return [heapq.heappop(h) for i in range(len(h))]

def fovs_info(fovs):
    # compute all sets for fovs
    all_sets = {}
    for i in range(len(fovs)):
        all_sets[i] = Set(fovs[i].cellids(param))

    # compute universe set as the union of all sets
    universe = Set([])
    for s in all_sets.values():
        universe = universe | s

    # weights corresponds to urgency values
    weights = {}

    # visual awareness is computed based on cell overlap
    unit_cell_area = rect_area([[param.x_min, param.y_min], [param.x_max, param.y_max]]) / param.GRID_SIZE ** 2
    # shakemap_cell_area = rect_area([[min_lat, min_lon],[max_lat, max_lon]]) / (SHAKEMAP_LAT_SIZE * SHAKEMAP_LON_SIZE)
    # cell_ratio = unit_cell_area/shakemap_cell_area

    # update values in universe & updated_unit_cells
    for item in universe:
        if not updated_unit_cells.has_key(item):  # add the unit cell
            # compute visual awareness
            coord = cell_coord(item, param)
            updated_unit_cells[item] = weights[item] = urgency_value(coord[0], coord[1]) * unit_cell_area
            updated_unit_cells_count[item] = 1
        else:
            # update the value of the element
            weights[item] = updated_unit_cells[item]
            updated_unit_cells_count[item] = updated_unit_cells_count[item] + 1

    return all_sets, universe, weights

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
def optimization(param,seed,timesnapshot,method, bandwidth, analyst_capacity):

    np.random.seed(seed)

    # update video file
    param.dataset = param.datadir + str(timesnapshot) + ".txt"

    # read videos
    videos = read_data(param.dataset)
    param.NDIM = 2

    # extract a set of fovs
    fovs = []
    for v in videos:
        v.size = np.random.zipf(param.ZIPFIAN_SKEW)
        if v.size > 10:
            v.size = 10
        v.fov_count = int(v.size)
        fovs = fovs + v.get_fovs()

    h = {}
    for fov in fovs:
        h[str(fov.lat) + ";" + str(fov.lon)] = fov

    # all fovs' locations
    locs = np.zeros((2, len(fovs)))
    for l in range(len(fovs)):
        locs[0][l], locs[1][l] = fovs[l].lat, fovs[l].lon

    all_sets, universe, weights = fovs_info(fovs)

    covered_sets, covered_items, covered_weight = max_cover(universe, all_sets, bandwidth, weights)
    uploaded_fovs = [fovs[idx] for idx in covered_sets]
    uploaded_locs = np.zeros((2, len(uploaded_fovs)))
    for l in range(len(uploaded_fovs)):
        uploaded_locs[0][l], uploaded_locs[1][l] = uploaded_fovs[l].lat, \
                                                   uploaded_fovs[l].lon

    if method == 'grid_standard':  # if grid --> partition first, select later
        tree = Grid_standard(uploaded_locs, param)
    elif method == 'quad_standard':
        tree = Quad_standard(uploaded_locs, param)
    elif method == 'kd_standard':
        tree = Kd_standard(uploaded_locs, param)
    else:
        logging.error('No such index structure!')
        sys.exit(1)

    tree.buildIndex()

    # get leaf nodes (work cells)
    if method == 'quad_standard' or method == 'kd_standard':
        leaf_nodes = getLeafNode(tree, 2)
    else:
        leaf_nodes = getLeafNode(tree, 1)

    all_values = []
    for node in leaf_nodes:
        # if node.n_count > 0:
        leaf_fovs = []
        # if method_list[k] == 'grid_standard' or method_list[k] == 'quad_standard':
        if node.n_data is not None:
            for l in range(node.n_data.shape[1]):
                leaf_fovs.append(h[str(node.n_data[0][l]) + ";" + str(node.n_data[1][l])])
        # else:
        #     for fov in fovs:
        #         if is_rect_cover(node.n_box, [fov.lat, fov.lon]):
        #             leaf_fovs.append(fov)

        if len(leaf_fovs) > 0:
            leaf_all_sets, leaf_universe, leaf_weights = fovs_info(leaf_fovs)
            # update weight in leaf_weights
            for item in leaf_universe:
                leaf_weights[item] = weights[item]
            leaf_covered_sets, leaf_covered_items, leaf_covered_weight = max_cover(leaf_universe, leaf_all_sets,
                                                                                   analyst_capacity, leaf_weights)
            all_values.append(leaf_covered_weight)

            # reduce VA value of the cells in covered_items by half
            for item in leaf_covered_items:
                updated_unit_cells[item] = updated_unit_cells[item] / updated_unit_cells_count[item]

    heap = heapsort(all_values)
    for o in range(len(all_values) - param.ANALYST_COUNT):
        heapq.heappop(heap)

    return sum(heap)

def multi_optimization(param, seed, method, analyst_capacity, analyst_count):
    bandwidth = analyst_capacity * analyst_count
    global updated_unit_cells

    updated_unit_cells = {}
    total_va = 0
    for time in range(param.TIME_SNAPSHOT):
        current_va = optimization(param,seed,time,method,bandwidth, analyst_capacity)
        total_va = total_va + current_va
        print current_va
    print total_va
    return total_va


# varying the number of analysts, measure the total visual awareness
def multi_eval_analyst(param):
    logging.info("multi_eval_analyst")
    exp_name = "multi_eval_analyst"

    method_list = ['grid_standard', 'quad_standard', 'kd_standard']

    res_cube_value = np.zeros((len(analyst), len(seed_list), len(method_list)))

    for j in range(len(seed_list)):
        param.seed = seed_list[j]
        for i in range(len(analyst)):
            param.part_size = analyst[i]
            param.ANALYST_COUNT = analyst[i] * analyst[i]
            for k in range(len(method_list)):
                res_cube_value[i, j, k] = multi_optimization(param, seed_list[j], method_list[k], analyst_capacity, param.ANALYST_COUNT)

    res_value_summary = np.average(res_cube_value, axis=1)
    np.savetxt(param.resdir + exp_name + '_' + param.DATASET , res_value_summary, fmt='%.4f\t')

# varying the bandwidth constraint
def multi_eval_capacity(param):
    logging.info("multi_eval_capacity")
    exp_name = "multi_eval_capacity"

    method_list = ['grid_standard', 'quad_standard', 'kd_standard']

    res_cube_value = np.zeros((len(capacity), len(seed_list), len(method_list)))

    for j in range(len(seed_list)):
        param.seed = seed_list[j]
        for i in range(len(capacity)):
            param.part_size = analyst_count
            param.ANALYST_COUNT = analyst_count * analyst_count
            for k in range(len(method_list)):
                res_cube_value[i, j, k] = multi_optimization(param, seed_list[j], method_list[k], capacity[i], param.ANALYST_COUNT)

    res_value_summary = np.average(res_cube_value, axis=1)
    np.savetxt(param.resdir + exp_name + '_' + param.DATASET , res_value_summary, fmt='%.4f\t')


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG, filename='../../log/debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")

    param = Params(1000)
    param.select_dataset()
    # data = data_readin(param)
    # param.NDIM, param.NDATA = data.shape[0], data.shape[1]
    # param.LOW, param.HIGH = np.amin(data, axis=1), np.amax(data, axis=1)

    # multi_eval_analyst(param)
    multi_eval_capacity(param)



    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  END")