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

seed_list = [2172]
# seed_list = [9110, 4064, 6903, 7509, 5342, 3230, 3584, 7019, 3564, 6456]

method_list = None
exp_name = None
dataset_identifier = "_fov_mediaq"


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
def optimization(param,fov_count,seed,timesnapshot):

    np.random.seed(seed)

    # update video file
    param.dataset = param.datadir + str(timesnapshot) + ".txt"

    # read location of the videos
    # data = data_readin2(param)
    # param.NDIM, param.NDATA = data.shape[0], data.shape[1]
    # param.LOW, param.HIGH = np.amin(data, axis=1), np.amax(data, axis=1)

    # create
    # if method == 'grid_standard':
    #     tree = Grid_standard(data, param)
    # elif method == 'quad_standard':
    #     tree = Quad_standard(data, param)
    # elif method == 'kd_standard':
    #     tree = Kd_standard(data, param)

    # read videos
    videos = read_data(param.dataset)

    # extract a set of fovs
    fovs = []
    for v in videos:
        # leaf_node = tree.leafCover(v.location())
        # if leaf_node:
        v.size = np.random.zipf(param.ZIPFIAN_SKEW)
        if v.size > 20:
            v.size = 20
        v.fov_count = int(v.size)
        fovs = fovs + v.get_fovs()
        # else:
        #     print "not a leaf node", v.location()

    # compute all sets for fovs
    all_sets = {}
    for i in range(len(fovs)):
        all_sets[i] = Set(fovs[i].cellids(param))
        if len(all_sets[i]) == 0:
            print all_sets[i]

    # compute universe set as the union of all sets
    universe = Set([])
    for s in all_sets.values():
        universe = universe | s
        # Set([i for i in range(param.GRID_SIZE*param.GRID_SIZE)])


    # weights corresponds to urgency values
    weights = {}

    # visual awareness is computed based on workcell
    # count = 0
    # last_weight = 0
    # for item in universe:
    #     coord = cell_coord(item, param)
    #     # print coord
    #     leaf_node = tree.leafCover(coord)
    #     if leaf_node is not None:
    #         compute_urgency(leaf_node)
    #         boundary = np.array([[param.x_min, param.y_min],[param.x_max, param.y_max]])
    #         coverage_ratio = min(1, rect_area(boundary)/(param.GRID_SIZE*param.GRID_SIZE*rect_area(leaf_node.n_box)))
    #         weights[item] = leaf_node.urgency*coverage_ratio
    #         last_weight = weights[item]
    #     else:
    #         weights[item] = last_weight
    #         # count = count + 1

    # visual awareness is computed based on cell overlap
    unit_cell_area = rect_area([[param.x_min, param.y_min], [param.x_max, param.y_max]])/param.GRID_SIZE**2
    shakemap_cell_area = rect_area([[min_lat, min_lon],[max_lat, max_lon]]) / (SHAKEMAP_LAT_SIZE * SHAKEMAP_LON_SIZE)
    cell_ratio = unit_cell_area/shakemap_cell_area

    # update values in universe & updated_unit_cells
    for item in universe:
        if not updated_unit_cells.has_key(item):    # add the unit cell
            # compute visual awareness
            coord = cell_coord(item, param)
            urgency = urgency_value(coord[0], coord[1])
            updated_unit_cells[item] = weights[item] = urgency * cell_ratio / 2
        else:
            # update the value of the element
            weights[item] = updated_unit_cells[item]

    # print len(universe), universe
    # print len(all_sets), all_sets
    # print fov_count, weights

    # start = time.time()
    covered_sets, covered_items, covered_weight = max_cover(universe, all_sets, fov_count, weights)
    # print "time ", time.time() - start
    # print covered_weight

    # reduce VA value of the cells in covered_items by half
    for item in covered_items:
        updated_unit_cells[item] = updated_unit_cells[item] / 2

    return covered_weight

def eval_optimization(param):

    seed = 1000
    fov_count = [5,10,15,20,25,30]
    global updated_unit_cells
    for count in fov_count:
        updated_unit_cells = {}
        total_va = 0
        for time in range(param.TIME_SNAPSHOT):
            total_va = total_va + optimization(param,count,seed,time)
        print total_va


# varying the number of analysts, measure the total visual awareness
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
    param.select_dataset()
    # data = data_readin(param)
    # param.NDIM, param.NDATA = data.shape[0], data.shape[1]
    # param.LOW, param.HIGH = np.amin(data, axis=1), np.amax(data, axis=1)

    eval_optimization(param)
    # eval_partition(data, param)

    # eval_analyst(data, param)
    # eval_bandwidth(data, param)
    # eval_skewness(data, param)


    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  END")