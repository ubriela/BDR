import time
import os

import numpy as np
import math
import random
import time

# import multiprocessing as mult
import sys
import re
import logging
from collections import deque
import random
from Exp import Exp
from Params import Params
from DataParser import read_data
from Grid_standard import Grid_standard
from Kd_standard import Kd_standard
from Quad_standard import Quad_standard
from knapsack import zeroOneKnapsack

sys.path.append('../plot')
sys.path.append('../plot/code')

from Utils import rect_area,zipf_pmf
# from draw_workcell import getLeafNode

seed_list = [2172]
# seed_list = [9110, 4064, 6903, 7509, 5342, 3230, 3584, 7019, 3564, 6456]

method_list = None
exp_name = None
dataset_identifier = "_gau"

def sample_data(data, p):
    # print data.shape
    samples = np.random.choice([True, False, False, False], (data.shape[1], 1)).flat[:]
    # print samples
    sampled_data = data.transpose()[samples].transpose()
    # print sampled_data.shape
    return sampled_data

def data_readin(p):
    """Read in spatial data and initialize global variables."""
    p.select_dataset()
    data = np.genfromtxt(p.dataset, unpack=True)
    data = sample_data(data, p)
    p.NDIM, p.NDATA = data.shape[0], data.shape[1]
    p.LOW, p.HIGH = np.amin(data, axis=1), np.amax(data, axis=1)
    logging.debug(data.shape)
    logging.debug(p.LOW)
    logging.debug(p.HIGH)
    return data[0:2]


def gen_query(queryShape, seed, param):
    logging.debug('Generating queries...')
    x1, y1, x2, y2 = param.x_min, param.y_min, param.x_max, param.y_max
    np.random.seed(seed)
    querylist = []
    x_range, y_range = queryShape[0] * (x2-x1), queryShape[1] * (y2-y1)

    point_x = np.random.uniform(x1, x2, param.nQuery)
    point_y = np.random.uniform(y1, y2, param.nQuery)
    x_low = point_x
    x_high = point_x + x_range
    y_low = point_y - y_range
    y_high = point_y
    for i in range(param.nQuery):
        querylist.append(np.array([[x_low[i], y_low[i]], [x_high[i], y_high[i]]]))

    return querylist

def test_query(data, queryShape, param):
    global method_list, exp_name
    exp_name = 'test_query'
    method_list = ['Quad_standard', 'Kd_standard']

    # Params.maxHeight = 10
    epsList = [0.1]
    res_cube_abs = np.zeros((len(epsList), len(seed_list), len(method_list)))
    res_cube_rel = np.zeros((len(epsList), len(seed_list), len(method_list)))

    for j in range(len(seed_list)):
        queryList = gen_query(queryShape, seed_list[j], param)
        kexp = Exp(data, queryList)
        for i in range(len(epsList)):
            for k in range(len(method_list)):
                param.Seed = seed_list[j]
                if method_list[k] == 'Quad_standard':
                    res_cube_abs[i, j, k], res_cube_rel[i, j, k] = kexp.run_Quad_standard(param)
                elif method_list[k] == 'Kd_standard':
                    res_cube_abs[i, j, k], res_cube_rel[i, j, k] = kexp.run_Kd_standard(param)
                else:
                    logging.error('No such index structure!')
                    sys.exit(1)

    res_abs_summary = np.average(res_cube_abs, axis=1)
    res_rel_summary = np.average(res_cube_rel, axis=1)
    np.savetxt(param.resdir + exp_name + '_abs_' + str(int(queryShape[0] * 10)) + '_' + str(int(queryShape[1] * 10)),
               res_abs_summary, fmt='%.4f')
    np.savetxt(param.resdir + exp_name + '_rel_' + str(int(queryShape[0] * 10)) + '_' + str(int(queryShape[1] * 10)),
               res_rel_summary, fmt='%.4f')

def getLeafNode(tree, type):
    leaf_boxes = []
    if type == 1:
        for l1_child in tree.root.children:
            if not l1_child.n_isLeaf and l1_child.children is not None:
                for l2_child in l1_child.children:  # child1 is a first-level cell
                    leaf_boxes.append((l2_child.n_box, l2_child.n_count))
            leaf_boxes.append((l1_child.n_box, l1_child.n_count))
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

populations_data = None
def compute_urgency(node):
    if Params.URGENCY_RANDOM == True:
        x = random.randint(0,10)
        # node.urgency = 3
        return
    global populations_data
    if populations_data == None:
        populations_data = np.genfromtxt(Params.POPULATION_FILE, unpack=True)
    populations = populations_data
    ndim = populations.shape[0]
    for dim in range(ndim):
        if populations.shape[1] == 0:
            break
        idx = np.argsort(populations[dim, :], kind='mergesort')
        populations[:, :] = populations[:, idx]
        query = node.n_box
        x = np.searchsorted(populations[dim, :], query[0, dim], side='left')
        y = np.searchsorted(populations[dim, :], query[1, dim], side='right')
        populations = populations[:, x:y + 1]
    node.urgency = populations.shape[1]

def optimization(tree, bandwidth, seed, param):
    np.random.seed(seed)
    fovs_file = re.sub(r'\.dat$', '', param.dataset) + ".txt"
    videos = read_data(fovs_file)

    # update video value
    for v in videos:
        leaf_node = tree.leafCover(v.location())
        if leaf_node:
            v.size = np.random.zipf(param.ZIPFIAN_SKEW)
            if v.size > 20:
                v.size = 20
            v.fov_count = int(v.size)
            # size = int(np.random.uniform(1,10))

            # compute urgency of leaf node
            compute_urgency(leaf_node)
            # ratio = 1000000 * v.area() / rect_area(leaf_node.n_box) / 6.25
            # print v.area()
            v.value = leaf_node.urgency * v.area()
            # print v.value
        # else:
        #     print "not a leaf node", v.location()

    weights = [v.size for v in videos]
    values  = [v.value for v in videos]
    # print weights
    # print values
    total_value = zeroOneKnapsack(values,weights,bandwidth)

    print "\n if my knapsack can hold %d bandwidth, i can get %f profit." % (bandwidth,total_value[0])
    print "\tby taking item(s): ",
    for i in range(len(total_value[1])):
        if (total_value[1][i] != 0):
            print i+1,

    return total_value

def eval_partition(data, param):
    # tree = Grid_standard(data, param)
    # tree = Quad_standard(data, param)
    tree = Kd_standard(data, param)
    tree.buildIndex()

    seed = 1000
    bandwidth = 1000
    answer = optimization(tree, bandwidth, seed, param)

    print "\n if my knapsack can hold %d bandwidth, i can get %f profit." % (bandwidth,answer[0])
    print "\tby taking item(s): ",
    for i in range(len(answer[1])):
        if (answer[1][i] != 0):
            print i+1,

def eval_workload(data, param):
    logging.info("eval_workload")
    exp_name = "eval_workload"

    analyst = [4,5,6,7,8]
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
                    tree.buildIndex()
                    leaf_boxes = getLeafNode(tree, 1)
                elif method_list[k] == 'quad_standard':
                    tree = Quad_standard(data, param)
                    tree.buildIndex()
                    leaf_boxes = getLeafNode(tree, 2)
                elif method_list[k] == 'kd_standard':
                    tree = Kd_standard(data, param)
                    tree.buildIndex()
                    leaf_boxes = getLeafNode(tree, 2)
                else:
                    logging.error('No such index structure!')
                    sys.exit(1)
                workload_counts = [leaf[1] for leaf in leaf_boxes]
                res_cube_value[i, j, k] = np.var(workload_counts)

    res_value_summary = np.average(res_cube_value, axis=1)
    np.savetxt(param.resdir + exp_name + dataset_identifier , res_value_summary, fmt='%.4f\t')


def eval_analyst(data, param):
    logging.info("eval_analyst")
    exp_name = "eval_analyst"

    analyst = [4,5,6,7,8]
    bandwidth = 20    # fixed
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

                answer = optimization(tree, bandwidth, seed_list[j], param)
                res_cube_value[i, j, k] = answer[0]

    res_value_summary = np.average(res_cube_value, axis=1)
    np.savetxt(param.resdir + exp_name + dataset_identifier , res_value_summary, fmt='%.4f\t')

def eval_bandwidth(data, param):
    logging.info("eval_bandwidth")
    exp_name = "eval_bandwidth"

    analyst = 6
    param.part_size = analyst
    param.ANALYST_COUNT = analyst * analyst

    bandwidth = [10,15,20,25,30]
    method_list = ['grid_standard', 'quad_standard', 'kd_standard']

    res_cube_value = np.zeros((len(bandwidth), len(seed_list), len(method_list)))

    for j in range(len(seed_list)):
        param.seed = seed_list[j]
        for i in range(len(bandwidth)):
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

                answer = optimization(tree, bandwidth[i], seed_list[j], param)
                res_cube_value[i, j, k] = answer[0]

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
    bandwidth = 20    # fixed

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

                answer = optimization(tree, bandwidth, seed_list[j], param)
                res_cube_value[i, j, k] = answer[0]

    res_value_summary = np.average(res_cube_value, axis=1)
    np.savetxt(param.resdir + exp_name + dataset_identifier , res_value_summary, fmt='%.4f\t')

def eval_videos(data, param):

    print ""

def eval_runtime(data, param):
    logging.info("eval_runtime")
    exp_name = "eval_runtime"

    analyst = 6
    param.part_size = analyst
    param.ANALYST_COUNT = analyst * analyst
    bandwidth = 20    # fixed

    method_list = ['grid_standard', 'quad_standard', 'kd_standard']
    res_cube_value = np.zeros((2, len(seed_list), len(method_list)))

    for j in range(len(seed_list)):
        param.seed = seed_list[j]
        for k in range(len(method_list)):
            start = time.time()
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
            partition_duration = time.time() - start

            answer = optimization(tree, bandwidth, seed_list[j], param)
            optimization_duration = time.time() - start - partition_duration

            res_cube_value[0, j, k] = partition_duration
            res_cube_value[1, j, k] = optimization_duration

    res_value_summary = np.average(res_cube_value, axis=1)
    np.savetxt(param.resdir + exp_name + dataset_identifier, res_value_summary, fmt='%.4f\t')



if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG, filename='../../log/debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")

    param = Params(1000)
    data = data_readin(param)
    param.NDIM, param.NDATA = data.shape[0], data.shape[1]
    param.LOW, param.HIGH = np.amin(data, axis=1), np.amax(data, axis=1)

    # eval_workload(data, param)
    eval_analyst(data, param)
    # eval_bandwidth(data, param)
    # eval_skewness(data, param)
    # eval_runtime(data, param)
    # eval_partition(data, param)

    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  END")