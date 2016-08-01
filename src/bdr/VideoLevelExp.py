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
import heapq
from Node import Node

sys.path.append('../plot')
sys.path.append('../plot/code')

from Utils import rect_area,zipf_pmf
# from draw_workcell import getLeafNode
from Utils import is_rect_cover

# seed_list = [2172]
seed_list = [9110, 4064, 6903, 7509, 5342, 3230, 3584, 7019, 3564, 6456]

analyst = [6, 7, 8, 9, 10]
analyst_count = 8

# each analyst can handle an amount of work
capacity = [2, 3, 4, 5, 6]
analyst_capacity = 4

method_list = None
exp_name = None

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


def getLeafNode2(tree, type):
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
    return  node.urgency

def gen_videos(param):
    np.random.seed(param.seed)
    fovs_file = re.sub(r'\.dat$', '', param.dataset) + ".txt"
    videos = read_data(fovs_file)

    # update video value
    for v in videos:
        v.size = np.random.zipf(param.ZIPFIAN_SKEW)
        if v.size > 10:
            v.size = 10
        v.fov_count = int(v.size)
        # size = int(np.random.uniform(1,10))

        # print v.area()
        if Params.URGENCY_RANDOM == True:
            v.value = int(np.random.uniform(1,10)) * v.area()
        else:
            node = Node()
            lat, lon = v.fovs[0].lat, v.fovs[0].lon
            node.n_box = np.array([[lat - Params.ONE_KM/5, lon - Params.ONE_KM/5], [lat + Params.ONE_KM/5, lon + Params.ONE_KM/5]])
            v.value = compute_urgency(node) * v.area()
        # print v.value

    return videos

def heapsort(iterable):
    h = []
    for value in iterable:
        heapq.heappush(h, value)
    return [heapq.heappop(h) for i in range(len(h))]

def video_eval_analyst(param):
    logging.info("video_eval_analyst")
    exp_name = "video_eval_analyst"

    method_list = ['grid_standard', 'quad_standard', 'kd_standard']

    res_cube_value = np.zeros((len(analyst), len(seed_list), len(method_list)))

    for j in range(len(seed_list)):
        param.seed = seed_list[j]
        videos = gen_videos(param)  # generate videos given a seed

        for k in range(len(method_list)):
            # all videos
            locs = np.zeros((2, len(videos)))
            for l in range(len(videos)):
                locs[0][l], locs[1][l] = videos[l].fovs[0].lat, videos[l].fovs[0].lon

            for i in range(len(analyst)):
                param.part_size = analyst[i]
                param.ANALYST_COUNT = analyst[i] * analyst[i]
                bandwidth = param.ANALYST_COUNT * analyst_capacity

                weights = [v.size for v in videos]
                values = [v.value for v in videos]

                if method_list[k] == 'grid_standard': # if grid --> partition first, select later
                    tree = Grid_standard(locs, param)
                else:
                    # upload best videos
                    result = zeroOneKnapsack(values, weights, bandwidth)
                    # optimal_va = result[0]

                    # locations of the uploaded videos
                    uploaded_videos = [videos[l] for l in range(len(result[1])) if result[1][l] != 0]

                    uploaded_locs = np.zeros((2, len(uploaded_videos)))
                    for l in range(len(uploaded_videos)):
                        uploaded_locs[0][l], uploaded_locs[1][l] = uploaded_videos[l].fovs[0].lat, \
                                                                   uploaded_videos[l].fovs[0].lon
                    if method_list[k] == 'quad_standard':
                        tree = Quad_standard(uploaded_locs, param)
                    elif method_list[k] == 'kd_standard':
                        tree = Kd_standard(uploaded_locs, param)

                tree.buildIndex()
                all_values = []

                # get leaf nodes (work cells)
                if method_list[k] == 'quad_standard' or method_list[k] == 'kd_standard':
                    leaf_nodes = getLeafNode(tree, 2)
                else:
                    leaf_nodes = getLeafNode(tree, 1)

                # each analyst chooses the best videos in their assigned work cells
                # print method_list[k], len(leaf_nodes)
                for (n_box, count) in leaf_nodes:
                    if count > 0:
                        leaf_values, leaf_weights = [], []
                        for l in range(len(videos)):
                            loc = [videos[l].fovs[0].lat, videos[l].fovs[0].lon]
                            if is_rect_cover(n_box, loc):
                                leaf_values.append(values[l])
                                leaf_weights.append(weights[l])

                        if len(leaf_values) > 0:
                            # print leaf_values
                            # print leaf_weights
                            # print threshold
                            val = zeroOneKnapsack(leaf_values, leaf_weights, analyst_capacity)
                            all_values.append(val[0])


                heap = heapsort(all_values)
                for h in range(len(all_values) - param.ANALYST_COUNT):
                    heapq.heappop(heap)

                res_cube_value[i, j, k] = sum(heap)

    res_value_summary = np.average(res_cube_value, axis=1)
    np.savetxt(param.resdir + exp_name + "_"+ param.DATASET , res_value_summary, fmt='%.4f\t')


def video_eval_capacity(param):
    logging.info("video_eval_capacity")
    exp_name = "video_eval_capacity"

    method_list = ['grid_standard', 'quad_standard', 'kd_standard']

    res_cube_value = np.zeros((len(capacity), len(seed_list), len(method_list)))

    for j in range(len(seed_list)):
        param.seed = seed_list[j]
        videos = gen_videos(param)  # generate videos given a seed

        for k in range(len(method_list)):
            # all videos
            locs = np.zeros((2, len(videos)))
            for l in range(len(videos)):
                locs[0][l], locs[1][l] = videos[l].fovs[0].lat, videos[l].fovs[0].lon

            for i in range(len(capacity)):
                param.part_size = analyst_count
                param.ANALYST_COUNT = analyst_count * analyst_count
                bandwidth = capacity[i] * param.ANALYST_COUNT

                # upload best videos
                weights = [v.size for v in videos]
                values = [v.value for v in videos]


                if method_list[k] == 'grid_standard': # if grid --> partition first, select later
                    tree = Grid_standard(locs, param)
                else:
                    # upload best videos
                    result = zeroOneKnapsack(values, weights, bandwidth)
                    # optimal_va = result[0]

                    # locations of the uploaded videos
                    uploaded_videos = [videos[l] for l in range(len(result[1])) if result[1][l] != 0]

                    uploaded_locs = np.zeros((2, len(uploaded_videos)))
                    for l in range(len(uploaded_videos)):
                        uploaded_locs[0][l], uploaded_locs[1][l] = uploaded_videos[l].fovs[0].lat, \
                                                                   uploaded_videos[l].fovs[0].lon
                    if method_list[k] == 'quad_standard':
                        tree = Quad_standard(uploaded_locs, param)
                    elif method_list[k] == 'kd_standard':
                        tree = Kd_standard(uploaded_locs, param)

                tree.buildIndex()
                all_values = []

                # get leaf nodes (work cells)
                if method_list[k] == 'quad_standard' or method_list[k] == 'kd_standard':
                    leaf_nodes = getLeafNode(tree, 2)
                else:
                    leaf_nodes = getLeafNode(tree, 1)

                # each analyst chooses the best videos in their assigned work cells
                # print method_list[k], len(leaf_nodes)
                for (n_box, count) in leaf_nodes:
                    if count > 0:
                        leaf_values, leaf_weights = [], []
                        for l in range(len(videos)):
                            loc = [videos[l].fovs[0].lat, videos[l].fovs[0].lon]
                            if is_rect_cover(n_box, loc):
                                leaf_values.append(values[l])
                                leaf_weights.append(weights[l])

                        if len(leaf_values) > 0:
                            val = zeroOneKnapsack(leaf_values, leaf_weights, capacity[i])
                            all_values.append(val[0])

                heap = heapsort(all_values)
                for h in range(len(all_values) - param.ANALYST_COUNT):
                    heapq.heappop(heap)

                res_cube_value[i, j, k] = sum(heap)

    res_value_summary = np.average(res_cube_value, axis=1)
    np.savetxt(param.resdir + exp_name + "_"+ param.DATASET , res_value_summary, fmt='%.4f\t')


def video_eval_skewness(param):
    logging.info("video_eval_skewness")
    exp_name = "video_eval_skewness"

    param.part_size = analyst_count
    param.ANALYST_COUNT = analyst_count * analyst_count
    bandwidth = analyst_capacity * param.ANALYST_COUNT

    skewness = [1.6,1.8,2.0,2.2,2.4]

    method_list = ['grid_standard', 'quad_standard', 'kd_standard']

    res_cube_value = np.zeros((len(skewness), len(seed_list), len(method_list)))

    for j in range(len(seed_list)):
        param.seed = seed_list[j]
        for k in range(len(method_list)):
            for i in range(len(skewness)):
                param.ZIPFIAN_SKEW = skewness[i]
                videos = gen_videos(param)  # generate videos given a seed

                # all videos
                locs = np.zeros((2, len(videos)))
                for l in range(len(videos)):
                    locs[0][l], locs[1][l] = videos[l].fovs[0].lat, videos[l].fovs[0].lon

                # upload best videos
                weights = [v.size for v in videos]
                values = [v.value for v in videos]


                if method_list[k] == 'grid_standard': # if grid --> partition first, select later
                    tree = Grid_standard(locs, param)
                else:
                    # upload best videos
                    result = zeroOneKnapsack(values, weights, bandwidth)
                    # optimal_va = result[0]

                    # locations of the uploaded videos
                    uploaded_videos = [videos[l] for l in range(len(result[1])) if result[1][l] != 0]

                    uploaded_locs = np.zeros((2, len(uploaded_videos)))
                    for l in range(len(uploaded_videos)):
                        uploaded_locs[0][l], uploaded_locs[1][l] = uploaded_videos[l].fovs[0].lat, \
                                                                   uploaded_videos[l].fovs[0].lon
                    if method_list[k] == 'quad_standard':
                        tree = Quad_standard(uploaded_locs, param)
                    elif method_list[k] == 'kd_standard':
                        tree = Kd_standard(uploaded_locs, param)

                tree.buildIndex()
                all_values = []

                # get leaf nodes (work cells)
                if method_list[k] == 'quad_standard' or method_list[k] == 'kd_standard':
                    leaf_nodes = getLeafNode(tree, 2)
                else:
                    leaf_nodes = getLeafNode(tree, 1)

                # each analyst chooses the best videos in their assigned work cells
                # print method_list[k], len(leaf_nodes)
                for (n_box, count) in leaf_nodes:
                    if count > 0:
                        leaf_values, leaf_weights = [], []
                        for l in range(len(videos)):
                            loc = [videos[l].fovs[0].lat, videos[l].fovs[0].lon]
                            if is_rect_cover(n_box, loc):
                                leaf_values.append(values[l])
                                leaf_weights.append(weights[l])

                        if len(leaf_values) > 0:
                            val = zeroOneKnapsack(leaf_values, leaf_weights, analyst_capacity)
                            all_values.append(val[0])

                heap = heapsort(all_values)
                for h in range(len(all_values) - param.ANALYST_COUNT):
                    heapq.heappop(heap)

                res_cube_value[i, j, k] = sum(heap)

    res_value_summary = np.average(res_cube_value, axis=1)
    np.savetxt(param.resdir + exp_name + "_"+ param.DATASET , res_value_summary, fmt='%.4f\t')


def video_eval_runtime(param):
    logging.info("video_eval_runtime")
    exp_name = "video_eval_runtime"

    param.part_size = analyst_count
    param.ANALYST_COUNT = analyst_count * analyst_count
    bandwidth = analyst_capacity * param.ANALYST_COUNT

    method_list = ['grid_standard', 'quad_standard', 'kd_standard']
    res_cube_value = np.zeros((2, len(seed_list), len(method_list)))

    for j in range(len(seed_list)):
        param.seed = seed_list[j]
        for k in range(len(method_list)):

            videos = gen_videos(param)  # generate videos given a seed

            start = time.time()

            # all videos
            locs = np.zeros((2, len(videos)))
            for l in range(len(videos)):
                locs[0][l], locs[1][l] = videos[l].fovs[0].lat, videos[l].fovs[0].lon

            # h = {}
            # for videos in videos:
            #     h[str(videos[l].fovs[0].lat) + ";" + str(videos[l].fovs[0].lon)] = videos

            # upload best videos
            weights = [v.size for v in videos]
            values = [v.value for v in videos]

            if method_list[k] == 'grid_standard':
                tree = Grid_standard(locs, param)
            else:
                # upload best videos
                result = zeroOneKnapsack(values, weights, bandwidth)
                # optimal_va = result[0]

                # locations of the uploaded videos
                uploaded_videos = [videos[l] for l in range(len(result[1])) if result[1][l] != 0]

                uploaded_locs = np.zeros((2, len(uploaded_videos)))
                for l in range(len(uploaded_videos)):
                    uploaded_locs[0][l], uploaded_locs[1][l] = uploaded_videos[l].fovs[0].lat, \
                                                               uploaded_videos[l].fovs[0].lon
                if method_list[k] == 'quad_standard':
                    tree = Quad_standard(uploaded_locs, param)
                elif method_list[k] == 'kd_standard':
                    tree = Kd_standard(uploaded_locs, param)

            tree.buildIndex()
            partition_duration = time.time() - start

            all_values = []

            # get leaf nodes (work cells)
            if method_list[k] == 'quad_standard' or method_list[k] == 'kd_standard':
                leaf_nodes = getLeafNode(tree, 2)
            else:
                leaf_nodes = getLeafNode(tree, 1)

            # each analyst chooses the best videos in their assigned work cells
            # print method_list[k], len(leaf_nodes)
            for (n_box, count) in leaf_nodes:
                if count > 0:
                    leaf_values, leaf_weights = [], []

                    # for l in range(node.n_data.shape[1]):
                    #     leaf_fovs.append(h[str(node.n_data[0][l]) + ";" + str(node.n_data[1][l])])

                    for l in range(len(videos)):
                        loc = [videos[l].fovs[0].lat, videos[l].fovs[0].lon]
                        if is_rect_cover(n_box, loc):
                            leaf_values.append(values[l])
                            leaf_weights.append(weights[l])

                    if len(leaf_values) > 0:
                        val = zeroOneKnapsack(leaf_values, leaf_weights, analyst_capacity)
                        all_values.append(val[0])

            heap = heapsort(all_values)
            for h in range(len(all_values) - param.ANALYST_COUNT):
                heapq.heappop(heap)

            optimization_duration = time.time() - start - partition_duration

            res_cube_value[0, j, k] = partition_duration
            res_cube_value[1, j, k] = optimization_duration

    res_value_summary = np.average(res_cube_value, axis=1)
    np.savetxt(param.resdir + exp_name + "_"+ param.DATASET , res_value_summary, fmt='%.4f\t')



if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG, filename='../../log/debug.log')
    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  START")

    param = Params(1000)
    data = data_readin(param)
    param.NDIM, param.NDATA = data.shape[0], data.shape[1]
    param.LOW, param.HIGH = np.amin(data, axis=1), np.amax(data, axis=1)

    video_eval_analyst(param)
    # video_eval_capacity(param)
    # video_eval_skewness(param)
    # video_eval_runtime(param)
    # eval_partition(data, param)

    logging.info(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()) + "  END")