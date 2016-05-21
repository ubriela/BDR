import logging
from collections import deque

import numpy as np

from Node import Node
from Params import Params

import sys
sys.path.append('../../../privategeocrowddynamic/src/common')
sys.path.append('../plot/code')

from Utils import is_rect_cover, rect_area


class Tree(object):
    """Generic tree template"""

    def __init__(self, data, param):
        self.param = param
        self.cell_count = 0
        # ## initialize the root
        self.root = Node()
        self.root.n_data = data
        self.root.n_box = np.array([param.LOW, param.HIGH])

    def getCoordinates(self, curr):
        """
        return the coordinate of lower-right point of the NW sub-node
        and the upper-left point of the SW sub-node and the data points
        in the four subnodes, i.e.
        return (x_nw,y_nw),(x_se,y_se), nw_data, ne_data, sw_data, se_data
        """
        raise NotImplementedError

    def getSplit(self, array, left, right, epsilon):
        """
        return the split point given an array, may be data-independent or
        true median or noisy median, depending on the type of the tree
        """
        raise NotImplementedError

    def getCount(self, curr):
        """ return true count """
        if curr.n_data is None:
            count = 0
        else:
            count = curr.n_data.shape[1]
        return count

    # this function is used in BDR paper
    def testLeaf_bdr(self, curr):
        """ test whether a node should be a leaf node """
        if (curr.n_data is None or curr.n_data.shape[1] == 0) or \
            self.cell_count >= self.param.ANALYST_COUNT or \
                (curr.n_count <= self.param.minPartSize) or \
                    rect_area(curr.n_box) < 0.01:
            return True
        return False

    def testLeaf(self, curr):
        """ test whether a node should be a leaf node """
        if (curr.n_data is None or curr.n_data.shape[1] == 0) or \
            curr.area() < 0.0004 or \
                (curr.n_count <= self.param.minPartSize) or \
                    rect_area(curr.n_box) < 0.0004:
            return True
        return False

    def cell_setLeaf(self, curr):
        """ will be overrided in kd_cell """
        return

    def buildIndex(self):
        """ Function to build the tree structure, fanout = 4 by default for spatial (2D) data """
        self.root.n_count = self.getCount(self.root)
        self.cell_count = 1
        try:
            import Queue as Q  # ver. < 3.0
        except ImportError:
            import queue as Q
        pqueue= Q.PriorityQueue()
        pqueue.put((-self.root.n_data.shape[1], self.root))
        max_depth = -1
        # ## main loop
        while not pqueue.empty():
            curr = pqueue.get()[1]
            # print curr.n_count
            if curr.n_depth > max_depth:
                max_depth = curr.n_depth

            if self.testLeaf(curr) is True:  # ## curr is a leaf node
                curr.n_count = self.getCount(curr)
                curr.n_isLeaf = True
                self.cell_setLeaf(curr)

            else:  # ## curr needs to split
                tmp = self.getCoordinates(curr)
                nw_node, ne_node, sw_node, se_node = Node(), Node(), Node(), Node()  # create sub-nodes
                nw_coord, ne_coord, nw_node.n_data, ne_node.n_data, sw_node.n_data, se_node.n_data = tmp
                x_nw, y_nw = nw_coord
                x_se, y_se = ne_coord
                # ## update bounding box, depth, count for the four subnodes
                nw_node.n_box = np.array([[curr.n_box[0, 0], y_nw], [x_nw, curr.n_box[1, 1]]])
                ne_node.n_box = np.array([[x_nw, y_se], [curr.n_box[1, 0], curr.n_box[1, 1]]])
                sw_node.n_box = np.array([[curr.n_box[0, 0], curr.n_box[0, 1]], [x_se, y_nw]])
                se_node.n_box = np.array([[x_se, curr.n_box[0, 1]], [curr.n_box[1, 0], y_se]])

                for sub_node in [nw_node, ne_node, sw_node, se_node]:
                    sub_node.n_depth = curr.n_depth + 1
                    # if (sub_node.n_depth == Params.maxHeight and sub_node.n_data is not None):
                    # print len(sub_node.n_data[0])
                    sub_node.n_count = self.getCount(sub_node)
                    pqueue.put((-sub_node.n_count, sub_node))

                curr.n_data = None  # ## do not need the data points coordinates now
                curr.nw, curr.ne, curr.sw, curr.se = nw_node, ne_node, sw_node, se_node
                self.cell_count += 3
        # end of while

        logging.debug("number of leaves: %d" % self.cell_count)
        logging.debug("max depth: %d" % max_depth)

    def leafCover(self, loc):
        """
        find a leaf node that cover the location
        """
        queue = deque()
        queue.append(self.root)
        while len(queue) > 0:
            curr = queue.popleft()
            _box = curr.n_box
            if curr.n_isLeaf is True:
                if is_rect_cover(_box, loc):
                    return curr
            else:  # if not leaf
                queue.append(curr.nw)
                queue.append(curr.ne)
                queue.append(curr.sw)
                queue.append(curr.se)

    def rect_intersect(self, hrect, query):
        """
        checks if the hyper-rectangle intersects with the
        hyper-rectangle defined by the query in every dimension
    
        """
        bool_m1 = query[0, :] >= hrect[1, :]
        bool_m2 = query[1, :] <= hrect[0, :]
        bool_m = np.logical_or(bool_m1, bool_m2)
        if np.any(bool_m):
            return False
        else:
            return True

    def rangeCount(self, query):
        """
        Query answering function. Find the number of data points within a query rectangle.
        """
        stack = deque()
        stack.append(self.root)
        count = 0.0
        # ## Below are three variables recording the number of 1) whole leaf 2) partial leaf 3) whole internal node,
        # ## respectively, which contribute to the query answer. For debug purpose only.
        l_whole, l_part, i_whole = 0, 0, 0

        while len(stack) > 0:
            curr = stack.popleft()
            _box = curr.n_box
            if curr.n_isLeaf is True:
                frac = 1
                if self.rect_intersect(_box, query):
                    for i in range(_box.shape[1]):
                        if _box[1, i] == _box[0, i]:
                            frac *= 1
                        else:
                            frac *= (min(query[1, i], _box[1, i]) - max(query[0, i], _box[0, i])) / (
                                _box[1, i] - _box[0, i])
                    count += curr.n_count * frac
                    if 1.0 - frac < 10 ** (-6):
                        l_whole += 1
                    else:
                        l_part += 1

            else:  # ## if not leaf
                bool_matrix = np.zeros((2, query.shape[1]))
                bool_matrix[0, :] = query[0, :] <= _box[0, :]
                bool_matrix[1, :] = query[1, :] >= _box[1, :]

                if np.all(bool_matrix):  # ## if query range contains node range
                    count += curr.n_count
                    i_whole += 1
                else:
                    if self.rect_intersect(curr.nw.n_box, query):
                        stack.append(curr.nw)
                    if self.rect_intersect(curr.ne.n_box, query):
                        stack.append(curr.ne)
                    if self.rect_intersect(curr.sw.n_box, query):
                        stack.append(curr.sw)
                    if self.rect_intersect(curr.se.n_box, query):
                        stack.append(curr.se)

        return float(count)  # , i_whole, l_whole, l_part

    def checkCorrectness(self, node, nodePoints=None):
        """
        Total number of data points of all leaf nodes should equal to the total data points
        """
        totalPoints = 0
        if node is None:
            return 0
        if node.n_isLeaf and node.n_data is not None:
            return node.n_data.shape[1]
        for child in [node.nw, node.ne, node.sw, node.se]:
            totalPoints += self.checkCorrectness(child)

        if nodePoints is None:
            return totalPoints

        if totalPoints == nodePoints:
            return True
        return False

    def pruning(self):
        """
        If the tree is grown without the stopping condition of minLeafSize, prune it here after post processing
        """
        logging.debug("pruning...")
        queue = deque()
        queue.append(self.root)
        while len(queue) > 0:
            curr = queue.popleft()
            if curr.n_isLeaf is False:
                if curr.n_count <= self.param.minPartSize:
                    curr.n_isLeaf = True
                else:
                    queue.append(curr.nw)
                    queue.append(curr.ne)
                    queue.append(curr.sw)
                    queue.append(curr.se)
