__author__ = 'ubriela'

from sets import Set
import copy

"""
weighted max cover problem
@universe: all unit cells' ids
@all_sets: a set of sets
@budget: number of fov can be selected
@weights: VA values of the unit cells
"""
def max_cover(universe, _all_sets, budget, weights):
    # having a copy of all_sets
    all_sets = copy.deepcopy(_all_sets)
    covered_sets = Set([])
    covered_items = Set([])

    # Run until either running out of budget or no more tasks to cover
    while len(covered_sets) < budget and len(universe) > 0 and len(all_sets) > 0:
        best_setid = -1   # track index of the best worker in universe
        max_uncovered_weight = 0.0

        # Find the set which covers maximum weight of uncovered elements
        for idx in all_sets.keys():
            curr_set = all_sets.get(idx)
            uncovered_items = curr_set - covered_items
            # print len(uncovered_items)
            uncovered_weight = sum([weights[item] for item in uncovered_items])

            if uncovered_weight >= max_uncovered_weight:
                 max_uncovered_weight, best_setid = uncovered_weight, idx

        if max_uncovered_weight == 0:
            break

        # print max_uncovered_weight

        if best_setid > -1:
            covered_sets.add(best_setid)
            # universe.difference_update(all_sets.get(best_setid))
            covered_items.update(all_sets.get(best_setid))
            # print best_setid, all_sets.get(best_setid)
            # print universe
            # print covered_items
            del all_sets[best_setid]

        # if best_setid == 84:
        #     print "xxx"

    covered_weight = sum([weights[item] for item in covered_items])

    return covered_sets, covered_items, covered_weight


if False:
    # test
    universe = Set([1,2,3,4,5])
    weights = {1:1,2:1,3:2,4:3,5:1}
    all_sets = {}
    all_sets[0] = Set([1,2,3])
    all_sets[1] = Set([2,4])
    all_sets[2] = Set([3,4])
    all_sets[3] = Set([4,5])
    budget = 2

    print max_cover(universe, all_sets, budget, weights)





