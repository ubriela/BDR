#!/usr/bin/python
import sys

def zeros(rows,cols):
	row = []
	data = []
	for i in range(cols):
		row.append(0)
	for i in range(rows):
		data.append(row[:])
	return data
a = zeros(10,10)


def getItemsUsed(w,c):
    # item count
	i = len(c)-1
	# weight
	currentW =  len(c[0])-1
	
	# set everything to not marked
	marked = []
	for i in range(i+1):
		marked.append(0)	
		
	while (i >= 0 and currentW >=0):
		# if this weight is different than
		# the same weight for the last item
		# then we used this item to get this profit
		#
		# if the number is the same we could not add
		# this item because it was too heavy		
		if (i==0 and c[i][currentW] >0 )or c[i][currentW] != c[i-1][currentW]:
			marked[i] =1
			currentW = currentW-w[i]
		i = i-1
	return marked
	
	


# v = list of item values or profit
# w = list of item weight or cost
# W = max weight or max cost for the knapsack
def zeroOneKnapsack(v, w, W):
	# c is the cost matrix
	c = []
	n = len(v)
	#  set inital values to zero
	c = zeros(n,W+1)
	#the rows of the matrix are weights
	#and the columns are items
	#cell c[i,j] is the optimal profit
	#for i items of cost j
	
	#for every item
	for i in range(0,n):
		#for ever possible weight
		for j in range(0,W+1):
			#if this weight can be added to this cell
			#then add it if it is better than what we aready have
			
			if (w[i] > j):
				
				# this item is to large or heavy to add
				# so we just keep what we aready have
				
				c[i][j] = c[i-1][j]
			else:
				# we can add this item if it gives us more value
				# than skipping it
				
				# c[i-1][j-w[i]] is the max profit for the remaining 
				# weight after we add this item.
				
				# if we add the profit of this item to the max profit
				# of the remaining weight and it is more than 
				# adding nothing , then it't the new max profit
				# if not just add nothing.
				
				c[i][j] = max(c[i-1][j],v[i] +c[i-1][j-w[i]])
	# print c
	return [c[n-1][W],getItemsUsed(w,c)]


# A greedy algorithm for the fractional knapsack problem.
# Note that we sort by v/w without modifying v or w so that we can
# output the indices of the actual items in the knapsack at the end
def fracKnapsack(v, w, W):
  order = bubblesortByRatio(v, w)            # sort by v/w (see bubblesort below)
  weight = 0.0                               # current weight of the solution
  value = 0.0                                # current value of the solution
  knapsack = []                              # items in the knapsack - a list of (item, faction) pairs
  n = len(v)
  index = 0                                  # order[index] is the index in v and w of the item we're considering
  while (weight < W) and (index < n):
    if weight + w[order[index]] <= W:        # if we can fit the entire order[index]-th item
      knapsack.append((order[index], 1.0))   # add it and update weight and value
      weight = weight + w[order[index]]
      value = value + v[order[index]]
    else:
      fraction = (W - weight) / w[order[index]]  # otherwise, calculate the fraction we can fit
      knapsack.append((order[index], fraction))  # and add this fraction
      weight = W
      value = value + v[order[index]] * fraction
    index = index + 1
  return (knapsack, value)                       # return the items in the knapsack and their value


# sort in descending order by ratio of list1[i] to list2[i]
# but instead of rearranging list1 and list2, keep the order in
# a separate array
# http://personal.denison.edu/~havill/courses/algorithmics/python/knapsack.py
def bubblesortByRatio(list1, list2):
  n = len(list1)
  order = range(n)
  for i in range(n - 1, 0, -1):     # i ranges from n-1 down to 1
    for j in range(0, i):           # j ranges from 0 up to i-1
      # if ratio of jth numbers > ratio of (j+1)st numbers then
      if ((1.0 * list1[order[j]]) / list2[order[j]]) < ((1.0 * list1[order[j+1]]) / list2[order[j+1]]):
        temp = order[j]              # exchange "pointers" to these items
        order[j] = order[j+1]
        order[j+1] = temp
  return order


if False:
    w = [1,2,3,4,5]
    v = [4,2,5,8,9]
    maxCost = 6

    answer = zeroOneKnapsack(v,w,maxCost)
    print "if my knapsack can hold %d pounds, i can get %d profit." % (maxCost,answer[0])
    print "\tby taking item(s): ",
    for i in range(len(answer[1])):
        if (answer[1][i] != 0):
            print i+1,

