"""
Simple demo of a scatter plot.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../bdr')

from Params import Params
from MainExp import data_readin

param = Params(1000)
data = data_readin(param)

x = data[0]
y = data[1]

colors = [0 for i in range(len(x))]
area = [1 for i in range(len(x))]

# print x
# print y

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()
