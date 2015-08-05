import sys
import logging

import numpy as np

from Kd_pure import Kd_pure
from Params import Params


class Kd_standard(Kd_pure):
    """ standard kd-tree """

    def __init__(self, data, param):
        Kd_pure.__init__(self, data, param)