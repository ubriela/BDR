__author__ = 'ubriela'

from matplotlib import pyplot
from shapely.geometry import Polygon
from descartes import PolygonPatch
from shapely.ops import cascaded_union

import sys
sys.path.append('../../../../../../_Research/_Crowdsourcing/_Privacy/privategeocrowddynamic/src/common')
sys.path.append('../plot/code')

from Utils import rect_area
from Params import Params
from FOV import FOV

from UtilsBDR import *

class Video(object):

    id = 0
    value = 0
    size = 0
    fov_count = 0

    def __init__(self):
        self.fovs = None    # list of FOVs

    def __init__(self, fovs):
        self.fovs = fovs

    def get_fovs(self):
        return self.fovs[0:self.fov_count]

    def c_union(self):
        if self.fovs is not None:
            polygons = [Polygon(mbr_to_path(fov.mbr())) for fov in self.fovs[0:self.fov_count]]
            u = cascaded_union(polygons)
            return u
        else:
            print None

    def area(self):
        if self.fovs is not None:
            return self.c_union().area / (1000000*Params.ONE_KM*Params.ONE_KM)
        else:
            print 'No FOV!'
            return 0

    """
    view point of the first FOV
    """
    def location(self):
        return [self.fovs[0].lat, self.fovs[0].lon]

    def sum_fov_area(self):
        return sum([fov.area() for fov in self.fovs])

    def to_str(self):
        return str(self.id) + "\n" + "\n".join(fov.to_str() for fov in self.fovs)


from figures import SIZE, GRAY, BLUE

COLOR = {
    True:  '#6699cc',
    False: '#ff3333'
    }

if False:
    fov1 = FOV(34.03331, -118.26935, 270, 60, 0.250)
    fov2 = FOV(34.03215, -118.26937, 255, 60, 0.250)
    fov_list = [fov2, fov1]
    video = Video(fov_list, 10)

    print fov1.mbr()
    print rect_area(fov2.mbr()), fov2.area(), video.area()

    fig = pyplot.figure(1, figsize=SIZE, dpi=90)

    ax = fig.add_subplot(122)
    patch2b = PolygonPatch(video.c_union(), fc=BLUE, ec=BLUE, alpha=0.5, zorder=2)
    ax.add_patch(patch2b)

    ax.set_title('union')
    xrange = [34.03, 34.04]
    yrange = [-118.28, -118.26]
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    ax.set_aspect(1)

    pyplot.show()