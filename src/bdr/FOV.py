__author__ = 'ubriela'

import math
from shapely.geometry import Polygon

from Params import Params
from UtilsBDR import mbr_to_cellids, mbr_to_path, distance_km, angle_bwn_two_points


"""
Field of View
"""
class FOV(object):
    id = 0

    def __init__(self):
        self.lat = 0.0
        self.lon = 0.0
        self.compass = 0
        self.alpha = 60
        self.R = 250


    def __init__(self, geojson):
        self.lat = geojson.geometry.coordinates[1]
        self.lon = geojson.geometry.coordinates[0]
        self.compass = float(geojson.properties['theta_x'])
        self.R = float(geojson.properties['r']) * 20
        self.alpha = float(geojson.properties['alpha'])

    def __init__(self, lat, lon, compass, R, alpha):
        self.lat = lat
        self.lon = lon
        self.compass = compass
        self.R = R
        self.alpha = alpha

    def metadata(self):
        return self.lat,self.lon,self.compass, self.alpha, self.R

    def area_m2(self):
        return 180/self.alpha*math.pi*self.R*self.R

    def area(self):
        polygon = Polygon(mbr_to_path(self.mbr()))
        return polygon.area/(1000000*Params.ONE_KM*Params.ONE_KM)

    """
    Mininum Bounding Rectangle of the FOV
    """
    def mbr(self):
        lat, lon, compass, alpha, R = self.metadata()
        R = R/1000.0

        sin_plus = 360 *math.sin((compass+alpha/2)*3.1415926/180);
        sin_minus = 360 *math.sin((compass-alpha/2)*3.1415926/180);
        cos_plus = 360 *math.cos((compass+alpha/2)*3.1415926/180);
        cos_minus = 360 *math.cos((compass-alpha/2)*3.1415926/180);

        # Min Lng
        mbr_left = lon
        if lon - R*sin_plus/40075.017 < mbr_left:
            mbr_left = lon - R*sin_plus/40075.017
        if lon + R*sin_plus/40075.017 < mbr_left:
            mbr_left = lon + R*sin_plus/40075.017
        if lon - R*sin_minus/40075.017 < mbr_left:
            mbr_left = lon - R*sin_minus/40075.017
        if lon + R*sin_minus/40075.017 < mbr_left:
            mbr_left = lon + R*sin_minus/40075.017

        # Max Lng
        mbr_right = lon;
        if lon - R*sin_plus/40075.017 > mbr_right:
            mbr_right = lon - R*sin_plus/40075.017
        if lon + R*sin_plus/40075.017 > mbr_right:
            mbr_right = lon + R*sin_plus/40075.017
        if lon - R*sin_minus/40075.017 > mbr_right:
            mbr_right = lon - R*sin_minus/40075.017
        if lon + R*sin_minus/40075.017 > mbr_right:
            mbr_right = lon + R*sin_minus/40075.017

        # Min Lat
        mbr_bottom = lat
        if lat - R*cos_plus/40007.86 < mbr_bottom:
            mbr_bottom = lat - R*cos_plus/40007.86
        if lat + R*cos_plus/40007.86 < mbr_bottom:
            mbr_bottom = lat + R*cos_plus/40007.86
        if lat - R*cos_minus/40007.86 < mbr_bottom:
            mbr_bottom = lat - R*cos_minus/40007.86
        if lat + R*cos_minus/40007.86 < mbr_bottom:
            mbr_bottom = lat + R*cos_minus/40007.86

        # Max Lat
        mbr_ceil = lat;
        if lat - R*cos_plus/40007.86 > mbr_ceil:
            mbr_ceil = lat - R*cos_plus/40007.86
        if lat + R*cos_plus/40007.86 > mbr_ceil:
            mbr_ceil = lat + R*cos_plus/40007.86
        if lat - R*cos_minus/40007.86 > mbr_ceil:
            mbr_ceil = lat - R*cos_minus/40007.86
        if lat + R*cos_minus/40007.86 > mbr_ceil:
            mbr_ceil = lat + R*cos_minus/40007.86

        return [[mbr_bottom, mbr_left], [mbr_ceil, mbr_right]]

    """
    Return a set of cell ids that this FOV covers
    """
    def cellids(self, param):
        return mbr_to_cellids(self.mbr(), param)

    def to_str(self):
        content = "\t".join(map(str, [self.lat, self.lon, self.compass, self.alpha, self.R]))
        if self.id:
            return str(self.id) + "\t" + content
        return content

    """
    Check if a point inside a circle sector
    """
    def cover(self, plat, plng):
        angle = angle_bwn_two_points(self.lat, self.lon, plat, plng)
        distance = distance_km(self.lat, self.lon, plat, plng)
        # print angle, distance
        if abs(angle - self.compass/2) < 30 and distance < self.R:
            return True
        else:
            return False
    # print within_circular(34.024734, -118.284988,34.018212,-118.291716,45,1)