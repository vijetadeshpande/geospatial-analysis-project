# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:11:39 2020

@author: Vijeta
"""

import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np


points = gpd.read_file('points.shp')
xmin,ymin,xmax,ymax =  points.total_bounds
width = 2000
height = 1000
rows = int(np.ceil(np.divide((ymax-ymin), height)))
cols = int(np.ceil(np.divide((xmax-xmin), width)))
XleftOrigin = xmin
XrightOrigin = xmin + width
YtopOrigin = ymax
YbottomOrigin = ymax - height
polygons = []
for i in range(cols):
   Ytop = YtopOrigin
   Ybottom =YbottomOrigin
   for j in range(rows):
       polygons.append(Polygon([(XleftOrigin, Ytop), (XrightOrigin, Ytop), (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)])) 
       Ytop = Ytop - height
       Ybottom = Ybottom - height
   XleftOrigin = XleftOrigin + width
   XrightOrigin = XrightOrigin + width

grid = gpd.GeoDataFrame({'geometry':polygons})
#grid.to_file("grid.shp")