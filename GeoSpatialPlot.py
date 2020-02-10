# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:21:32 2020

@author: Vijeta
"""

import pandas as pd
import geopandas as gpd
import libpysal as lp
import matplotlib.pyplot as plt
#import rasterio as rio
import numpy as np
#import contextily as ctx
import shapely.geometry as geom
from shapely.geometry import Point
import seaborn as sns
import os


class GeoSpatialPlot:
    
    def __init__(self, var, val_type = None, cmap = 'RdBu', plot_title = '', save_path = os.path.dirname(os.path.realpath(__file__))):
        
        self.variable_name = var
        self.variable_value_type = val_type
        self.plot_title = plot_title
        self.save_path = save_path
        self.file_name = os.path.join(save_path, plot_title + '.jpg')
        self.color_map = cmap
        
    def plot_map(self, geo_df, title = None):
        
        # check type of input
        if not isinstance(geo_df, gpd.GeoDataFrame):
            raise ValueError("Input data should be of type GeoDataFrame")
        
        # plot values on map
        f, ax = plt.subplots(1, figsize=(12, 12))
        geo_df.plot(column = self.variable_name, cmap = self.color_map, 
                    legend = True, ax = ax, linewidth = 0.1, edgecolor = 'white')
        ax.set_axis_off()
        plt.axis('equal')
        plt.title(self.plot_title)
        plt.savefig(self.file_name)