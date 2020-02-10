# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:43:34 2020

@author: Vijeta
"""

import pandas as pd
import geopandas as gpd
import libpysal as lp
import esda
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geom
from shapely.geometry import Point
import seaborn as sns
import os
from matplotlib import colors
from GeoSpatialPlot import GeoSpatialPlot as GSP


class SpatialAutocorrelation:
    
    def __init__(self, df, var, save_path = os.path.dirname(os.path.realpath(__file__)), type_w = 'Queen'):
        
        self.variable               = var
        self.save_path              = save_path
        self.weight_mat             = None
        self.spatial_lag_vector     = None#self.get_spatial_lag(df, type_w)
        self.moran_i_coeff_object   = self.get_moran_coefficient(df, type_w)
        
    def get_moran_coefficient(self, df, type_w):
        
        if type_w == 'Queen':
            wq = lp.weights.Queen.from_dataframe(df)
        else:
            wq = lp.weights.Rook.from_dataframe(df)
        wq.transform = 'r'
        self.weight_mat = wq
        
        # calculate spatial lag
        y = df[self.variable]
        y_lag = lp.weights.lag_spatial(wq, y)
        
        # store value
        self.spatial_lag_vector = y_lag
        
        # calculate moran coefficient
        mi = esda.moran.Moran(y, wq)
        
        return mi
    
    def plot_null_hypo(self, var):
        
        # check input type
        moran_i = self.moran_i_coeff_object
        
        # plot
        # moran_i.sim = (if permutations > 0) vector of I values for permuted samples
        plt.figure()
        sns.kdeplot(moran_i.sim, shade=True)
        plt.vlines(moran_i.I, 0, 1, color='r')
        plt.vlines(moran_i.EI, 0, 1)
        plt.xlabel("Moran's I")
        file_name = os.path.join(self.save_path, 'Moran\'s NULL plot for ' + var + '.jpg')
        plt.savefig(file_name)
        
        return
    
    def plot_moran_scatter(self, y, var):
        
        # get variables
        ylag = self.spatial_lag_vector
        
        # Moran scatter plot
        b, a = np.polyfit(y, ylag, 1)
        f, ax = plt.subplots(1, figsize=(9, 9))
        
        plt.plot(y, ylag, '.', color='firebrick')
        
         # dashed vert at mean of the price
        plt.vlines(y.mean(), ylag.min(), ylag.max(), linestyle='--')
         # dashed horizontal at mean of lagged price 
        plt.hlines(ylag.mean(), y.min(), y.max(), linestyle='--')
        
        # red line of best fit using global I as slope
        plt.plot(y, a + b*y, 'r')
        plt.title('Moran Scatterplot')
        plt.ylabel('Spatial Lag of' + var)
        plt.xlabel(var)
        file_name = os.path.join(self.save_path, 'Moran\'s scatter plot for ' + var + '.jpg')
        plt.savefig(file_name)
        
        return
    
    def plot_spot_map(self, df, var, var_type = 'high desirable'):
        
        # access avlues
        wq = self.weight_mat
        moran_i = esda.moran.Moran_Local(df[var], wq)
        
        # laber the points as per the quadrant that they fall into for
        # the y vs y_lag plot
        sig = 1 * (moran_i.p_sim < 0.05)
        if var_type == 'high desirable':
            coldspot = 3 * (sig * moran_i.q==1)
            hotspot = 1 * (sig * moran_i.q==3)
            diamond = 4 * (sig * moran_i.q==2)
            doughnut = 2 * (sig * moran_i.q==4)
        else:
            coldspot = 1 * (sig * moran_i.q==1)
            hotspot = 3 * (sig * moran_i.q==3)
            diamond = 2 * (sig * moran_i.q==2)
            doughnut = 4 * (sig * moran_i.q==4)
        spots = hotspot + coldspot + doughnut + diamond
        spot_labels = [ '0 ns', '1 hot spot', '2 doughnut', '3 cold spot', '4 diamond']
        labels = [spot_labels[i] for i in spots]
        df = df.assign(spot_label=labels)
        
        # define colors
        hmap = colors.ListedColormap([ 'lightgrey', 'red', 'lightblue', 'blue', 'pink'])
        
        # create figure, plot and save
        plot_t = 'Moran\'s I spot map for' + var
        GSP(var = 'spot_label', cmap = hmap, plot_title = plot_t, 
            save_path = self.save_path).plot_map(df)
        
        return df

        
        