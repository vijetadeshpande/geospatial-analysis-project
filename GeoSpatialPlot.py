# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:21:32 2020

@author: Vijeta
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
#import rasterio as rio
import seaborn as sns
import os


class GeoSpatialPlot:
    
    def __init__(self, var, categorical = False,
                 cmap = 'RdBu', plot_title = '', 
                 save_path = os.path.dirname(os.path.realpath(__file__))):
        
        self.variable_name = var
        self.variable_type = categorical
        self.plot_title = plot_title
        self.save_path = save_path
        self.file_name = os.path.join(save_path, plot_title + '.jpg')
        self.color_map = cmap
        
    def plot_map(self, geo_df, optional_boundary = pd.DataFrame(), boundary_col = 'k'):
        
        # convert to geopandas
        geo_df = gpd.GeoDataFrame(geo_df)
        
        # plot values on map
        f, ax = plt.subplots(1, figsize=(12, 12))
        geo_df.plot(column = self.variable_name, cmap = self.color_map,
                    categorical = self.variable_type, legend = True, 
                    ax = ax, linewidth = 0, edgecolor = 'white')
        if optional_boundary.empty:
            geo_df.geometry.boundary.plot(color = None, edgecolor = boundary_col, 
                                                          linewidth = 0.7, ax = ax)
        else:
            optional_boundary.geometry.boundary.plot(color = None, edgecolor = boundary_col, 
                                                          linewidth = 0.7, ax = ax)
              
        # naming and stuff
        ax.set_axis_off()
        plt.axis('equal')
        plt.title(self.plot_title)
        plt.savefig(self.file_name)
        
class ClusterDistribution:
    
    def __init__(self, var_list, label_column, plot_title, save_path, facet_title = 'Variable', x_title = 'Normalized values'):
        self.facet_variables = var_list
        self.plot_title = plot_title
        self.facet_title = facet_title
        self.x_titile = x_title
        self.label_column = label_column
        self.save_path = os.path.join(save_path, plot_title + '.jpg')
    
    def plot_cluster_distribution(self, df):
        
        # extract values
        var_list = self.facet_variables
        label_col = self.label_column
        
        # TODO: plot de-normalized values here
        # Name (index) the rows after the category they belong
        to_plot = df.set_index(label_col)
        to_plot = to_plot[var_list]
        to_plot = to_plot.stack()
        to_plot = to_plot.reset_index()
        to_plot = to_plot.rename(columns={'level_1': self.facet_title, 0: self.x_titile})
        # Setup the facets
        facets = sns.FacetGrid(data=to_plot, row=self.facet_title, hue=label_col, \
                          sharey=False, sharex=False, aspect=2)
        # Build the plot as a `sns.kdeplot`
        cluster_facet_plot = facets.map(sns.kdeplot, self.x_titile, shade=True).add_legend()
        cluster_facet_plot.savefig(self.save_path)

        
        
        
        