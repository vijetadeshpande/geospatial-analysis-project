# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:57:47 2020

@author: Vijeta
"""


import seaborn as sns
import pandas as pd
import pysal as ps
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import DBSCAN
import hdbscan
import os
from shapely.geometry import Point, Polygon
from sklearn import preprocessing
from yellowbrick.cluster import KElbowVisualizer
from copy import deepcopy
#from geopy import geocoders  
#from geopy.geocoders import Nominatim

class KMeans:
    def __init__(self):
        self.optimal_k = {'Distortion score': None, 'Calinski-Harabasz score': None,
                          'Silhouette score': None}
        
    def elbow_test(self, df, var_list, k_values, path_out):
        
        # check if a directory for elbow test exists
        if not os.path.exists(os.path.join(path_out, 'Elbow test results')):
            os.makedirs(os.path.join(path_out, 'Elbow test results'))
        path_out = deepcopy(os.path.join(path_out, 'Elbow test results'))
        
        # based on distortion score
        plt.figure()
        elbow_k = cluster.KMeans()
        visualizer = KElbowVisualizer(elbow_k, k = (min(k_values),max(k_values)))
        visualizer.fit(df[var_list])
        visualizer.show(outpath = os.path.join(path_out, "kelbow_minibatchkmeans.jpg"))
        optimal_k = {'Distortion score': visualizer.knee_value}
        
        # based on calinski_harabasz
        plt.figure()
        visualizer = KElbowVisualizer(elbow_k, k = (min(k_values),max(k_values)), 
                                      metric='calinski_harabasz', 
                                      timings=False, locate_elbow=True)
        visualizer.fit(df[var_list])
        visualizer.show(outpath = os.path.join(path_out, "kelbow_calinski-harabasz.jpg"))
        optimal_k['Calinski-Harabasz score'] = visualizer.knee_value
        
        # based on silhouette score
        plt.figure()
        visualizer = KElbowVisualizer(elbow_k, k = (min(k_values),max(k_values)), 
                                      metric='silhouette', 
                                      timings=False, locate_elbow=True)
        visualizer.fit(df[var_list])
        visualizer.show(outpath = os.path.join(path_out, "silhouette.jpg"))
        optimal_k['Silhouette score'] = visualizer.get_params()
        
        # set the optimal values of k
        self.optimal_k = optimal_k
        
        
    def plot_clusters(self, k_val, df, var_list, map_KwaZulu, path_out):
        
        # change path 
        if not os.path.exists(os.path.join(path_out, 'KMeans plots')):
            os.makedirs(os.path.join(path_out, 'KMeans plots'))
        path_out = deepcopy(os.path.join(path_out, 'KMeans plots'))
        
        # KwaZulu Natal plot
        #dist_list = ['Amajuba District', 'Ugu District', 'uMgungundlovu District',
        #             'Umzinyathi District', 'Uthukela District', 'Zululand District',
        #             'Umkhanyakude District', 'iLembe District', 'eThekwini Metropolitan',
        #             'Sisonke District']
        #map_KwaZulu = map_sa_districts.loc[map_sa_districts.name.isin(dist_list), :]
        f, ax = plt.subplots(1, figsize=(9, 9))
        col_map = 'Pastel1'#sns.color_palette("muted", n_colors = 4)
        df.plot(column='KMeans cluster labels_k = ' + str(k_val), categorical=True, legend=True, 
                linewidth=0, axes=ax, cmap = col_map)
        # boundary plot
        map_KwaZulu.geometry.boundary.plot(color = None, edgecolor = 'k', 
                                           linewidth = 0.7, ax = ax)
        ax.set_axis_off()
        plt.axis('equal')
        plt.title('Regionalization with KMeans (K = ' + str(k_val) + ')')
        plt.savefig(os.path.join(path_out, 'Regionalization_KMeans_KwaZulu_N_' + str(k_val) + '.jpg'))
        
        # eThekwini plot
        #map_eThekwini = map_KwaZulu.loc[map_KwaZulu.CAT_B == 'ETH', :]
        df_ETH = deepcopy(df)#df.loc[df.District == 'ETH', :]
        f, ax = plt.subplots(1, figsize=(9, 9))
        col_map = 'Pastel1'#sns.color_palette("muted", n_colors = 4)
        df_ETH.plot(column='KMeans cluster labels_k = ' + str(k_val), categorical=True, legend=True, 
                    linewidth=0, axes=ax, cmap = col_map)
        # boundary plot
        df_ETH.geometry.boundary.plot(color = None, edgecolor = 'k', 
                                             linewidth = 0.7, ax = ax)
        ax.set_axis_off()
        plt.axis('equal')
        plt.title('Regionalization with KMeans (K = ' + str(k_val) + ')')
        plt.savefig(os.path.join(path_out, 'Regionalization_KMeans_eThekwini_' + str(k_val) + '.jpg'))

        # distribution within clusters
        #ksizes = df.groupby('KMeans cluster labels').size()
        #cluster_size_plot = ksizes.plot(kind='bar')
        #cluster_size_plot.savefig('Cluster size plot' + str() + '.jpg')
        #kmeans = df.groupby('kcls')[var_list].mean()
        #kdesc = df.groupby('kcls')[var_list].describe()
        
        # Name (index) the rows after the category they belong
        to_plot = df.set_index('KMeans cluster labels_k = ' + str(k_val))
        to_plot = to_plot[var_list]
        # Display top of the table
        to_plot.head()
        to_plot = to_plot.stack()
        to_plot.head()
        to_plot = to_plot.reset_index()
        to_plot.head()
        to_plot = to_plot.rename(columns={'level_1': 'Variable', 0: 'Normalized values'})
        to_plot.head()
        # Setup the facets
        facets = sns.FacetGrid(data=to_plot, row='Variable', hue='KMeans cluster labels_k = ' + str(k_val), \
                          sharey=False, sharex=False, aspect=2)
        # Build the plot as a `sns.kdeplot`
        cluster_facet_plot = facets.map(sns.kdeplot, 'Normalized values', shade=True).add_legend()
        cluster_facet_plot.savefig(os.path.join(path_out, 'Cluster facet plot' + str(k_val) + '.jpg'))

        
    def get_clusters(self, df, var_list, k_values, map_sa_districts, path_out):
        
        # check if the elbow test is done or not
        #if self.optimal_k['Distortion score'] == None:
        #    self.elbow_test(df, var_list, k_values, path_out)
        
        # use distortion score-knee point as the optimal k value
        #optimal_k = self.optimal_k['Distortion score']
        for k_val in k_values:
            km_cluster = cluster.KMeans(n_clusters = k_val)
            km_cluster.fit(df[var_list])
            df['KMeans cluster labels_k = ' + str(k_val)] = km_cluster.labels_
        
            # plotting (first we will plot the whole KwaZulu and then we will plot
            # only for ETH district)
            self.plot_clusters(k_val, df, var_list, map_sa_districts, path_out)

        return df

class HDBSCAN_SA:
    def __init__(self):
        self.optimal_cluster_size = None
        
    def plot_clusters(self, hdb, cluster_s, df, var_list, map_KwaZulu, path_out):
        
        # change path 
        if not os.path.exists(os.path.join(path_out, 'HDBSCAN plots')):
            os.makedirs(os.path.join(path_out, 'HDBSCAN plots'))
        path_out = deepcopy(os.path.join(path_out, 'HDBSCAN plots'))
        
        # Min spanning tree
        plt.figure()
        hdb.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                              edge_alpha=0.6,
                                              node_size=80,
                                              edge_linewidth=2)
        plt.savefig(os.path.join(path_out, 'HDBSCAN_MST_cluster_size_' + str(cluster_s) + '.jpg'))
        
        # single linkage tree
        plt.figure()
        hdb.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
        plt.savefig(os.path.join(path_out, 'HDBSCAN_Single_linkage_tree_cluster_size_' + str(cluster_s) + '.jpg'))
        
        # condensed tree
        plt.figure()
        hdb.condensed_tree_.plot()
        plt.savefig(os.path.join(path_out, 'HDBSCAN_Condensed_tree_cluster_size_' + str(cluster_s) + '.jpg'))

        # geospatial plot
        # KwaZulu plot
        #dist_list = ['Amajuba District', 'Ugu District', 'uMgungundlovu District',
        #             'Umzinyathi District', 'Uthukela District', 'Zululand District',
        #             'Umkhanyakude District', 'iLembe District', 'eThekwini Metropolitan',
        #             'Sisonke District']
        #map_KwaZulu = map_sa_districts.loc[map_sa_districts.name.isin(dist_list), :]
        f, ax = plt.subplots(1, figsize=(9, 9))
        col_map = 'Pastel1'#sns.color_palette("muted", n_colors = 5)
        df.plot(column = 'HDBSCAN cluster labels_' + 'CS = ' + str(cluster_s), categorical=True, legend=True, 
                    linewidth=0, axes=ax, cmap = col_map)
        # boundary plot
        map_KwaZulu.geometry.boundary.plot(color = None, edgecolor = 'k', 
                                           linewidth = 0.7, ax = ax)
        ax.set_axis_off()
        plt.axis('equal')
        plt.title('Regionalization with HDBSCAN (cluster size = ' + str(cluster_s) + ')')
        plt.savefig(os.path.join(path_out, 'Regionalization_DBSCAN_KwaZulu_N_size_' + str(cluster_s) + '.jpg'))
        
        # eThekwini plot
        #map_eThekwini = map_KwaZulu.loc[map_KwaZulu.CAT_B == 'ETH', :]
        df_ETH = df.loc[df.District == 'ETH', :]
        f, ax = plt.subplots(1, figsize=(9, 9))
        col_map = 'Pastel1'#sns.color_palette("muted", n_colors = 4)
        df_ETH.plot(column='HDBSCAN cluster labels_' + 'CS = ' + str(cluster_s), categorical=True, legend=True, 
                    linewidth=0, axes=ax, cmap = col_map)
        # boundary plot
        df_ETH.geometry.boundary.plot(color = None, edgecolor = 'k', 
                                             linewidth = 0.7, ax = ax)
        ax.set_axis_off()
        plt.axis('equal')
        plt.title('Regionalization with HDBSCAN (cluster size = ' + str(cluster_s) + ')')
        plt.savefig(os.path.join(path_out, 'Regionalization_HDBSCAN_eThekwini_size_' + str(cluster_s) + '.jpg'))
        
        # Name (index) the rows after the category they belong
        to_plot = df.set_index('HDBSCAN cluster labels_' + 'CS = ' + str(cluster_s))
        to_plot = to_plot[var_list]
        # Display top of the table
        to_plot.head()
        to_plot = to_plot.stack()
        to_plot.head()
        to_plot = to_plot.reset_index()
        to_plot.head()
        to_plot = to_plot.rename(columns={'level_1': 'Variable', 0: 'Normalized values'})
        to_plot.head()
        # Setup the facets
        facets = sns.FacetGrid(data=to_plot, row='Variable', hue='HDBSCAN cluster labels_' + 'CS = ' + str(cluster_s), \
                          sharey=False, sharex=False, aspect=2)
        # Build the plot as a `sns.kdeplot`
        cluster_facet_plot = facets.map(sns.kdeplot, 'Normalized values', shade=True).add_legend()
        cluster_facet_plot.savefig(os.path.join(path_out, 'HDBSCAN cluster labels_' + 'CS = ' + str(cluster_s) + '.jpg'))
        
    def get_clusters(self, df, var_list, cluster_size_values, map_sa, path_out):
        
        # iterate over the lowerbound values for cluster size
        for cluster_s in cluster_size_values:
            
            # clustering
            hdb_cluster = hdbscan.HDBSCAN(min_cluster_size = int(cluster_s), 
                                          gen_min_span_tree = True, 
                                          cluster_selection_method = 'leaf')
            hdb_cluster.fit(df[var_list])
            df['HDBSCAN cluster labels_' + 'CS = ' + str(cluster_s)] = hdb_cluster.labels_
            
            # plotting
            self.plot_clusters(hdb_cluster, cluster_s, df, var_list, map_sa, path_out)
        
        return df
    
    
    