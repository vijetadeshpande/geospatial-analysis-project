# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:57:47 2020

@author: Vijeta
"""


import seaborn as sns
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from kmodes.kprototypes import KPrototypes as KPro
import hdbscan
import os
from yellowbrick.cluster import KElbowVisualizer
from copy import deepcopy
from GeoSpatialPlot import GeoSpatialPlot as GSP
from GeoSpatialPlot import ClusterDistribution as CDP
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

class KMeans:
    def __init__(self, var, save_path):
        self.variable_list = var
        self.save_path = save_path
        self.optimal_k = {'Distortion score': None, 'Calinski-Harabasz score': None,
                          'Silhouette score': None}
        
    def elbow_test(self, df, k_values):
        
        # extract
        var_list = self.variable_list
        path_out = self.save_path
        
        # check if a directory for elbow test exists
        path_out = os.path.join(path_out, 'Elbow test results')
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        
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
        
        
    def plot_clusters(self, df, k_val, optional_boundary):
        
        # extract
        var_list = self.variable_list
        path_out = self.save_path
        
        # change path 
        path_out = os.path.join(path_out, 'KMeans plots')
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        
        # KwaZulu Natal plot
        plot_KZL = GSP(var = 'KMeans cluster labels_k = ' + str(k_val),
                       categorical = True,
                       plot_title = 'Regionalization with KMeans (K = ' + str(k_val) + ')',
                       save_path = path_out)
        plot_KZL.plot_map(geo_df = df, optional_boundary = optional_boundary)
        
        # eThekwini plot
        df_ETH = df #df.loc[df.District == 'ETH', :]
        plot_ETH = GSP(var = 'KMeans cluster labels_k = ' + str(k_val), 
                        categorical = True,
                        plot_title = 'Regionalization with KMeans (K = ' + str(k_val) + ')',
                        save_path = path_out)
        plot_ETH.plot_map(geo_df = df_ETH)
        
        # distribution of variables in each cluster
        CDP(var_list = var_list, 
            label_column = 'KMeans cluster labels_k = ' + str(k_val), 
            plot_title = 'K-means cluster facet plot' + str(k_val),
            save_path = path_out).plot_cluster_distribution(df)
        
        '''
        # TODO: plot de-normalized values here
        # Name (index) the rows after the category they belong
        to_plot = df.set_index('KMeans cluster labels_k = ' + str(k_val))
        to_plot = to_plot[var_list]
        to_plot = to_plot.stack()
        to_plot = to_plot.reset_index()
        to_plot = to_plot.rename(columns={'level_1': 'Variable', 0: 'Normalized values'})
        # Setup the facets
        facets = sns.FacetGrid(data=to_plot, row='Variable', hue='KMeans cluster labels_k = ' + str(k_val), \
                          sharey=False, sharex=False, aspect=2)
        # Build the plot as a `sns.kdeplot`
        cluster_facet_plot = facets.map(sns.kdeplot, 'Normalized values', shade=True).add_legend()
        cluster_facet_plot.savefig(os.path.join(path_out, 'Cluster facet plot' + str(k_val) + '.jpg'))

        '''
        
    def get_clusters(self, df, k_val, optional_boundary = None):
        
        # extract
        var_list = self.variable_list
        
        # create kmeans object and fit
        km_cluster = cluster.KMeans(n_clusters = k_val)
        km_cluster.fit(df[var_list])
        df['KMeans cluster labels_k = ' + str(k_val)] = km_cluster.labels_
    
        # plotting (first we will plot the whole KwaZulu and then we will plot
        # only for ETH district)
        self.plot_clusters(df, k_val, optional_boundary)

        return df

class HDBSCAN:
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
        # KwaZulu Natal plot
        plot_KZL = GSP(var = 'HDBSCAN cluster labels_' + 'CS = ' + str(cluster_s),
                       categorical = True,
                       plot_title = 'KwaZulu-Natal regionalization with HDBSCAN (cluster size = ' + str(cluster_s) + ')',
                       save_path = path_out)
        plot_KZL.plot_map(geo_df = df, optional_boundary = map_KwaZulu)
        

        # eThekwini plot
        df_ETH = df #df.loc[df.District == 'ETH', :]
        plot_KZL = GSP(var = 'HDBSCAN cluster labels_' + 'CS = ' + str(cluster_s),
                       categorical = True,
                       plot_title = 'eThekwini regionalization with HDBSCAN (cluster size = ' + str(cluster_s) + ')',
                       save_path = path_out)
        plot_KZL.plot_map(geo_df = df_ETH)
        
        # distribution of variables in each cluster
        CDP(var_list = var_list, 
            label_column = 'HDBSCAN cluster labels_' + 'CS = ' + str(cluster_s), 
            plot_title = 'HDBSCAN cluster facet plot ' + 'CS = ' + str(cluster_s),
            save_path = path_out).plot_cluster_distribution(df)
        
        '''
        # TODO: plot de-normalized values here
        # Name (index) the rows after the category they belong
        to_plot = df.set_index('HDBSCAN cluster labels_' + 'CS = ' + str(cluster_s))
        to_plot = to_plot[var_list]
        to_plot = to_plot.stack()
        to_plot = to_plot.reset_index()
        to_plot = to_plot.rename(columns={'level_1': 'Variable', 0: 'Normalized values'})
        # Setup the facets
        facets = sns.FacetGrid(data=to_plot, row='Variable', hue='HDBSCAN cluster labels_' + 'CS = ' + str(cluster_s), \
                          sharey=False, sharex=False, aspect=2)
        # Build the plot as a `sns.kdeplot`
        cluster_facet_plot = facets.map(sns.kdeplot, 'Normalized values', shade=True).add_legend()
        cluster_facet_plot.savefig(os.path.join(path_out, 'HDBSCAN cluster labels_' + 'CS = ' + str(cluster_s) + '.jpg'))
        
        '''

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
    
class KPrototype:
    
    def __init__(self, k_array = []):
        self.k_values = k_array
        self.optimal_k = {}
        
    def plot_clusters(self, k_val, df, var_list, map_KwaZulu, path_out):
        
        # change path 
        path_out = deepcopy(os.path.join(path_out, 'KPrototype plots'))
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        
        # KwaZulu Natal plot
        plot_KZL = GSP(var = 'KPrototype cluster labels_k = ' + str(k_val),
                       categorical = True,
                       plot_title = 'Regionalization with KPrototype (K = ' + str(k_val) + ')',
                       save_path = path_out)
        plot_KZL.plot_map(geo_df = df, optional_boundary = map_KwaZulu)
        
        # eThekwini plot
        df_ETH = df.loc[df.District == 'ETH', :]
        plot_ETH = GSP(var = 'KPrototype cluster labels_k = ' + str(k_val), 
                        categorical = True,
                        plot_title = 'Regionalization with KPrototype (K = ' + str(k_val) + ')',
                        save_path = path_out)
        plot_ETH.plot_map(geo_df = df_ETH)
        
        # distribution of variables in each cluster
        CDP(var_list = var_list, 
            label_column = 'KPrototype cluster labels_k = ' + str(k_val), 
            plot_title = 'K-prototype cluster facet plot' + str(k_val),
            save_path = path_out).plot_cluster_distribution(df)
        
        '''
        # distribution plot
        to_plot = df.set_index('KPrototype cluster labels_k = ' + str(k_val))
        to_plot = to_plot[var_list]
        to_plot = to_plot.stack()
        to_plot = to_plot.reset_index()
        to_plot = to_plot.rename(columns={'level_1': 'Variable', 0: 'Normalized values'})
        # Setup the facets
        facets = sns.FacetGrid(data=to_plot, row='Variable', 
                               hue='KPrototype cluster labels_k = ' + str(k_val), \
                               sharey=False, sharex=False, aspect=2)
        # Build the plot as a `sns.kdeplot`
        cluster_facet_plot = facets.map(sns.kdeplot, 'Normalized values', shade=True).add_legend()
        cluster_facet_plot.savefig(os.path.join(path_out, 'Cluster facet plot' + str(k_val) + '.jpg'))
    
        '''
        

    def get_clusters(self, df, var_list, k_values, map_sa_districts, path_out, cat_list = []):
        
        for k in k_values:
            # k prototype
            KPro_model = KPro(n_clusters = k)
            #df_geo.loc[:, columns4] = preprocessing.normalize(df_geo.loc[:, columns4].values)
            KPro_fit = KPro_model.fit(X = df[var_list], categorical = cat_list)
            df['KPrototype cluster labels'] = KPro_fit.labels_
            
            # plot
            self.plot_clusters(k, df, var_list, map_sa_districts, path_out)
            
        return df
        
class HierarchicalAC:
    
    def __init__(self, var_list, linkage_type, save_path):
        self.variable_list = var_list
        self.linkage_type = linkage_type
        self.save_path = save_path
        
    def plot_dendrogram(self, df):
        
        
        link_obj = linkage(df.loc[:, self.variable_list].values, self.linkage_type)
        plt.figure()
        dendrogram(link_obj)
        plt.savefig(os.path.join(self.save_path, 'Dendrogram for '+self.linkage_type+' linkage'+'.jpg'))
        
        return link_obj
    
    def plot_scatter(self, df, max_label, linkage_type, save_path):
        
        plt.figure()
        col_list = ['red', 'blue', 'orange', 'black', 'purple', 'gray', 'yellow']
        for i in range(0, max_label+1):
            plt.scatter(df.loc[df.loc[:, 'labels for '+linkage_type+' linkage'] == i, 'Social support index'],
                        df.loc[df.loc[:, 'labels for '+linkage_type+' linkage'] == i, 'Mental health index'],
                        color = col_list[i])
        plt.savefig(os.path.join(save_path, 'Scatter plot for '+linkage_type+' linkage'+'.jpg'))
        
        return
        
    
    def get_clusters(self, df, label_column, n_cluster = 4):
        
        # extract
        var_list = self.variable_list
        linkage_type = self.linkage_type
        
        # dendrogram
        _ = self.plot_dendrogram(df)
        
        # scatter plot
        model = AgglomerativeClustering(n_clusters = n_cluster, linkage = linkage_type)
        df[label_column] = model.fit(df.loc[:, var_list]).labels_
        self.plot_scatter(df, df[label_column].max(), linkage_type, self.save_path)
        
        # geospatial plot
        GSP(var = label_column, categorical = True,
            plot_title = 'Geospatial clusters for '+linkage_type+' linkage',
            save_path = self.save_path).plot_map(df)
        
        # cluster distribution
        CDP(var_list = var_list, 
                label_column = label_column, 
                plot_title = 'HAC cluster facet plot for ' +linkage_type+' linkage',
                save_path = self.save_path).plot_cluster_distribution(df)
        
        return df
        
    