# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:52:56 2020

@author: Vijeta
"""

'''
EXPLORING GEO_CLUSTERS IN CLINICAL TRIAL DATA
'''

from shapely.geometry import Point
import seaborn as sns
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import os
from sklearn import preprocessing
import GeoSpatialClustering as GSC
from AuxMethods import AuxMethods as Aux
from GeoSpatialPlot import GeoSpatialPlot as GSP
from SpatialAutocorrelation import SpatialAutocorrelation as SAC
import data_cleaning

#%% read data
if False:
    path = r'C:\Users\Vijeta\Documents\Projects\Sizanani\Data'
    dict_maps, dict_geo_df, df_clustering = data_cleaning.clean_data(path)


#%% distribution and correlation

if False: 
    # create folder
    if not os.path.exists(os.path.join(path, 'Exploratory plots')):
        os.makedirs(os.path.join(path, 'Exploratory plots'))
    path_expl = os.path.join(path, 'Exploratory plots')
    f, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 12))
    # Make the axes accessible with single indexing
    axs = axs.flatten()
    
    # Start the loop over all the variables of interest
    for i, col in enumerate(['support', 'CD4Results', 'MHIINDX3']):
        # select the axis where the map will go
        ax = axs[i]
        dict_geo_df['HIV'].plot(column=col, axes=ax, scheme='Quantiles', linewidth=0, colormap='Blues')
        dict_maps['KwaZulu-Natal districts'].geometry.boundary.plot(color = None, edgecolor = 'k', 
                                               linewidth = 0.5, ax = ax)
        ax.set_axis_off()
        ax.set_title(col)
    # save the figure
    plt.savefig(os.path.join(path_expl, 'Geospatial exploration.jpg'))
    
    # correlation matrix plot
    corr_plot = sns.pairplot(dict_geo_df['HIV'].loc[:, ['support', 'CD4Results', 'MHIINDX3']], kind='reg')#, diag_kind='kde')
    corr_plot.savefig(os.path.join(path_expl, 'Correlation exploration.jpg'))


#%% CLUSTERING 
def df_grouping(df, var, by = 'ward number'):
    ward_geo = dict_maps['Wards'].loc[dict_maps['Wards'].loc[:, 'MUNICNAME'] == 'Ethekwini Metropolitan Municipality', ['WARDNO', 'geometry']] 
    ward_geo['ward number'] = ward_geo['WARDNO']
    df = gpd.GeoDataFrame(df.loc[:, var].groupby(by).median()).reset_index()
    #df = gpd.GeoDataFrame(df.loc[:, var].groupby(by).sum()).reset_index()
    df = df.merge(ward_geo, on = 'ward number', how = 'left')
    
    return df

# First we will restrict our analysis to eThekwini
df_clustering_all = dict_geo_df['All'].loc[dict_geo_df['All'].district == 'ETH', :]
df_clustering_all = df_clustering_all.rename(columns = {'MHIINDX3': 'Mental health index', 'support': 'Social support index'})
df_clustering_hiv = dict_geo_df['HIV'].loc[dict_geo_df['HIV'].district == 'ETH', :]
df_clustering_hiv = df_clustering_hiv.rename(columns = {'MHIINDX3': 'Mental health index', 'support': 'Social support index', 'CD4Results': 'CD4 count'})

# clustering for social support and mental health index
df_clustering_all = df_grouping(df_clustering_all, var = ['Social support index', 'Mental health index', 'ward number'])
df_clustering_hiv = df_grouping(df_clustering_hiv, var = ['Social support index', 'Mental health index', 'CD4 count', 'ward number'])
   

if False: 

    # Kmeans clustering
    # only ssi and mhi
    KMeans_ss_mhi = GSC.KMeans()
    var = ['Social support index', 'Mental health index']
    KMeans_ss_mhi.elbow_test(df_clustering_all, var, np.arange(3, 10), path)
    df_KMeans_ss_mhi = KMeans_ss_mhi.get_clusters(df_clustering_all, var, [4], dict_maps['KwaZulu-Natal districts'], path)
    # only cd4
    KMeans_hiv = GSC.KMeans()
    var = ['CD4 count']
    KMeans_hiv.elbow_test(df_clustering_hiv, var, np.arange(3, 10), path)
    df_KMeans_hiv = KMeans_hiv.get_clusters(df_clustering_hiv, var, [6], dict_maps['KwaZulu-Natal districts'], path)
    # together
    KMeans_all = GSC.KMeans()
    var = ['Social support index', 'Mental health index', 'CD4 count']
    KMeans_all.elbow_test(df_clustering_hiv, var, np.arange(3, 10), path)
    df_KMeans_all = KMeans_all.get_clusters(df_clustering_hiv, var, [4], dict_maps['KwaZulu-Natal districts'], path)

    # HDBSCAN
    # only ssi and mhi
    HDBSCAN_ssi_mhi = GSC.HDBSCAN()
    var = ['Social support index', 'Mental health index']
    HDBSCAN_ssi_mhi = HDBSCAN_ssi_mhi.get_clusters(df_clustering_all, var, np.arange(2,12, 2), dict_maps['KwaZulu-Natal districts'], path)
    # together
    HDBSCAN_all = GSC.HDBSCAN()
    var = ['Social support index', 'Mental health index', 'CD4 count']
    HDBSCAN_all = HDBSCAN_all.get_clusters(df_clustering_hiv, var, np.arange(2,12, 2), dict_maps['KwaZulu-Natal districts'], path)
    


#%% Hierarchical clustering
var = ['Social support index', 'Mental health index', 'CD4 count']
# agglomerative with different linkage
save_path = os.path.join(path, 'Agglomerative clustering results')
if not os.path.exists(save_path):
    os.mkdir(save_path)

for link in ['single', 'complete', 'average', 'ward']:
    
    df_clustering_hiv = GSC.HierarchicalAC(var_list = var,
                                           linkage_type = link,
                                           save_path = save_path).get_clusters(df_clustering_hiv, label_column = 'labels for '+link+' linkage', n_cluster = 3)

#%% Clujstering on 9-month follow-up data

if False:  
    # right join the location df
    var = ['StudyID', 'support_9mon', 'MHIINDX3_9mon', 'Longitude', 'Latitude']
    df_clustering_all = df_clustering.loc[:, var].dropna()
    
    #
    x = dict_df_clustering
    dict_df_clustering = {'Baseline': x, 'Follow-up': {'HIV': {'Original': None, 'Standardized': None}, 'All': {'Original': None, 'Standardized': None}}}
    del x
    #dict_df_clustering['Follow-up']['HIV']['Original'] = df_clustering_HIV
    dict_df_clustering['Follow-up']['All']['Original'] = df_clustering_all
    del df_clustering_all#, df_clustering_HIV
    
    # normalizing values to for creating a common heatmap
    #df.loc[:, columns] = preprocessing.normalize(df.loc[:, columns].values)
    #df.loc[:, columns] = preprocessing.MinMaxScaler().fit_transform(df.loc[:, columns].values)
    #dict_df_clustering['Follow-up']['HIV']['Standardized'] = dict_df_clustering['Follow-up']['HIV']['Original']
    #dict_df_clustering['Follow-up']['HIV']['Standardized'].loc[:, ['support', 'CD4Results', 'MHIINDX3']] = preprocessing.scale(dict_df_clustering['Follow-up']['HIV']['Original'].loc[:, ['support_9mon', 'CD4Results', 'MHIINDX3_9mon']].values)
    dict_df_clustering['Follow-up']['All']['Standardized'] = dict_df_clustering['Follow-up']['All']['Original']
    dict_df_clustering['Follow-up']['All']['Standardized'].loc[:, ['support_9mon', 'MHIINDX3_9mon']] = preprocessing.scale(dict_df_clustering['Follow-up']['All']['Original'].loc[:, ['support_9mon', 'MHIINDX3_9mon']].values)
    
    # geopandas object
    dict_geo_df_9 = {'HIV': None, 'All': None}
    df_dist_ward = dict_maps['Wards'].loc[dict_maps['Wards'].PROVINCE.isin(['KwaZulu-Natal', 'Eastern Cape']), ['CAT_B', 'geometry', 'WARDNO']]
    df_dist_ward = df_dist_ward.reset_index(drop = True)
    dict_geo_df_9['All'] = Aux().pair_point_to_polygon(dict_df_clustering['Follow-up']['All']['Standardized'], df_dist_ward)
    
    # grouping
    df_clustering_all = dict_geo_df_9['All'].loc[dict_geo_df_9['All'].district == 'ETH', :]
    df_clustering_all = df_clustering_all.rename(columns = {'MHIINDX3_9mon': 'Mental health index', 'support_9mon': 'Social support index'})
    # clustering for social support and mental health index
    df_clustering_all = df_grouping(df_clustering_all, var = ['Social support index', 'Mental health index', 'ward number'])
    
    # clustering
    # Kmeans clustering
    # only ssi and mhi
    KMeans_ssi_mhi = GSC.KMeans()
    var = ['Social support index', 'Mental health index']
    KMeans_ssi_mhi.elbow_test(df_clustering_all, var, np.arange(3, 10), path)
    df_KMeans_ssi_mhi = KMeans_ssi_mhi.get_clusters(df_clustering_all, var, [5], dict_maps['KwaZulu-Natal districts'], path)
    
    #%% Analyze death cases
    df_clustering['total deaths'] = df_clustering['death_9mon'] + df_clustering['death_after9']
    
    df_death = df_clustering.loc[df_clustering['total deaths'] == 1, :]
    df_death_hiv = df_death.loc[df_death['CD4Results'] >= 0, :]
    df_death = Aux().pair_point_to_polygon(df_death, df_dist_ward)
    df_death = df_death.loc[df_death.district == 'ETH', :]
    df_death = df_grouping(df_death, ['total deaths', 'ward number'])
    # plot total deaths
    GSP_death = GSP(var = 'total deaths', 
                    plot_title = 'Total deaths in wards',
                    save_path = path,
                    cmap = 'RdBu_r')
    GSP_death.plot_map(df_death)
    
    # check autocorrelation of total deaths
    # Moran's I
    SAC_object_death = SAC(df_death, var = ['total deaths'], save_path = path)
    SAC_object_death.plot_null_hypo('total deaths')
    SAC_object_death.plot_moran_scatter(df_death['total deaths'], 'total deaths')
    SAC_object_death.plot_spot_map(df_death, 'total deaths', var_type = 'low desirable')
    
    df_death = df_death.reset_index(drop = True)
    
    #%% point wise analysis
    '''
    df = deepcopy(df_KMeans_clusters)
    # eThekwini plot
    df_ETH = df.loc[df.District == 'ETH', :]
    df_ETH['Ward'] = df_ETH['geometry']
    df_ETH_point = deepcopy(df_ETH)
    df_ETH_point['geometry'] = df_ETH_point['points']
    
    for cluster_val in [0,1,2,3]:
        
        # get only specific cluster
        df_cluster = df_ETH_point.loc[df_ETH_point['KMeans cluster labels_k = 4'] == cluster_val, :]
        
        f, ax = plt.subplots(1, figsize=(9, 9))
        col_map = 'Pastel1'#sns.color_palette("muted", n_colors = 4)
        df_cluster.plot(column='KMeans cluster labels_k = 4', categorical=True, legend=True, 
                    linewidth=0, axes=ax, cmap = col_map)
        # boundary plot
        df_ETH.geometry.boundary.plot(color = None, edgecolor = 'k', 
                                             linewidth = 0.1, ax = ax)
        ax.set_axis_off()
        plt.axis('equal')
        plt.title('Regionalization with KMeans (K = 4)')
        plt.savefig('Regionalization_KMeans_eThekwini_k = 4_cluster = ' + str(cluster_val) + '.jpg')
    
    
    f, ax = plt.subplots(1, figsize=(9, 9))
    col_map = 'tab10'#sns.color_palette("muted", n_colors = 4)
    df_ETH_point.plot(column='KMeans cluster labels_k = 4', categorical=True, legend=True, 
                linewidth=0, axes=ax, cmap = col_map, markersize = 5)
    # boundary plot
    df_ETH.geometry.boundary.plot(color = None, edgecolor = 'k', 
                                         linewidth = 0.1, ax = ax)
    ax.set_axis_off()
    plt.axis('equal')
    plt.title('Regionalization with KMeans (K = 4)')
    plt.savefig('Regionalization_KMeans_eThekwini_k = 41 ' + '.jpg')
    
    
    '''






