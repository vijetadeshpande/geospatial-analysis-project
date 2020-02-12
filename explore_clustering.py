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
from kmodes.kprototypes import KPrototypes as KPro
import scipy as sp

#%% data cleaning

#if False:
# import file
path = r'C:\Users\Vijeta\Documents\Projects\Sizanani\Data'


dict_patient_data = {}
dict_patient_data['location'] = pd.read_excel(os.path.join(path, 'GIS_Combine_20190529all.xlsx'))
dict_patient_data['clinical trial'] = {"combine": pd.read_sas(os.path.join(path, 'combine_20191224.sas7bdat')),
            "baseline": pd.read_sas(os.path.join(path, 'baseline_20191224.sas7bdat'))}

# location
df_loc = dict_patient_data['location'].loc[:, ['StudyID', 'Longitude', 'Latitude']]
columns = ['studyID', 'support', 'CD4Results', 'MHIINDX3']
#columns = ['studyID', 'CD4Results', 'NGT_HSPT', 'EMHAPPY', 'EMSAD', 'EMCALM', 'EMNERVOU', 'EMWKSAD', 'support', 'total_health', 'total_barrier', 'HOWFAR', 'MHIINDX3']
df_cd4 = dict_patient_data['clinical trial']['baseline'].loc[:, columns]
df_cd4 = df_cd4.rename(columns = {'studyID': 'StudyID'})

# dictionaries for storing variables
dict_df_clustering = {'HIV': {'Standardized': None, 'Original': None}, 'All': {'Standardized': None, 'Original': None}}
dict_maps = {'Wards': None, 'Districts': None}

# get map of south africa
map_SA_districts = gpd.read_file(os.path.join(path, 'south-africa.json'))
map_SA_districts.crs = {'init': u'epsg:27700'}
map_SA_wards = gpd.read_file(os.path.join(path, 'try_1.json'))
map_SA_wards.crs = {'init': u'epsg:27700'}
dict_maps['Wards'] = map_SA_wards
dict_maps['Districts'] = map_SA_districts
del map_SA_districts, map_SA_wards

# right join the location df
df_clustering = df_cd4.merge(df_loc, on = 'StudyID', how = 'left')#, lsuffix='_left', rsuffix='_right')
df_clustering_HIV = df_clustering.dropna()
df_clustering_all = df_clustering.loc[:, ['StudyID', 'support', 'MHIINDX3', 'Longitude', 'Latitude']].dropna()

#
dict_df_clustering['HIV']['Original'] = df_clustering_HIV
dict_df_clustering['All']['Original'] = df_clustering_all
del df_clustering_all, df_clustering_HIV, df_clustering

# normalizing values to for creating a common heatmap
#df.loc[:, columns] = preprocessing.normalize(df.loc[:, columns].values)
#df.loc[:, columns] = preprocessing.MinMaxScaler().fit_transform(df.loc[:, columns].values)
dict_df_clustering['HIV']['Standardized'] = dict_df_clustering['HIV']['Original']
dict_df_clustering['HIV']['Standardized'].loc[:, ['support', 'CD4Results', 'MHIINDX3']] = preprocessing.scale(dict_df_clustering['HIV']['Original'].loc[:, ['support', 'CD4Results', 'MHIINDX3']].values)
dict_df_clustering['All']['Standardized'] = dict_df_clustering['All']['Original']
dict_df_clustering['All']['Standardized'].loc[:, ['support', 'MHIINDX3']] = preprocessing.scale(dict_df_clustering['All']['Original'].loc[:, ['support', 'MHIINDX3']].values)


# geopandas object
dict_geo_df = {'HIV': None, 'All': None}
df_dist_ward = dict_maps['Wards'].loc[dict_maps['Wards'].PROVINCE.isin(['KwaZulu-Natal', 'Eastern Cape']), ['CAT_B', 'geometry', 'WARDNO']]
df_dist_ward = df_dist_ward.reset_index(drop = True)
for i in dict_df_clustering:
    df_geo = gpd.GeoDataFrame(dict_df_clustering[i]['Standardized'])
    df_geo['points'] = [Point(xy) for xy in zip(df_geo.Longitude, df_geo.Latitude)]
    df_geo = df_geo.reset_index(drop = True)
    which_row = []
    for row in df_geo.index:
        #print(row)
        paired_point = 0
        row_idx = -1
        for area in df_dist_ward.geometry:
            row_idx += 1
            if df_geo.loc[row, 'points'].within(area):
                df_geo.loc[row, 'geometry'] = area
                df_geo.loc[row, 'District'] = df_dist_ward.loc[row_idx, 'CAT_B']
                df_geo.loc[row, 'ward number'] = df_dist_ward.loc[row_idx, 'WARDNO']
                paired_point = 1
                break
        if paired_point == 0:
            which_row.append(row)
    dict_geo_df[i] = df_geo
    del df_geo
del df_dist_ward

#%% EXPLORE VALUES AND DISTRIBUTIONS
# create folder
if not os.path.exists(os.path.join(path, 'Exploratory plots')):
    os.makedirs(os.path.join(path, 'Exploratory plots'))
path_expl = os.path.join(path, 'Exploratory plots')
f, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 12))
# Make the axes accessible with single indexing
axs = axs.flatten()
# Districts in KwaZulu-Natal
dist_list = ['Amajuba District', 'Ugu District', 'uMgungundlovu District',
             'Umzinyathi District', 'Uthukela District', 'Zululand District',
             'Umkhanyakude District', 'iLembe District', 'eThekwini Metropolitan',
             'Sisonke District']
dict_maps['KwaZulu-Natal districts'] = dict_maps['Districts'].loc[dict_maps['Districts'].name.isin(dist_list), :]

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
    df = df.merge(ward_geo, on = 'ward number', how = 'left')
    
    return df
    
# First we will restrict our analysis to eThekwini
df_clustering_all = dict_geo_df['All'].loc[dict_geo_df['All'].District == 'ETH', :]
df_clustering_all = df_clustering_all.rename(columns = {'MHIINDX3': 'Mental health index', 'support': 'Social support index'})
df_clustering_hiv = dict_geo_df['HIV'].loc[dict_geo_df['HIV'].District == 'ETH', :]
df_clustering_hiv = df_clustering_hiv.rename(columns = {'MHIINDX3': 'Mental health index', 'support': 'Social support index', 'CD4Results': 'CD4 count'})

# clustering for social support and mental health index
df_clustering_all = df_grouping(df_clustering_all, var = ['Social support index', 'Mental health index', 'ward number'])
df_clustering_hiv = df_grouping(df_clustering_hiv, var = ['Social support index', 'Mental health index', 'CD4 count', 'ward number'])


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


#%% More visualization
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






