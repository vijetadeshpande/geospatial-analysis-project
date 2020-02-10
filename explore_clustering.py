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

#if False:
# import file
path = r'C:\Users\Vijeta\Documents\Projects\Sizanani\Data'


patient_data = {}
patient_data['location'] = pd.read_excel(os.path.join(path, 'GIS_Combine_20190529all.xlsx'))
patient_data['clinical trial'] = {"combine": pd.read_sas(os.path.join(path, 'combine_20191224.sas7bdat')),
            "baseline": pd.read_sas(os.path.join(path, 'baseline_20191224.sas7bdat'))}

# location
df_loc = patient_data['location'].loc[:, ['StudyID', 'Longitude', 'Latitude']]
df_cd4 = patient_data['clinical trial']['baseline'].loc[:, ['studyID', 'CD4Results', 'NGT_HSPT', 'EMHAPPY', 
                     'EMSAD', 'EMCALM', 'EMNERVOU', 'EMWKSAD', 'support', 'total_health',
                     'total_barrier', 'HOWFAR', 'MHIINDX3']]
df_cd4 = df_cd4.rename(columns = {'studyID': 'StudyID'})

# get map of south africa
map_SA_districts = gpd.read_file(os.path.join(path, 'south-africa.json'))
map_SA_districts.crs = {'init': u'epsg:27700'}
map_SA_wards = gpd.read_file(os.path.join(path, 'try_1.json'))
map_SA_wards.crs = {'init': u'epsg:27700'}

# right join the location df
df = df_cd4.merge(df_loc, on = 'StudyID', how = 'left')#, lsuffix='_left', rsuffix='_right')
df = df.dropna()
columns = ['support', 'CD4Results'] #'total_health', 'NGT_HSPT'
columns2 = ['total_health', 'NGT_HSPT', 'support', 'CD4Results', 'total_barrier', 'HOWFAR']
columns3 = ['total_health', 'NGT_HSPT', 'support', 'CD4Results', 'HOWFAR']
columns4 = ['CD4Results', 'HOWFAR', 'total_barrier', 'EMNERVOU']
# normalizing values to for creating a common heatmap
#df.loc[:, columns] = preprocessing.normalize(df.loc[:, columns].values)
#df.loc[:, columns] = preprocessing.MinMaxScaler().fit_transform(df.loc[:, columns].values)
df.loc[:, columns] = preprocessing.scale(df.loc[:, columns].values)


# geopandas object
df_geo = gpd.GeoDataFrame(df)
df_geo['points'] = [Point(xy) for xy in zip(df_geo.Longitude, df_geo.Latitude)]
df_geo = df_geo.reset_index(drop = True)
df_dist_ward = map_SA_wards.loc[map_SA_wards.PROVINCE.isin(['KwaZulu-Natal', 'Eastern Cape']), ['CAT_B', 'geometry', 'WARDNO']]
df_dist_ward = df_dist_ward.reset_index(drop = True)
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

# explore
# create folder
if not os.path.exists(os.path.join(path, 'Exploratory plots')):
    os.makedirs(os.path.join(path, 'Exploratory plots'))
path_expl = os.path.join(path, 'Exploratory plots')
f, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 12))
# Make the axes accessible with single indexing
axs = axs.flatten()
# Districts in KwaZulu-Natal
dist_list = ['Amajuba District', 'Ugu District', 'uMgungundlovu District',
             'Umzinyathi District', 'Uthukela District', 'Zululand District',
             'Umkhanyakude District', 'iLembe District', 'eThekwini Metropolitan',
             'Sisonke District']
map_KwaZulu = map_SA_districts.loc[map_SA_districts.name.isin(dist_list), :]

# Start the loop over all the variables of interest
for i, col in enumerate(columns2):
    # select the axis where the map will go
    ax = axs[i]
    df_geo.plot(column=col, axes=ax, scheme='Quantiles', linewidth=0, colormap='Blues')
    map_KwaZulu.geometry.boundary.plot(color = None, edgecolor = 'k', 
                                           linewidth = 0.5, ax = ax)
    ax.set_axis_off()
    ax.set_title(col)
# save the figure
plt.savefig(os.path.join(path_expl, 'Geospatial exploration.jpg'))

# correlation matrix plot
corr_plot = sns.pairplot(df_geo[columns], kind='reg')#, diag_kind='kde')
corr_plot.savefig(os.path.join(path_expl, 'Correlation exploration.jpg'))

#%% CLUSTERING 

df_kmeans = gpd.GeoDataFrame(df_geo.loc[:, ['support', 'CD4Results', 'ward number']].groupby('ward number').median()).reset_index()
ward_geo = map_SA_wards.loc[map_SA_wards.loc[:, 'MUNICNAME'] == 'Ethekwini Metropolitan Municipality', ['WARDNO', 'geometry']] 
ward_geo['ward number'] = ward_geo['WARDNO']
df_kmeans = df_kmeans.merge(ward_geo, on = 'ward number', how = 'left')

# K-MEANS
KMeans_SA = GSC.KMeans()
KMeans_SA.elbow_test(df_kmeans, columns, np.arange(2, 12), path)
df_KMeans_clusters = KMeans_SA.get_clusters(df_kmeans, columns, [3], map_KwaZulu, path)

# vizualization of clusters in feature space
plt.figure()
sns.scatterplot(data = df_KMeans_clusters, x = 'CD4Results', y = 'support', 
                hue = 'KMeans cluster labels_k = 4')

#%%^

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



#%%



# k prototype
KPro_model = KPro(n_clusters = 3)
df_geo.loc[:, columns4] = preprocessing.normalize(df_geo.loc[:, columns4].values)
KPro_fit = KPro_model.fit(X = df_geo[columns4], categorical = [1, 2, 3])
df_KMeans_clusters['K-prototype cluster labels'] = KPro_fit.labels_

# eThekwini plot
#map_eThekwini = map_KwaZulu.loc[map_KwaZulu.CAT_B == 'ETH', :]
df_ETH = df_KMeans_clusters.loc[df_KMeans_clusters.District == 'ETH', :]
f, ax = plt.subplots(1, figsize=(9, 9))
col_map = 'Pastel1'#sns.color_palette("muted", n_colors = 4)
df_ETH.plot(column='K-prototype cluster labels', categorical=True, legend=True, 
            linewidth=0, axes=ax, cmap = col_map)
# boundary plot
df_ETH.geometry.boundary.plot(color = None, edgecolor = 'k', 
                                     linewidth = 0.7, ax = ax)
ax.set_axis_off()
plt.axis('equal')
plt.title('Regionalization with K-prototype (K = ' + str(3) + ')')
plt.savefig('Kprototype_regionalization_eThekwini.jpg')

to_plot = df_KMeans_clusters.set_index('K-prototype cluster labels')
to_plot = to_plot[columns4]
# Display top of the table
to_plot.head()
to_plot = to_plot.stack()
to_plot.head()
to_plot = to_plot.reset_index()
to_plot.head()
to_plot = to_plot.rename(columns={'level_1': 'Variable', 0: 'Normalized values'})
to_plot.head()
# Setup the facets
facets = sns.FacetGrid(data=to_plot, row='Variable', hue='K-prototype cluster labels', \
                  sharey=False, sharex=False, aspect=2)
# Build the plot as a `sns.kdeplot`
cluster_facet_plot = facets.map(sns.kdeplot, 'Normalized values', shade=True).add_legend()
cluster_facet_plot.savefig('Cluster properties_Kprototype.jpg')


f, ax = plt.subplots(1, figsize=(9, 9))
col_map = 'Pastel1'#sns.color_palette("muted", n_colors = 4)
df_KMeans_clusters.plot(column='K-prototype cluster labels', categorical=True, legend=True, 
        linewidth=0, axes=ax, cmap = col_map)
# boundary plot
map_KwaZulu.geometry.boundary.plot(color = None, edgecolor = 'k', 
                                   linewidth = 0.7, ax = ax)
ax.set_axis_off()
plt.axis('equal')
plt.title('Regionalization with K-prototype (K = 3)')
plt.savefig('Regionalization_Kprototype_KwaZulu_N_k_3.jpg')




# HDBSCAN
HDBSCAN_SA = GSC.HDBSCAN_SA()
df_HDBSCAN_clusters = HDBSCAN_SA.get_clusters(df_geo, columns, np.arange(20,121, 20), map_KwaZulu, path)





