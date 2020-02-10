# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 18:13:21 2020

@author: Vijeta
"""

import pandas as pd
import geopandas as gpd
#import libpysal as lp
import mapclassify as mc
import esda
from matplotlib import colors
import matplotlib.pyplot as plt
#import rasterio as rio
import numpy as np
#import contextily as ctx
import shapely.geometry as geom
from shapely.geometry import Point, Polygon
import seaborn as sns
from copy import deepcopy
import os
from sklearn import preprocessing
import GeoSpatialClustering as GSC
from kmodes.kprototypes import KPrototypes as KPro
from GeoSpatialPlot import GeoSpatialPlot as GSPlot
from SpatialAutocorrelation import SpatialAutocorrelation as SAC
import Mantel
from sklearn.metrics import pairwise_distances as pair_dist
import skgstat

if False: 
    # import file
    path = r'C:\Users\Vijeta\Documents\Projects\Sizanani\Data'
    
    
    patient_data = {}
    patient_data['location'] = pd.read_excel(os.path.join(path, 'GIS_Combine_20190529all.xlsx'))
    patient_data['clinical trial'] = {"combine": pd.read_sas(os.path.join(path, 'combine_20191224.sas7bdat')),
                "baseline": pd.read_sas(os.path.join(path, 'baseline_20191224.sas7bdat'))}
    
    # location
    df_loc = patient_data['location'].loc[:, ['StudyID', 'Longitude', 'Latitude']]
    var_to_study = ['studyID', 'CD4Results', 'support', 'total_health',
                    'total_barrier', 'HOWFAR', 'MHIINDX3']
    df_cd4 = patient_data['clinical trial']['baseline'].loc[:, var_to_study]
    df_cd4 = df_cd4.rename(columns = {'studyID': 'StudyID'})
    
    # get map of south africa
    map_SA_districts = gpd.read_file(os.path.join(path, 'south-africa.json'))
    map_SA_districts.crs = {'init': u'epsg:27700'}
    map_SA_wards = gpd.read_file(os.path.join(path, 'try_1.json'))
    map_SA_wards.crs = {'init': u'epsg:27700'}
    
    # right join the location df
    df = df_cd4.merge(df_loc, on = 'StudyID', how = 'left')#, lsuffix='_left', rsuffix='_right')
    df = df.dropna()
    columns = ['support', 'CD4Results', 'MHIINDX3'] #'total_health', 'NGT_HSPT'
    # normalizing values to for creating a common heatmap
    #df.loc[:, columns] = preprocessing.normalize(df.loc[:, columns].values)
    
    
    # refused to answer
    #refused_to_answer_count = {}
    #for var in columns:
    #    refused_to_answer_count[var] = df.loc[df[var] == 98, var]
    
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
            
    # get dataframes for hotspot analysis
    hotspot_var_to_study = ['CD4Results', 'support', 'MHIINDX3', 'ward number']
    hotspot_df = df_geo.loc[df_geo.District == 'ETH', hotspot_var_to_study]
    hotspot_df['ward number'] = hotspot_df['ward number'].astype(int)
    hotspot_dict = {}
    hotspot_dict['median'] = gpd.GeoDataFrame(hotspot_df.groupby('ward number').median())
    hotspot_dict['mean'] = gpd.GeoDataFrame(hotspot_df.groupby('ward number').mean())
    hotspot_dict['ward polygon pair'] = map_SA_wards.loc[map_SA_wards.MUNICNAME == 'Ethekwini Metropolitan Municipality', ['WARDNO', 'geometry']].sort_values('WARDNO').set_index('WARDNO')
    hotspot_dict['median']['geometry'] = hotspot_dict['ward polygon pair']['geometry']
    hotspot_dict['mean']['geometry'] = hotspot_dict['ward polygon pair']['geometry']
    
    
    # where to save plots
    hotspot_save_path = os.path.join(path, 'Hotspot analysis plots')
    if not os.path.exists(hotspot_save_path):
        os.makedirs(hotspot_save_path)
    
    # plot values for each variable
    for var in hotspot_var_to_study:
        if var == 'ward number':
            continue
        else:        
            GSPlot(var = var, plot_title = 'Median values of ' + var, save_path = hotspot_save_path).plot_map(hotspot_dict['median'])
            GSPlot(var = var, plot_title = 'Mean values of ' + var, save_path = hotspot_save_path).plot_map(hotspot_dict['mean'])
    
    # Autocorrelation
    SAC_objects = {}
    Mantel_test = {}
    sample = gpd.GeoDataFrame(0, index = hotspot_dict['median'].index, columns = hotspot_dict['median'].index)
    distance_matrix = {'geometry': deepcopy(sample), 
                       'feature space': deepcopy(sample)}
    for ward in hotspot_dict['median'].index:
        for var in hotspot_dict['median'].columns:
            if var == 'geometry':
                distance_matrix[var].loc[:, ward] = hotspot_dict['median'][var].centroid.distance(hotspot_dict['median'][var][ward].centroid)
    distance_matrix['feature space'] = pd.DataFrame(pair_dist(hotspot_dict['median'].loc[:, columns]), index = hotspot_dict['median'].index, columns = hotspot_dict['median'].index)


for var in hotspot_var_to_study:
    if var == 'ward number':
        continue
    else:
        # Moran's I
        SAC_objects[var] = SAC(hotspot_dict['median'].loc[:, [var, 'geometry']], var = var, save_path = hotspot_save_path)
        # plot
        SAC_objects[var].plot_null_hypo(var)
        SAC_objects[var].plot_moran_scatter(hotspot_dict['median'][var], var)
        SAC_objects[var].plot_spot_map(hotspot_dict['median'], var)
        
# matel test
columns = ['MHIINDX3', 'support']
distance_matrix['feature space'] = pd.DataFrame(pair_dist(hotspot_dict['median'].loc[:, columns]), index = hotspot_dict['median'].index, columns = hotspot_dict['median'].index)
Mantel_test['veridical correlation'], Mantel_test['p-value'], Mantel_test['z-score'] = Mantel.test(distance_matrix['geometry'].values, distance_matrix['feature space'].values, perms = 10000)


#%%     
# autocorrelation after gridification
# gridify
df_grid = deepcopy(df_geo.loc[df_geo.District == 'ETH', :])
df_grid['ward'] = df_grid.geometry
df_grid['geometry'] = df_grid['points']
xmin,ymin,xmax,ymax =  df_grid.total_bounds
width = 0.005
height = 0.005
rows = int(np.ceil(np.divide((ymax-ymin), height)))
cols = int(np.ceil(np.divide((xmax-xmin), width)))
XleftOrigin = xmin
XrightOrigin = xmin + width
YtopOrigin = ymax
YbottomOrigin = ymax - height
polygons = []
polygon_number = []
p_num = -1
for i in range(cols):
   Ytop = YtopOrigin
   Ybottom =YbottomOrigin
   for j in range(rows):
       p_num += 1
       polygon_number.append(p_num)
       polygons.append(Polygon([(XleftOrigin, Ytop), (XrightOrigin, Ytop), (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)])) 
       Ytop = Ytop - height
       Ybottom = Ybottom - height
   XleftOrigin = XleftOrigin + width
   XrightOrigin = XrightOrigin + width

grid = gpd.GeoDataFrame(0, index = np.arange(0, len(polygons)), columns = ['geometry', 'grid number'])
grid['geometry'] = polygons
grid['grid number'] = polygon_number
#grid['geometry'].boundry.plot()

which_row = []
for row in df_grid.index:
    #print(row)
    paired_point = 0
    row_idx = -1
    for area in grid['geometry']:
        row_idx += 1
        if df_grid.loc[row, 'points'].within(area):
            df_grid.loc[row, 'grid polygon'] = area
            df_grid.loc[row, 'grid number'] = row_idx
            #df_grid.loc[row, 'District'] = df_dist_ward.loc[row_idx, 'CAT_B']
            #df_grid.loc[row, 'ward number'] = df_dist_ward.loc[row_idx, 'WARDNO']
            paired_point = 1
            break
    if paired_point == 0:
        which_row.append(row)

df_grid['geometry'] = df_grid['grid polygon']
df_grid = df_grid.dropna()

#
hotspot_df = deepcopy(df_grid)
hotspot_df['grid number'] = hotspot_df['grid number'].astype(int)
hotspot_dict = {}
hotspot_dict['median'] = gpd.GeoDataFrame(hotspot_df.groupby('grid number').median()).reset_index()
hotspot_dict['mean'] = gpd.GeoDataFrame(hotspot_df.groupby('grid number').mean()).reset_index()
hotspot_dict['ward polygon pair'] = grid
hotspot_dict['median'] = hotspot_dict['median'].merge(hotspot_dict['ward polygon pair'], on = 'grid number', how = 'left')
hotspot_dict['mean'] = hotspot_dict['mean'].merge(hotspot_dict['ward polygon pair'], on = 'grid number', how = 'left')


df_float_join = df_geo.loc[:, ['geometry', 'ward number']]
df_float_join['ward geometry'] = df_float_join.geometry
df_float_join = df_float_join.loc[:, ['ward geometry', 'ward number']]
hotspot_dict['median'] = hotspot_dict['median'].merge(df_float_join, on = 'ward number', how = 'left')
hotspot_dict['mean'] = hotspot_dict['mean'].merge(df_float_join, on = 'ward number', how = 'left')

# auto correlation
SAC_objects_grid = {}

for var in hotspot_var_to_study:
    if var == 'ward number':
        continue
    else:
        SAC_objects_grid[var] = SAC(hotspot_dict['median'].loc[:, [var, 'geometry']], var = var, save_path = hotspot_save_path)
        # plot
        SAC_objects_grid[var].plot_null_hypo(var)
        SAC_objects_grid[var].plot_moran_scatter(hotspot_dict['median'][var], var)
        SAC_objects_grid[var].plot_spot_map(hotspot_dict['median'], var)    
#%%








