# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:19:06 2020

@author: Vijeta
"""

import pandas as pd
import geopandas as gpd
import os
from sklearn import preprocessing
from AuxMethods import AuxMethods as Aux

    #%% data cleaning
def clean_data(path):
    #if False:
    # import file
    #path = r'C:\Users\Vijeta\Documents\Projects\Sizanani\Data'
    
    
    dict_patient_data = {}
    dict_patient_data['location'] = pd.read_excel(os.path.join(path, 'GIS_Combine_20190529all.xlsx'))
    dict_patient_data['clinical trial'] = {"combined": pd.read_sas(os.path.join(path, 'combine_20191224.sas7bdat')),
                "baseline": pd.read_sas(os.path.join(path, 'baseline_20191224.sas7bdat'))}
    
    # location
    df_loc = dict_patient_data['location'].loc[:, ['StudyID', 'Longitude', 'Latitude']]
    columns = ['studyID', 'support', 'CD4Results', 'MHIINDX3', 'death_9mon', 'MHIINDX3_9mon', 'MHIINDX3_diff', 'support_diff', 'death_after9']
    #columns = ['studyID', 'CD4Results', 'NGT_HSPT', 'EMHAPPY', 'EMSAD', 'EMCALM', 'EMNERVOU', 'EMWKSAD', 'support', 'total_health', 'total_barrier', 'HOWFAR', 'MHIINDX3']
    df_cd4 = dict_patient_data['clinical trial']['combined'].loc[:, columns]
    df_cd4 = df_cd4.rename(columns = {'studyID': 'StudyID'})
    df_cd4['support_9mon'] = df_cd4['support'] + df_cd4['support_diff']
    
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
    df_clustering_HIV = df_clustering.loc[:, ['StudyID', 'support', 'CD4Results', 'MHIINDX3', 'Longitude', 'Latitude']].dropna()
    df_clustering_all = df_clustering.loc[:, ['StudyID', 'support', 'MHIINDX3', 'Longitude', 'Latitude']].dropna()
    
    #
    dict_df_clustering['HIV']['Original'] = df_clustering_HIV
    dict_df_clustering['All']['Original'] = df_clustering_all
    del df_clustering_all, df_clustering_HIV
    
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
        # TODO: following line takes looong time. Figure out ways to make it efficient
        dict_geo_df[i] = Aux().pair_point_to_polygon(dict_df_clustering[i]['Standardized'], df_dist_ward)
        dict_geo_df[i]['geometry'] = dict_geo_df[i]['polygon']
    
    #%% EXPLORE VALUES AND DISTRIBUTIONS
    
    # Districts in KwaZulu-Natal
    dist_list = ['Amajuba District', 'Ugu District', 'uMgungundlovu District',
                 'Umzinyathi District', 'Uthukela District', 'Zululand District',
                 'Umkhanyakude District', 'iLembe District', 'eThekwini Metropolitan',
                 'Sisonke District']
    dict_maps['KwaZulu-Natal districts'] = dict_maps['Districts'].loc[dict_maps['Districts'].name.isin(dist_list), :]
    
    
    return dict_maps, dict_geo_df, df_clustering
#%% write excel files
'''
cleaned_data_path = os.path.join(path, 'cleaned data')
if not os.path.exists(cleaned_data_path):
    os.mkdir(cleaned_data_path)
    
# geo datagrames
writer = pd.ExcelWriter(os.path.join(cleaned_data_path, 'dict_geo_df.xlsx'), engine='xlsxwriter')
for key in dict_geo_df:
    dict_geo_df[key].to_excel(writer, sheet_name=key)
writer.save()

# required maps
writer = pd.ExcelWriter(os.path.join(cleaned_data_path, 'dict_maps.xlsx'), engine='xlsxwriter')
for key in dict_maps:
    dict_maps[key].to_excel(writer, sheet_name=key)
writer.save()

# original values
df_clustering.to_csv(os.path.join(cleaned_data_path, 'df_clustering.csv'))


'''

