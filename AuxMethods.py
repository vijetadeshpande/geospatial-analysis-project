# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:02:43 2020

@author: Vijeta
"""
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
from copy import deepcopy

class AuxMethods:
    
    def __init__(self):
        
        return
        
    def create_point(self, long, lat):
        
        pts_out = [Point(xy) for xy in zip(long, lat)]
        
        return pts_out
    
    def pair_point_to_polygon(self, df_pts, df_polyg, optional_pairing = True):
        
        if not isinstance(df_pts, pd.DataFrame):
            raise TypeError('Input type should pandas dataframe')
            return
        # avoid mutability
        df_pts, df_polyg = deepcopy(df_pts), deepcopy(df_polyg)
        
        # convert to gpd.df
        df_pts = df_pts.reset_index(drop = True)
        df_pts = gpd.GeoDataFrame(df_pts)
        
        # add column of points calculated from long-lat
        df_pts['points'] = self.create_point(df_pts.Longitude, df_pts.Latitude)
        
        # add column of ward (or any other) geometry
        observations_not_found = []
        for row in df_pts.index:
            paired_point = 0
            row_idx = -1
            for area in df_polyg.geometry:
                row_idx += 1
                if df_pts.loc[row, 'points'].within(area):
                    df_pts.loc[row, 'polygon'] = area
                    if optional_pairing:
                        df_pts.loc[row, 'district'] = df_polyg.loc[row_idx, 'CAT_B']
                        df_pts.loc[row, 'ward number'] = df_polyg.loc[row_idx, 'WARDNO']
                    paired_point = 1
                    break
            if paired_point == 0:
                observations_not_found.append(row)
        
        return df_pts
    