'''
Created on 20 nov. 2025

@author: robert
'''
import platform
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import logging
import unittest
from tabulate import tabulate

from trajectory.FlightList.FlightListReader import FlightListDatabase
from trajectory.Flights.FlightsReader import FlightsDatabase


''' generate parquets with interpolated values '''
''' for all train , rank and final fllight data files '''

def interpolateInternal( flightsData , train_rank_final , flight_id):
    
    df_flightsDataframe  =  flightsData.readOneFlightFileLite(train_rank_final , flight_id )
    if ( df_flightsDataframe.empty == False ):
            
        interpolatedColumnList = ['latitude', 'longitude','altitude','groundspeed','track','vertical_rate', 'mach', 'TAS', 'CAS']
        print(''' ============ count nan before interpolation =================''')
        for column in interpolatedColumnList:
            nan_count_col1 = df_flightsDataframe[column].isna().sum()
            print(f"NaN values in {column}: {nan_count_col1}")
        print("-"*90)
        
        #method = method='polynomial', order=2
        try:
            df_flightsDataframe[interpolatedColumnList] = df_flightsDataframe[interpolatedColumnList] \
                            .interpolate(method='polynomial', order = 2, axis=0, Direction='forward', limit_area="inside" , fill_value='extrapolate')
        except:
            pass
        try:
            df_flightsDataframe[interpolatedColumnList] = df_flightsDataframe[interpolatedColumnList] \
                            .interpolate(method='polynomial', order = 2, axis=0, Direction='backward', limit_area="inside" , fill_value='extrapolate')
        except:
            pass
        try:
            df_flightsDataframe[interpolatedColumnList] = df_flightsDataframe[interpolatedColumnList] \
                            .interpolate(method='polynomial', order = 2, axis=0, Direction='forward', limit_area="outside" , fill_value='extrapolate')
        except:
            pass                
        try:
            df_flightsDataframe[interpolatedColumnList] = df_flightsDataframe[interpolatedColumnList] \
                            .interpolate(method='polynomial', order = 2, axis=0, Direction='backward', limit_area="outside" , fill_value='extrapolate')
        except:
            pass
        ''' 20th november 2025 add TAS and CAS computed from aerocalc '''
            #df_flightsData["TAS_aerocalc"] = df_flightsData
            
        print(''' ============ count nan after interpolation =================''')
        for column in interpolatedColumnList:
            nan_count_col1 = df_flightsDataframe[column].isna().sum()
            print(f"NaN values in {column}: {nan_count_col1}")
                
        ''' generate a parquet file '''
        path = flightsData.getFlightsInterpolatedFolderPathStr(train_rank_final)
        path = os.path.join(path , flight_id + ".parquet")
        print ( path )
        df_flightsDataframe.to_parquet(path, index = False)
        print("-"*90)
        print("-"*90)

def generateInterpolatedFlightDataParquetFiles(train_rank_final):      
        
        assert train_rank_final == "train" or train_rank_final == "rank" or train_rank_final == "final"
        
        flightList = FlightListDatabase()
        if flightList.readTrainRankFinalFlightListLite(train_rank_final):
            logging.info("train rank final flight list read correctly")
            
            df_flightList = flightList.getTrainRankFinalFlightListDataframe(train_rank_final)
            print(tabulate(df_flightList[:10], headers='keys', tablefmt='grid' , showindex=False , ))
                                                
            flightsData = FlightsDatabase()
                                        
            count = 0
            ''' read the flight_id '''
            for index, row in df_flightList.iterrows():
            
                print(f"----- Index: {index} , Name: { row['flight_id'] } ----- ")
                flight_id = row['flight_id']
                print(flight_id)                                    
            
                interpolateInternal( flightsData , train_rank_final , flight_id)

            