'''
Created on 25 oct. 2025

@author: robert
'''

import matplotlib.pyplot as plt
import os
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
# Set the option to display all columns
pd.options.display.max_columns = None

import numpy as np 
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

from tabulate import tabulate
from trajectory.Utils.utils import dropUnusedColumns , oneHotEncoderSklearn , getCurrentDateTimeAsStr

''' warning - use tensor flow 2.12.0 not the latest 2.20.0 that is causing DLL problems '''
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler

from trajectory.Fuel.FuelReader import FuelDatabase

import logging
import unittest
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope

from pathlib import Path
from tabulate import tabulate

def scaleDataset( df ):
    
    '''  Apply MinMaxScaler '''
    columnNameListToScale = []
    for columnName in list ( df ):
        columnNameListToScale.append(columnName)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df[columnNameListToScale])
    
    print ( str ( df.shape ))
    return df

''' Root mean square between prediction and actual values '''
def rmse(y_true, y_pred):
    return backend.sqrt( backend.mean (backend.square(y_pred - y_true)))

''' compute fuel kg from fuel flow '''
def computeFuelKg( row ):
    return (abs( row['fuel_flow_kg_sec'] ) * row['time_diff_seconds'])

def capped_value( row , columnName ):
    if row[columnName] > row['upper_bound']:
        return row['upper_bound']
    if row[columnName] < row['lower_bound']:
        return row['lower_bound']
    else:
        return row[columnName]
    
def clean_outliers_capping_with_groupby ( df , groupByColumnName, list_of_columnNames_to_clean ):
    
    #grouped = df.groupby( groupByColumnName , axis = 1)
    for columnName in list_of_columnNames_to_clean:
        
        df['q1'] = df.groupby(groupByColumnName)[columnName].transform('quantile', (0.25))
        df['q3'] = df.groupby(groupByColumnName)[columnName].transform('quantile', (0.75))
        
        df['IQR'] = df['q3'] - df['q1']
        
        #df['lower_bound'] = df['q1'] - 1.5 * df['IQR']
        df['lower_bound'] = df.apply(lambda row: row['q1'] - ( 1.5 * row['IQR']), axis=1)
        
        #df['upper_bound'] = df['q3'] + 1.5 * df['IQR']
        df['upper_bound'] = df.apply(lambda row: row['q3'] + ( 1.5 * row['IQR']), axis=1)
        
        df[columnName] = df.apply( capped_value , axis = 1 , args = [columnName])
        df = dropUnusedColumns( df , ['q1','q3','IQR','lower_bound','upper_bound'])
    return df
    

#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        logging.basicConfig(level=logging.INFO)

        print("tensor flow version = " , tf.__version__)
        
        logging.info (' -------------- Rank Fuel -------------')
        
        # Load the model
        model_file_name = "results_model_2025-10-16-06-46-23.h5"
        model_file_name = "results_model_2025-10-16-06-46-23.h5"
        model_file_name = "results_model_2025-10-16-23-49-37.h5"
        model_file_name = "results_model_2025-10-17-14-33-54.h5"
        model_file_name = "results_model_2025-10-20-18-33-11.h5"
        model_file_name = "results_model_2025-10-20-18-33-11.h5"
        ''' from java parquet file '''
        model_file_name = "results_model_2025-10-25-16-58-04.h5"
        model_file_name = "results_model_2025-10-25-18-08-44.h5"
        model_file_name = "results_model_2025-10-26-11-56-15.h5"
        model_file_name = "results_model_2025-10-27-19-42-31.h5"
        model_file_name = "results_model_2025-10-31-14-50-56-with-outliers.h5"
        model_file_name = "results_model_2025-10-31-14-50-56-with-outliers.h5"
        model_file_name = "results_model_2025-10-31-16-25-19-without-outliers-median.h5"
        model_file_name = "results_model_2025-10-31-17-35-29-without-outliers-capping.h5"
        model_file_name = "results_model_2025-11-01-09-51-00-capped-outliers-groupby-flightId.h5"
        filesFolder = os.path.dirname(__file__)
        filePathModel = os.path.join(filesFolder , model_file_name)
        
        # Save and load a model with the custom activation
        with CustomObjectScope({'rmse': rmse}):
            model = load_model(filePathModel)
            
        ''' first file create from the java features analyser '''
        extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-10-25-17-24-14.parquet"
        extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-10-26-12-04-34.parquet"
        extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-10-27-19-52-33.parquet"
        extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-10-31-17-36-58.parquet"
        filesFolder = "C:/Users/rober/eclipse-2025-09/eclipse-jee-2025-09-R-win32-x86_64/Data-Challenge-2025/documents"
        
        filePath = os.path.join( filesFolder , extendedRankFuelDataFileName)
        file = Path(filePath )
        
        directory = Path(filesFolder)
        if directory.is_dir() and file.is_file():
            
            start_time = time.time()
            
            X_rank = pd.read_parquet ( filePath )
            print( X_rank.shape )
            print ( list (X_rank ))
            
            print ( X_rank.describe().transpose() )
 
            X_rank = dropUnusedColumns(X_rank , ['idx' , 'start' , 'end' ,  'fuel_kg' , 'fuel_flow_kg_sec'])
            X_rank = X_rank.fillna(0.0)
            
            listOfColumnsWithOutliers = ["aircraft_altitude_ft_at_fuel_start","aircraft_altitude_ft_at_fuel_end" , 
                                         "aircraft_vertical_rate_ft_min_at_fuel_start","aircraft_vertical_rate_ft_min_at_fuel_end",
                                         "aircraft_mach_at_fuel_start","aircraft_mach_at_fuel_end",
                                         "aircraft_groundspeed_kt_X_at_fuel_start","aircraft_groundspeed_kt_Y_at_fuel_start",
                                         "aircraft_groundspeed_kt_X_at_fuel_end","aircraft_groundspeed_kt_X_at_fuel_end",
                                         "fuel_burnt_start_relative_to_takeoff_sec","fuel_burnt_end_relative_to_takeoff_sec",
                                         "fuel_burnt_end_relative_to_landed_sec",
                                         "aircraft_vertical_rate_ft_min_at_fuel_start","aircraft_vertical_rate_ft_min_at_fuel_end"]
            
            ''' use groupby flight id to clean outliers '''
            X_rank = clean_outliers_capping_with_groupby( X_rank , 'flight_id' , listOfColumnsWithOutliers)

            ''' drop column flight id '''
            X_rank = dropUnusedColumns(X_rank , ['flight_id'])

            print( str ( X_rank.shape ))
            print(tabulate(X_rank[:10], headers='keys', tablefmt='grid' , showindex=True , ))
            
            print ( list (X_rank ))
            X_rank = scaleDataset( X_rank )

            ''' convert True False to float '''
            X_rank = np.asarray(X_rank).astype(np.float32)
            
            ''' generate predictions '''            #predictions = model.predict(X_rank[np.newaxis, ...])
            predictions = model.predict(X_rank)
            print ( predictions )
            # Convert predictions to a Pandas DataFrame
            y_columnName = 'fuel_flow_kg_sec'
            df_predictions = pd.DataFrame(predictions, columns=[y_columnName])
            print(tabulate(df_predictions[:10], headers='keys', tablefmt='grid' , showindex=True , ))
            
            # Name the index column -> starting zero
            df_predictions.index.name = 'idx'
    
            # Write DataFrame to a CSV file
            filesFolder = os.path.dirname(__file__)
            currentDateTimeAsStr = getCurrentDateTimeAsStr( )
            rankSubmissionfileName = 'fuel_rank_submission' +'_' + currentDateTimeAsStr +'.csv'
            rankSubmissionFilePath = os.path.join(filesFolder , rankSubmissionfileName)
            df_predictions.to_csv(rankSubmissionFilePath, na_rep='N/A', sep=';',  index=True)  
            
            end_time = time.time
            #elapsed_time = end_time - start_time
            #print(f"Elapsed time: {elapsed_time:.2f} seconds")

            
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("tensorflow version = " + tf.__version__)
    print("pandas version = " + pd. __version__)
    
    unittest.main()