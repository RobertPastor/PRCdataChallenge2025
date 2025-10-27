'''
Created on 26 oct. 2025

@author: rober
'''

import matplotlib.pyplot as plt
import os
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
# Set the option to display all columns
pd.options.display.max_columns = None
from datetime import datetime, timezone

import numpy as np 
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

from tabulate import tabulate
from trajectory.utils import dropUnusedColumns , oneHotEncoderSklearn , getCurrentDateTimeAsStr , keepOnlyColumns

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

''' Root mean square between prediction and actual values '''
def rmse(y_true, y_pred):
    return backend.sqrt( backend.mean (backend.square(y_pred - y_true)))

''' compute fuel kg from fuel flow '''
def computeFuelKg( row ):
    return (abs( row['fuel_flow_kg_sec'] ) * row['time_diff_seconds'])

def suppressUTC ( row , columnName ):
    return row[columnName].replace(tzinfo=timezone.utc).astimezone(tz=None)

#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        logging.basicConfig(level=logging.INFO)

        print("tensor flow version = " , tf.__version__)
        
        logging.info (' -------------- Post Processing to convert fuel flow to fuel kg Fuel -------------')
    
        extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-10-26-12-04-34.parquet"
        filesFolder = "C:/Users/rober/eclipse-2025-09/eclipse-jee-2025-09-R-win32-x86_64/Data-Challenge-2025/documents"
            
        filePath = os.path.join( filesFolder , extendedRankFuelDataFileName)
        file = Path(filePath )
        print( file.absolute())
        directory = Path(filesFolder)
        if directory.is_dir() and file.is_file():
            
            print("---- start predictions post processing -- ")
            
            start_time = time.time()
                
            X_rank = pd.read_parquet ( filePath )
            
            print ("final shape = " +  str (  X_rank .shape ) ) 
            #assert df.shape[0] == fuelDatabase.getFuelRankDataframeNbRows()
            
            ''' list of columns to keep only '''
            listOfColumnsToKeep = ['idx', 'flight_id', 'start', 'end','time_diff_seconds']
            X_rank =  keepOnlyColumns ( X_rank , listOfColumnsToKeep )
             
            print ( str ( X_rank.shape ))
            print ( list ( X_rank ))
            #assert X_train.shape[0] == Count_of_FlightsFiles_to_read
            print(tabulate(X_rank[:10], headers='keys', tablefmt='grid' , showindex=True , ))
        
            submissionCsvFile = "fuel_rank_submission_2025-10-21-02-22-14.csv"
            submissionCsvFile = "fuel_rank_submission_2025-10-26-12-14-25.csv"
            print("input CSV file Warning - with fuel flow = " , submissionCsvFile)
            df_predictions = pd.read_csv(submissionCsvFile , sep=';')

            # Affichage des 5 premières lignes
            print(df_predictions.head())
            print(df_predictions.shape)
            
            # Join on index
            df_result = pd.merge(X_rank, df_predictions, left_index=True, right_index=True)
            print(tabulate(df_result[:10], headers='keys', tablefmt='grid' , showindex=True , ))
            
            ''' compute absolute consumption for the time difference '''
            df_result['fuel_kg'] = df_result.apply ( computeFuelKg , axis = 1)
            print(tabulate(df_result[:10], headers='keys', tablefmt='grid' , showindex=True , ))
            
            df_result = df_result.rename( columns= {'fuel_burn_start':'start','fuel_burn_end':'end' ,'idx_x':'idx'} )
            df_result = df_result.drop ( ['idx_y', 'fuel_flow_kg_sec' , 'time_diff_seconds' ], axis = 1)
            print(tabulate(df_result[:10], headers='keys', tablefmt='grid' , showindex=True , ))
           
            df_result['start_no_utc'] = df_result.apply ( suppressUTC , args = { 'start' }, axis = 1)
            df_result['end_no_utc'] = df_result.apply ( suppressUTC , args = { 'end' }, axis = 1)
            print(tabulate(df_result[:10], headers='keys', tablefmt='grid' , showindex=True , ))
            
            df_result = df_result.drop ( ['start', 'end'  ], axis = 1)
            df_result = df_result.rename( columns= {'start_no_utc':'start','end_no_utc':'end' } )
            print(tabulate(df_result[:10], headers='keys', tablefmt='grid' , showindex=True , ))
           
            # Rearrange columns order 
            new_order = ['idx', 'flight_id', 'start', 'end','fuel_kg']
            df_result = df_result[new_order]
    
            print(tabulate(df_result[-10:], headers='keys', tablefmt='grid' , showindex=False , ))
            print(tabulate(df_result[:10], headers='keys', tablefmt='grid' , showindex=False , ))
            
            ''' write to parquet '''
            #df_result.to_parquet('understated-zucchini_v1.parquet')
            #df_result.to_parquet('understated-zucchini_v2.parquet')
            #targetTeamParquetFileName = 'understated-zucchini_v3.parquet'
            targetTeamParquetFileName = 'understated-zucchini_v5.parquet'
            targetTeamParquetFileName = 'understated-zucchini_v6.parquet'
            print("final surmission parquet file = " + targetTeamParquetFileName)
            df_result.to_parquet(targetTeamParquetFileName)
            
            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.2f} seconds")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("tensorflow version = " + tf.__version__)
    print("pandas version = " + pd. __version__)
    
    unittest.main()