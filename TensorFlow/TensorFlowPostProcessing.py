'''
Created on 17 oct. 2025

@author: robert
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
from trajectory.utils import dropUnusedColumns , oneHotEncoderSklearn , getCurrentDateTimeAsStr, keepOnlyColumns

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

''' Root mean square between prediction and actual values '''
def rmse(y_true, y_pred):
    return backend.sqrt( backend.mean (backend.square(y_pred - y_true)))

''' compute flight duration in seconds '''
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

        Count_of_FlightsFiles_to_read = None
        fuelDatabase = FuelDatabase(Count_of_FlightsFiles_to_read)
        assert fuelDatabase.readFuelRank() == True
        
        print ( str (list ( fuelDatabase.getFuelRankDataframe())))
        df_rank = fuelDatabase.getFuelRankDataframe()
        
        listOfColumnsToKeep = ['idx', 'flight_id', 'fuel_burn_start', 'fuel_burn_end','time_diff_seconds']
        df_rank =  keepOnlyColumns ( df_rank , listOfColumnsToKeep )
        print(df_rank.head())
        print(df_rank.shape)
        
        ''' keep only columns as defined in th PRC Challenge web site '''
        # Lecture d'un fichier CSV
        df_predictions = pd.read_csv("fuel_rank_submission.csv", sep=';')

        # Affichage des 5 premières lignes
        print(df_predictions.head())
        print(df_predictions.shape)
        
        # Join on index
        df_result = pd.merge(df_rank, df_predictions, left_index=True, right_index=True)
        #print(df_result.head)
        ''' compute absolute consumption for the time difference '''
        df_result['fuel_kg'] = df_result.apply ( computeFuelKg , axis = 1)
        
        df_result = df_result.rename( columns= {'fuel_burn_start':'start','fuel_burn_end':'end' ,'idx_x':'idx'} )
        df_result = df_result.drop ( ['idx_y', 'fuel_flow_kg_sec' , 'time_diff_seconds' ], axis = 1)
        
        df_result['start_no_utc'] = df_result.apply ( suppressUTC , args = { 'start' }, axis = 1)
        df_result['end_no_utc'] = df_result.apply ( suppressUTC , args = { 'end' }, axis = 1)
        
        df_result = df_result.drop ( ['start', 'end'  ], axis = 1)
        df_result = df_result.rename( columns= {'start_no_utc':'start','end_no_utc':'end' } )
        
        # Rearrange columns
        new_order = ['idx', 'flight_id', 'start', 'end','fuel_kg']
        df_result = df_result[new_order]

        print(tabulate(df_result[-10:], headers='keys', tablefmt='grid' , showindex=False , ))
        print(tabulate(df_result[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        
        ''' write to parquet '''
        df_result.to_parquet('understated-zucchini_v1.parquet')



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("tensorflow version = " + tf.__version__)
    print("pandas version = " + pd. __version__)
    
    unittest.main()