'''
Created on 15 oct. 2025

@author: robert

exploring regression with tensor flow
https://www.youtube.com/watch?v=XJFSXH8E6CA 

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
from trajectory.utils import dropUnusedColumns , oneHotEncoderSklearn , getCurrentDateTimeAsStr

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

def prepare_Predictions_Ranking(Count_of_FlightsFiles_to_read):
    
    fuelDatabase = FuelDatabase(Count_of_FlightsFiles_to_read)
    assert fuelDatabase.readFuelRank() == True
    assert fuelDatabase.checkFuelRankHeaders() == True
    assert fuelDatabase.extendFuelRankWithFlightData()

    df = fuelDatabase.getFuelRankDataframe()
    
    print ("final shape = " +  str (  df .shape ) ) 
    
    ''' decision to the use the fuel flow as Y '''
    Y_columnName = 'fuel_flow_kg_sec'
    listOfColumnsToDrop = ['idx'] + [Y_columnName]
    
    print( listOfColumnsToDrop )
    ''' do not put Y value in the train data set '''
    X_rank = dropUnusedColumns ( df , listOfColumnsToDrop)

    ''' check unique names of aircraft type code '''
    aircraft_code_list = X_rank['aircraft_type_code'].unique().tolist()
    print(aircraft_code_list)

    print ( str ( X_rank.shape ))
    #assert X_train.shape[0] == Count_of_FlightsFiles_to_read
    print(tabulate(X_rank[:10], headers='keys', tablefmt='grid' , showindex=True , ))
    
    return X_rank 

def oneHotEncodeXTrainTest(df , columnListToEncode):
    
    for columnName in columnListToEncode:
        df = oneHotEncoderSklearn ( df , columnName)
        
    ''' suppress column with name source_0.0 '''
    df = dropUnusedColumns ( df , ['source_0.0' ,'aircraft_type_code'] )
    #X_train = tf.one_hot( X_train, depth=3 )
    ''' fill not a number with zeros '''
    #df = df.fillna(0)
    
    ''' check if there are null values '''
    print( str ( df.isnull().sum() ))
    return df

#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        logging.basicConfig(level=logging.INFO)

        print(tf.__version__)
        
        logging.info (' -------------- Rank Fuel -------------')
        
        # Load the model
        model_file_name = "model_full_2025-10-16-02-55-27.h5"
        model_file_name = "results_model_2025-10-16-06-46-23.h5"
        filesFolder = os.path.dirname(__file__)
        filePathModel = os.path.join(filesFolder , model_file_name)
        
        # Save and load a model with the custom activation
        with CustomObjectScope({'rmse': rmse}):
            model = load_model(filePathModel)
        
        Count_of_FlightsFiles_to_read = None # get the whole ranking fuel database
        Count_of_FlightsFiles_to_read = 100
        Count_of_FlightsFiles_to_read = 1000
        X_rank = prepare_Predictions_Ranking(Count_of_FlightsFiles_to_read)

        print ( str ( X_rank.shape ))
        
        ''' do not encode only column '''
        ''' one hot encoder for aircraft_type_code  and source '''
        columnListToEncode = [ 'source']
        #X_rank = oneHotEncodeXTrainTest(X_rank , columnListToEncode)
        
        X_rank = dropUnusedColumns ( X_rank , ['fuel_kg','aircraft_type_code','source'] )
        print( str ( X_rank.shape ))
        print(tabulate(X_rank[:10], headers='keys', tablefmt='grid' , showindex=True , ))

        ''' convert True False to float '''
        X_rank = np.asarray(X_rank).astype(np.float32)
        #print(tabulate(X_rank[:10], headers='keys', tablefmt='grid' , showindex=True , ))
        
        ''' generate predictions '''
        #predictions = model.predict(X_rank[np.newaxis, ...])
        predictions = model.predict(X_rank)
        print ( predictions )
        # Convert predictions to a Pandas DataFrame
        y_columnName = 'fuel_flow_kg_sec'
        df_predictions = pd.DataFrame(predictions, columns=[y_columnName])
        print(tabulate(df_predictions[:10], headers='keys', tablefmt='grid' , showindex=True , ))

        # Write DataFrame to a CSV file
        filesFolder = os.path.dirname(__file__)
        rankSubmissionfileName = 'fuel_rank_submission.csv'
        rankSubmissionFilePath = os.path.join(filesFolder , rankSubmissionfileName)
        df_predictions.to_csv(rankSubmissionFilePath, na_rep='N/A', sep=';',  index=True)  # index=False prevents writing row indices


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("tensorflow version = " + tf.__version__)
    print("pandas version = " + pd. __version__)
    
    unittest.main()