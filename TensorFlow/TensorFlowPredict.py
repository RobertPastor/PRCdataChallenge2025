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

''' Root mean square between prediction and actual values '''
def rmse(y_true, y_pred):
    return backend.sqrt( backend.mean (backend.square(y_pred - y_true)))

def prepare_X_test(Count_of_FlightsFiles_to_read):
    
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
    X_test = dropUnusedColumns ( df , listOfColumnsToDrop)

    ''' check unique names of aircraft type code '''
    aircraft_code_list = X_test['aircraft_type_code'].unique().tolist()
    print(aircraft_code_list)

    print ( str ( X_test.shape ))
    #assert X_train.shape[0] == Count_of_FlightsFiles_to_read
    print("----- after scaling ----- ")
    print(tabulate(X_test[:10], headers='keys', tablefmt='grid' , showindex=True , ))

    ''' convert True False to float '''
    X_test = np.asarray(X_test).astype(np.float32)
    print(tabulate(X_test[:10], headers='keys', tablefmt='grid' , showindex=True , ))
    return X_test 

#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        logging.basicConfig(level=logging.INFO)

        print(tf.__version__)
        
        logging.info (' -------------- Train Flight list -------------')
        
        # Load the model
        model_file_name = "model_full_2025-10-15-01-20-32.h5"
        filesFolder = os.path.dirname(__file__)
        filePathModel = os.path.join(filesFolder , model_file_name)

        model = load_model(filePathModel)
        
        Count_of_FlightsFiles_to_read = 100
        X_test = prepare_X_test(Count_of_FlightsFiles_to_read)

        print ( str ( X_test.shape ))
        print(tabulate(X_test[:10], headers='keys', tablefmt='grid' , showindex=True , ))

        #loss, accuracy = model.evaluate(X_test, y_test, batch_size=32)
        #print(f"Test Loss: {loss:.4f}")
        #print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("tensorflow version = " + tf.__version__)
    print("pandas version = " + pd. __version__)
    
    unittest.main()