'''
Created on 12 oct. 2025

@author: robert


exploring regression with tensor flow
https://www.youtube.com/watch?v=XJFSXH8E6CA 
'''

import matplotlib.pyplot as plt

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

''' Root mean square between prediction and actual values '''
def rmse(y_true, y_pred):
    return backend.sqrt( backend.mean (backend.square(y_pred - y_true)))

transformer = make_column_transformer( 
    (MinMaxScaler(), [ 'time_diff_seconds', 'fuel_flow_kg_sec', 'origin_longitude', 'origin_latitude', 'origin_elevation', 
    'destination_longitude', 'destination_latitude', 'destination_elevation', 'flight_distance_Nm', 'flight_duration_sec', 
    'fuel_burn_relative_start', 'fuel_burn_relative_end',  'latitude', 'longitude', 'altitude', 'groundspeed', 'track', 'vertical_rate', 'mach', 'TAS', 'CAS'] ) ,
    (OneHotEncoder(handle_unknown='ignore') , ['aircraft_type_code' , 'source'] ) )

def plot_loss(history , y_limit):
    plt.plot(history.history['loss'], label='training_loss')
    plt.plot(history.history['val_loss'], label='validation_loss')
    plt.title("convergence versus epochs")
    plt.ylim([0,y_limit])
    plt.xlabel('Epoch')
    plt.ylabel('Error (fuel_burn_kg / seconds)')
    plt.legend()
    plt.grid(True)
    plt.show()

def prepare_Rank(Count_of_FlightsFiles_to_read):
    
    fuelDatabase = FuelDatabase(Count_of_FlightsFiles_to_read)
    assert fuelDatabase.readFuelRank() == True
    assert fuelDatabase.checkFuelRankHeaders() == True
    assert fuelDatabase.extendFuelRankWithFlightData()

    df = fuelDatabase.getFuelRankDataframe()
    
    #print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))

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

    ''' one hot encoder for aircraft_type_code  and source '''
    '''
    columnListToEncode = ['aircraft_type_code' , 'source']
    
    for columnName in columnListToEncode:
        X_test = oneHotEncoderSklearn ( X_test , columnName)
        
    print(list ( X_test ))
  
    X_test = dropUnusedColumns ( X_test , ['source_0.0' ,'aircraft_type_code'] )
    #X_train = tf.one_hot( X_train, depth=3 )
    '''
    ''' fill not a number with zeros '''
    '''
    X_test = X_test.fillna(0)
    
    print(" ---- after one hot encoder ------")
    print(tabulate(X_test[:10], headers='keys', tablefmt='grid' , showindex=True , ))
    '''
    ''' check if there are null values '''
    '''
    print( str ( X_test.isnull().sum() ))
    '''
    '''  Specify columns to rescale '''
    ''' do not rescale fuel_flow_kg_sec as it is the Y currently '''
    '''
    print(list(X_test))
    print(tabulate(X_test[:10], headers='keys', tablefmt='grid' , showindex=True , ))
    '''
    ''' suppress column with name 0.0 '''
    
    '''  Apply MinMaxScaler '''
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_test = scaler.fit_transform(X_test)
    
    print ( str ( X_test.shape ))
    #assert X_train.shape[0] == Count_of_FlightsFiles_to_read
    
    print("----- after scaling ----- ")
    print(tabulate(X_test[:10], headers='keys', tablefmt='grid' , showindex=True , ))
    '''
    ''' convert True False to float '''
    '''
    X_test = np.asarray(X_test).astype(np.float32)
    
    print(tabulate(X_test[:10], headers='keys', tablefmt='grid' , showindex=True , ))
    '''
    return X_test 
    
def prepare_Train(Count_of_FlightsFiles_to_read ):
    
    fuelDatabase = FuelDatabase(Count_of_FlightsFiles_to_read)
    assert fuelDatabase.readFuelTrain() == True
    assert fuelDatabase.checkFuelTrainHeaders() == True
    assert fuelDatabase.extendFuelTrainWithFlightData()
    
    df = fuelDatabase.getFuelTrainDataframe()
    
    ''' decision to the use the fuel flow as Y '''
    Y_columnName = 'fuel_flow_kg_sec'
    listOfColumnsToDrop = ['idx'] + [Y_columnName]
    
    print( listOfColumnsToDrop )
    ''' do not put Y value in the train data set '''
    X_train = dropUnusedColumns ( df , listOfColumnsToDrop)
    
    print ( list ( X_train ) )
    
    ''' You do not need to scale the Y of train data '''
    y_train = df[Y_columnName]
    
    ''' check unique names of aircraft type code '''
    #aircraft_code_list = X_train['aircraft_type_code'].unique().tolist()
    #print(aircraft_code_list)
    
    print(list ( X_train ))
    
    '''  Specify columns to rescale '''
    ''' do not rescale fuel_flow_kg_sec as it is the Y currently '''
    '''
    print( list(X_train) )
    print( tabulate(X_train[:10], headers='keys', tablefmt='grid' , showindex=True , ))
    '''
    ''' to return and to be used in preparation of the X_test with category columns of aircraft type code '''
    '''
    list_of_columns_X_Train = list(X_train)
    
    
    #print(tabulate(X_train[:10], headers='keys', tablefmt='grid' , showindex=True , ))
    '''
    ''' return also list of X_Train columns particularly the aircraft code after one hot encoder '''
    return X_train , y_train 

def oneHotEncodeXTrainTest(df):
    
    ''' one hot encoder for aircraft_type_code  and source '''
    columnListToEncode = ['aircraft_type_code' , 'source']
    
    for columnName in columnListToEncode:
        df = oneHotEncoderSklearn ( df , columnName)
        
    ''' suppress column with name source_0.0 '''
    df = dropUnusedColumns ( df , ['source_0.0' ,'aircraft_type_code'] )
    #X_train = tf.one_hot( X_train, depth=3 )
    ''' fill not a number with zeros '''
    df = df.fillna(0)
    
    ''' check if there are null values '''
    print( str ( df.isnull().sum() ))
    return df

def scaleXTrainXTest( df , columnNameNotToScale ):
    
    '''  Apply MinMaxScaler '''
    columnNameListToScale = []
    for columnName in list ( df ):
        if str(columnName).strip() != str(columnNameNotToScale).strip():
            columnNameListToScale.append(columnName)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df[columnNameListToScale])
    
    print ( str ( df.shape ))
    
    ''' convert True False to float '''
    df = np.asarray(df).astype(np.float32)
    return df

def tf_model_fit( X_train, y_train, epochs):
    
    ''' declare the model '''
    #tf.random.set_seed ( 42 )
    model = Sequential ( [ Dense( 256 , activation = 'relu' ),
                              Dense( 256 , activation = 'relu' ),
                              Dense( 128 , activation = 'relu' ),
                              Dense(1)])
    
    model.compile(loss = rmse , optimizer = 'adam' , metrics = [rmse])
    history = model.fit( x = X_train , y = y_train , epochs = epochs , validation_split=0.2 , verbose=1)
    
    # Save the entire model to a file
    modelFileName = "model_name_" + getCurrentDateTimeAsStr() + ".h5"
    model.save(modelFileName)  # HDF5 format
    
    plot_loss(history = history , y_limit = 20)
    

#============================================
class Test_Main(unittest.TestCase):


    def test_a_Train(self):
        
        logging.basicConfig(level=logging.INFO)
        start_time = time.time()
        logging.info (' -------------- Train Fuel database -------------')
        
        Count_of_FlightsFiles_to_read = 100
        Count_of_FlightsFiles_to_read = 1000
        Count_of_FlightsFiles_to_read = None
        epochs = 200
        
        X_train , y_train  = prepare_Train(Count_of_FlightsFiles_to_read )
        
        X_test = prepare_Rank(Count_of_FlightsFiles_to_read)
        
        list_columns_X_train = list ( X_train )
        list_columns_X_train.sort()
        list_columns_X_test  = list ( X_test )
        list_columns_X_test.sort()
        print ("X train columns = " , str(list_columns_X_train) )
        print ("X test columns = " ,  str(list_columns_X_test) )
        
        print ( X_train.shape )
        print ( X_test.shape )
        
        ''' add column to split again on train or test '''
        columnName_train_test = 'train_or_test'
        X_train[columnName_train_test] = True
        X_test[columnName_train_test] = False
        ''' concat on columns ie vertically '''
        X_result = pd.concat ( [X_train , X_test ], axis=0, ignore_index=True )
        
        print ( X_result.shape )
        list_columns_result = list ( X_result )
        list_columns_result.sort()
        print("result columns = ", str ( list_columns_result))
        
        print(tabulate(X_result[-10:], headers='keys', tablefmt='grid' , showindex=True , ))
        print(tabulate(X_result[:10], headers='keys', tablefmt='grid' , showindex=True , ))
        
        ''' do not encode train_or_test column '''
        X_result = oneHotEncodeXTrainTest(X_result)
        
        print(tabulate(X_result[-10:], headers='keys', tablefmt='grid' , showindex=True , ))
        print(tabulate(X_result[:10], headers='keys', tablefmt='grid' , showindex=True , ))

        ''' split again to keep only a X_Train '''
        X_train = X_result[ X_result[columnName_train_test].isin([True])]
        print( X_train.shape )
        
        X_train = X_train.drop ( columnName_train_test, axis=1)
        print( X_train.shape )
        
        X_train = scaleXTrainXTest(X_train , columnName_train_test )
        
        print(tabulate(X_train[-10:], headers='keys', tablefmt='grid' , showindex=True , ))
        print(tabulate(X_train[:10], headers='keys', tablefmt='grid' , showindex=True , ))

        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
       
        #print(tabulate(X_train[:10], headers='keys', tablefmt='grid' , showindex=True , ))
        
        tf_model_fit( X_train, y_train , epochs)
        
        
    '''
    def test_b_Rank(self):
        
        logging.basicConfig(level=logging.INFO)
        start_time = time.time()
        logging.info (' -------------- Rank Fuel database -------------')
        
        Count_of_FlightsFiles_to_read = 100
        epochs = 200
        X_test  = prepare_Rank(Count_of_FlightsFiles_to_read )
        
        print( str ( X_test.shape ))

        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        
        print(tabulate(X_test[:10], headers='keys', tablefmt='grid' , showindex=True , ))
    '''
    

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    
    print("tensorflow version = " + tf.__version__)
    print("pandas version = " + pd. __version__)
    
    unittest.main()
