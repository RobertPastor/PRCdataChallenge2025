'''
Created on 12 oct. 2025

@author: robert


exploring regression with tensor flow
https://www.youtube.com/watch?v=XJFSXH8E6CA 
'''

import matplotlib.pyplot as plt

import pandas as pd
import time
import os
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
from sklearn.model_selection import train_test_split
from trajectory.Fuel.FuelReader import FuelDatabase

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope

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

def plot_loss(history , y_limit , currentDateTimeAsString):
    plt.plot(history.history['loss'], label='training_loss')
    plt.plot(history.history['val_loss'], label='validation_loss')
    plt.title("convergence versus epochs")
    plt.ylim([0,y_limit])
    plt.xlabel('Epoch')
    plt.ylabel('Error (fuel_burn_kg / seconds)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file
    plotFileName = 'results_training_loss_vs_validation_loss' + '_'+ currentDateTimeAsString + '.png'
    filesFolder = os.path.dirname(__file__)
    plotFilePath = os.path.join(filesFolder , plotFileName)
 
    plt.savefig(plotFilePath)  # Save as PNG
    
    # Close the plot to free memory
    plt.close()
    #plt.show()

def prepare_X_test(Count_of_FlightsFiles_to_read):
    
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

    print ( str ( X_test.shape ))
    #assert X_train.shape[0] == Count_of_FlightsFiles_to_read
    print("----- after scaling ----- ")
    print(tabulate(X_test[:10], headers='keys', tablefmt='grid' , showindex=True , ))

    ''' convert True False to float '''
    X_test = np.asarray(X_test).astype(np.float32)
    #print(tabulate(X_test[:10], headers='keys', tablefmt='grid' , showindex=True , ))
    return X_test 
    
def prepare_Train_dataset (Count_of_FlightsFiles_to_read ):
    
    fuelDatabase = FuelDatabase(Count_of_FlightsFiles_to_read)
    assert fuelDatabase.readFuelTrain() == True
    assert fuelDatabase.checkFuelTrainHeaders() == True
    assert fuelDatabase.extendFuelTrainWithFlightData()
    
    df = fuelDatabase.getFuelTrainDataframe()
    
    ''' decision to the use the fuel flow as Y '''
    listOfColumnsToDrop = ['idx']
    
    print( listOfColumnsToDrop )
    ''' do not put Y value in the train data set '''
    df = dropUnusedColumns ( df , listOfColumnsToDrop)
    
    ''' return also list of X_Train columns particularly the aircraft code after one hot encoder '''
    return df 

def oneHotEncodeTrainDatase(df , columnListToEncode):
    
    ''' one hot encoder for aircraft_type_code  and source '''
    
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

def scaleDataset( df ):
    
    '''  Apply MinMaxScaler '''
    columnNameListToScale = []
    for columnName in list ( df ):
        columnNameListToScale.append(columnName)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df[columnNameListToScale])
    
    print ( str ( df.shape ))
    
    return df

def tf_model_fit( X_train, y_train, epochs):
    
    ''' declare the model '''
    #tf.random.set_seed ( 42 )
    model = Sequential ( [ Dense( 256 , activation = 'relu' ),
                              Dense( 256 , activation = 'relu' ),
                              Dense( 128 , activation = 'relu' ),
                              Dense(1)])
    
    model.compile(loss = rmse , optimizer = 'adam' , metrics = [rmse])
    history = model.fit( x = X_train , y = y_train , epochs = epochs , validation_split=0.2 , verbose=0)
    
    # Save the entire model to a file
    currentDateTimeAsString = getCurrentDateTimeAsStr()
    modelFileName = "results_model_" +  currentDateTimeAsString + ".h5"

    filesFolder = os.path.dirname(__file__)
    modelFilePath = os.path.join(filesFolder , modelFileName)
    model.save(modelFilePath)  # HDF5 format
    
    plot_loss(history = history , y_limit = 20 , currentDateTimeAsString=currentDateTimeAsString)
    return modelFilePath , currentDateTimeAsString
    

#============================================
class Test_Main(unittest.TestCase):


    def test_a_Train(self):
        
        logging.basicConfig(level=logging.INFO)
        start_time = time.time()
        logging.info (' -------------- Train Fuel database -------------')
        
        Count_of_FlightsFiles_to_read = None
        Count_of_FlightsFiles_to_read = 1000
        #Count_of_FlightsFiles_to_read = None
        epochs = 300
        
        train_dataset  = prepare_Train_dataset(Count_of_FlightsFiles_to_read )
        
        ''' only encode some column '''
        listOfColumnsToEncode = [ 'source']
        #train_dataset = oneHotEncodeTrainDatase(train_dataset , listOfColumnsToEncode)
        
        train_dataset = dropUnusedColumns ( train_dataset , ['fuel_kg','aircraft_type_code','source'] )
        
        print(tabulate(train_dataset[-10:], headers='keys', tablefmt='grid' , showindex=True , ))
        print(tabulate(train_dataset[:10], headers='keys', tablefmt='grid' , showindex=True , ))
        
        y_columnName = 'fuel_flow_kg_sec'
        X = train_dataset.drop( y_columnName , axis = 1)
        
        X = scaleDataset( X )
        print ( str ( list (X) ))
        
        ''' convert True False to float '''
        X = np.asarray(X).astype(np.float32)
        
        #print(tabulate(X[-10:], headers='keys', tablefmt='grid' , showindex=True , ))
        #print(tabulate(X[:10], headers='keys', tablefmt='grid' , showindex=True , ))
        
        y = train_dataset[[y_columnName]]
        print ( str ( list (y) ))
        
        ''' convert True False to float '''
        #Neural Networks and Complex Models: For models like neural networks, 
        #scaling the target variable is often necessary to ensure that the loss function operates within a manageable range.
        y = np.asarray(y).astype(np.float32)

        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        
        # Split the data (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        ''' split data set in 0% train and 20% test '''
        model_file_path , currentDateTimeAsString = tf_model_fit( X_train, y_train , epochs )
        print ( model_file_path )
        
        with CustomObjectScope({'rmse': rmse}):
            model = load_model(model_file_path)
            
        ''' evaluate the model '''
        loss, accuracy = model.evaluate(X_test, y_test)
        #The loss function quantifies the difference between the predicted outputs and the actual target values.
        #It is a continuous value that the model tries to minimize during training.
        # Common loss functions in CNNs include Cross-Entropy Loss for classification tasks and Mean Squared Error (MSE) for regression tasks.
        print(f"Test Loss: {loss}")
        #Accuracy measures the percentage of correct predictions made by the model out of all predictions. It is a discrete metric
        # and is often used to evaluate the model's performance after training. 
        #For example, if a CNN classifies 95 out of 100 test samples correctly, its accuracy is 95%
        print(f"Test Accuracy: {accuracy}")
        
        # Using a context manager to create and write to a file
        accuracyfileName = "results_accuracy_results" + "_" + currentDateTimeAsString + ".txt"
        filesFolder = os.path.dirname(__file__)
        accuracyFilePath = os.path.join(filesFolder , accuracyfileName)
 
        with open(accuracyFilePath, "w") as file:
            file.write(f"Test Loss: {loss}\n")
            file.write(f"Test Accuracy: {accuracy}")
 
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
