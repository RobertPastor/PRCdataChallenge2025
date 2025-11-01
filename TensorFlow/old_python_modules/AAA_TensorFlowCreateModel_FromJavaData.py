'''
Created on 25 oct. 2025

@author: robert
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

from pathlib import Path
from tabulate import tabulate

''' Root mean square between prediction and actual values '''
def rmse(y_true, y_pred):
    return backend.sqrt( backend.mean (backend.square(y_pred - y_true)))

def scaleDataset( df ):
    
    '''  Apply MinMaxScaler '''
    columnNameListToScale = []
    for columnName in list ( df ):
        columnNameListToScale.append(columnName)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df[columnNameListToScale])
    
    print ( str ( df.shape ))
    return df

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

def tf_model_fit( X_train, y_train, epochs):
    
    
    print(tabulate(X_train[:10], headers='keys', tablefmt='grid' , showindex=True , ))
    print ( X_train.shape )
    print(tabulate(y_train[:10], headers='keys', tablefmt='grid' , showindex=True , ))
    print ( y_train.shape )

    ''' declare the model '''
    #tf.random.set_seed ( 42 )
    model = Sequential ( [ Dense( 256 , activation = 'relu' ),
                              Dense( 256 , activation = 'relu' ),
                              Dense( 128 , activation = 'relu' ),
                              Dense(1)])
    
    model.compile(loss = rmse , optimizer = 'adam' , metrics = [rmse])
    history = model.fit( x = X_train , y = y_train , epochs = epochs , validation_split=0.2 , verbose=1)
    
    # Save the entire model to a file
    currentDateTimeAsString = getCurrentDateTimeAsStr()
    modelFileName = "results_model_" +  currentDateTimeAsString + ".h5"

    filesFolder = os.path.dirname(__file__)
    modelFilePath = os.path.join(filesFolder , modelFileName)
    model.save(modelFilePath)  # HDF5 format
    
    plot_loss(history = history , y_limit = 0.6 , currentDateTimeAsString=currentDateTimeAsString)
    return modelFilePath , currentDateTimeAsString


def clean_outliers_with_median( df , list_of_columnNames_to_clean ):
    for columnName in list_of_columnNames_to_clean:
        Q1 = df[columnName].quantile(0.25)
        Q3 = df[columnName].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df[columnName] = np.where((df[columnName] < lower_bound) | (df[columnName] > upper_bound), df[columnName].median(), df[columnName])
        #print("Data after Replacing Outliers:\n", df)
        
        return df
    
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


    def test_a_Train(self):
        
        extendedFuelTrainDataFileName = "ExtendedFuel_train_2025-10-25-16-29-19.parquet"
        extendedFuelTrainDataFileName = "ExtendedFuel_train_2025-10-26-10-44-58.parquet"
        extendedFuelTrainDataFileName = "ExtendedFuel_train_2025-10-27-18-08-12.parquet"
        extendedFuelTrainDataFileName = "ExtendedFuel_train_2025-10-31-12-44-23.parquet"
        filesFolder = "C:/Users/rober/eclipse-2025-09/eclipse-jee-2025-09-R-win32-x86_64/Data-Challenge-2025/documents"
        
        filePath = os.path.join( filesFolder , extendedFuelTrainDataFileName)
        file = Path(filePath )
        
        directory = Path(filesFolder)
        if directory.is_dir() and file.is_file():
            
            start_time = time.time()
            
            train_dataset = pd.read_parquet ( filePath )
            print( train_dataset.shape )
            print ( list (train_dataset))
            
            #print ( train_dataset.describe().transpose() )
 
            #train_dataset = dropUnusedColumns(train_dataset , ['idx', 'fuel_kg', 'start' , 'end' , 'flight_id'])
            train_dataset = dropUnusedColumns(train_dataset , ['idx', 'fuel_kg', 'start' , 'end' ])
            train_dataset = train_dataset.fillna(0.0)
                        
            ''' clean outliers '''
            listOfColumnsWithOutliers = ["aircraft_altitude_ft_at_fuel_start","aircraft_altitude_ft_at_fuel_end" , 
                                         "aircraft_vertical_rate_ft_min_at_fuel_start","aircraft_vertical_rate_ft_min_at_fuel_end",
                                         "aircraft_mach_at_fuel_start","aircraft_mach_at_fuel_end",
                                         "aircraft_groundspeed_kt_X_at_fuel_start","aircraft_groundspeed_kt_Y_at_fuel_start",
                                         "aircraft_groundspeed_kt_X_at_fuel_end","aircraft_groundspeed_kt_X_at_fuel_end",
                                         "fuel_burnt_start_relative_to_takeoff_sec","fuel_burnt_end_relative_to_takeoff_sec",
                                         "fuel_burnt_end_relative_to_landed_sec",
                                         "aircraft_vertical_rate_ft_min_at_fuel_start","aircraft_vertical_rate_ft_min_at_fuel_end"]
            
            ''' use groupby flight id to clean outliers '''
            train_dataset = clean_outliers_capping_with_groupby( train_dataset , 'flight_id' , listOfColumnsWithOutliers)
            print ( list (train_dataset))
            
            print ( tabulate( train_dataset.describe().transpose() , headers='keys', tablefmt='grid' , showindex=True , ))
            
            ''' drop column flight id '''
            train_dataset = dropUnusedColumns(train_dataset , ['flight_id'])
            print( list ( train_dataset ))
            print(tabulate(train_dataset[-10:], headers='keys', tablefmt='grid' , showindex=True , ))
            print(tabulate(train_dataset[:10], headers='keys', tablefmt='grid' , showindex=True , ))

            ''' do not scale the independent variable Y '''
            y_columnName = 'fuel_flow_kg_sec'
            X = train_dataset.drop( y_columnName , axis = 1)
            ''' scale only the dependent variables  '''
            X = scaleDataset( X )
            #print ( str ( list (X) ))
            
            #print(tabulate(X[-10:], headers='keys', tablefmt='grid' , showindex=True , ))
            #print(tabulate(X[:10], headers='keys', tablefmt='grid' , showindex=True , ))
            ''' pandas series with only one column -> the y '''
            y = train_dataset[[y_columnName]]
            print ( str ( list (y) ))
            ''' whole array must contain only floats no any string '''
            y = np.asarray(y).astype(np.float32)

            ''' convert True False to float '''
            #Neural Networks and Complex Models: For models like neural networks, 
            #scaling the target variable is often necessary to ensure that the loss function operates within a manageable range.
            #y = np.asarray(y).astype(np.float32)
            '''  Split the data (70% train, 20% test)'''
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            ''' split data set in 0% train and 20% test '''
            epochs = 150
            model_file_path , currentDateTimeAsString = tf_model_fit( X_train, y_train , epochs )
            print ( model_file_path )
            
            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            
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


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    
    print("tensorflow version = " + tf.__version__)
    print("pandas version = " + pd. __version__)
    
    unittest.main()
        