'''
Created on 8 nov. 2025

@author: robert
'''
from minio import Minio
from minio.datatypes import Object
import re

import numpy as np
eps_single = np.finfo(np.float32).eps

from trajectory.utils import dropUnusedColumns , oneHotEncoderSklearn , getCurrentDateTimeAsStr
from pathlib import Path
from tabulate import tabulate

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
from trajectory.utils import dropUnusedColumns , oneHotEncoderSklearn , getCurrentDateTimeAsStr, keepOnlyColumns

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

from pathlib import Path
from tabulate import tabulate

from trajectory.Guidance.GeographicalPointFile import GeographicalPoint
from trajectory.Environment.Constants import Meter2NauticalMiles

class TensorFlowBaseClass(object):
    
    def __init__(self, TrainDataSetRowCount , RankDataSetRowCount, listOfColumnsWithOutliers):
        self.TrainDataSetRowCount = TrainDataSetRowCount
        self.RankDataSetRowCount = RankDataSetRowCount
        self.listOfColumnsWithOutliers =  listOfColumnsWithOutliers
        
    def plot_loss(self, history , y_limit , currentDateTimeAsString):
        
        plt.plot(history.history['loss'], label='training_loss')
        plt.plot(history.history['val_loss'], label='validation_loss')
        plt.title("convergence versus ")
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

    ''' Root mean square between prediction and actual values '''
    def rmse(self, y_true, y_pred):
        return backend.sqrt( backend.mean (backend.square(y_pred - y_true)))
    
    ''' mean absolute error less sensitive to outliers '''
    def meanAbsoluteError(self , y_true , y_pred ):
        return backend.mean ( abs ( y_true - y_pred ))

    ''' used by all activitities '''
    def tf_model_fit( self, X_train, y_train, epochs):
    
        ''' declare the model '''
        #tf.random.set_seed ( 42 )
        model = Sequential ( [ Dense( 256 , activation = 'relu' ),
                                  Dense( 256 , activation = 'relu' ),
                                  Dense( 128 , activation = 'relu' ),
                                  Dense(1)])
        ''' use Mean Absolute Error because it is less sensible to outliers '''
        # 6 Novembre 2025 - use mean_absolute_error
        #model.compile(loss = 'mean_absolute_error' , optimizer = 'adam' , metrics = 'mean_absolute_error')
        model.compile(loss = self.rmse , optimizer = 'adam' , metrics = self.rmse )
        history = model.fit( x = X_train , y = y_train , epochs = epochs , validation_split=0.2, verbose=1)
        
        # Save the entire model to a file
        currentDateTimeAsString = getCurrentDateTimeAsStr()
        modelFileName = "results_model_" +  currentDateTimeAsString + ".h5"

        filesFolder = os.path.dirname(__file__)
        modelFilePath = os.path.join(filesFolder , modelFileName)
        model.save(modelFilePath)  # HDF5 format
        
        self.plot_loss(history = history , y_limit = 0.4 , currentDateTimeAsString=currentDateTimeAsString)
        return modelFilePath , currentDateTimeAsString
        
    ''' possibly use a keras layer '''
    def scaleDataset( self , df ):
    
        '''  Apply MinMaxScaler '''
        columnNameListToScale = []
        for columnName in list ( df ):
            columnNameListToScale.append(columnName)
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        df = scaler.fit_transform(df[columnNameListToScale])
        
        print ( df.shape )
        return df
        
    def clean_outliers_capped(self , df , list_of_columnNames_to_clean):
        for columnName in list_of_columnNames_to_clean:
            Q1 = df[columnName].quantile(0.25)
            Q3 = df[columnName].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df[columnName] = np.clip(df[columnName], lower_bound, upper_bound)
            return df
    
    ''' 6th November 2025 - common method to make predictions from ranking dataframe '''
    def predictFromRankAndModel(self , modelFilePath , X_rank): 
        
        start_time = time.time

        logging.basicConfig(level=logging.INFO)
        logging.info (' -------------- Rank Fuel -------------')
        
        ''' Save and load a model with the custom activation '''
        with CustomObjectScope({'loss' :  self.rmse}, {'rmse': self.rmse}):
            model = load_model(modelFilePath)
            
        listOfColumnsToDrop = [ 'fuel_kg', 'start' , 'end' ,'flight_id','aircraft_type','train_rank', 'aircraft_icao_code']
        X_rank = dropUnusedColumns(X_rank , listOfColumnsToDrop)

        print ( X_rank.isnull().any(axis=1).sum() )
        print ( X_rank.info())
        X_rank = X_rank.fillna(0.0)
        
        ''' DO NOT USE -> do not use groupby flight id to clean outliers '''
        ''' 6th November 2025- not clear if outliers capping is usefull on the ranking dataframe '''
        #X_rank = clean_outliers_capping_with_groupby( X_rank , 'flight_id' , listOfColumnsWithOutliers)
        X_rank = self.clean_outliers_capped( X_rank , self.listOfColumnsWithOutliers)
        assert X_rank.shape[0] == self.RankDataSetRowCount
        
        print ( list (X_rank ))
        X_rank = self.scaleDataset( X_rank )

        ''' convert True False to float '''
        X_rank = np.asarray(X_rank).astype(np.float32)
            
        ''' generate predictions '''            #predictions = model.predict(X_rank[np.newaxis, ...])
        predictions = model.predict(X_rank)
        print ( predictions.shape )
        assert predictions.shape[0] == self.RankDataSetRowCount

        ''' ensure that there no empty values '''
        # Convert predictions to a Pandas DataFrame
        y_columnName = 'fuel_flow_kg_sec'
        df_predictions = pd.DataFrame(predictions, columns=[y_columnName])
        print ( df_predictions.shape )
        assert df_predictions.shape[0] == self.RankDataSetRowCount
        
        print ("number of null values in the predictions = " +  str(df_predictions.isnull().any(axis=1).sum()) )
        ''' empty or N/A submissions are rejected '''
        assert df_predictions.isnull().any(axis=1).sum() == 0

        ''' make all predictions greater than zero '''
        #df_predictions = df_predictions.abs()
        
        #print(tabulate(df_predictions[:10], headers='keys', tablefmt='grid' , showindex=True , ))
        
        assert df_predictions.shape[0] == self.RankDataSetRowCount
        # Name the index column -> starting zero
        df_predictions.index.name = 'idx'
        
        # Write DataFrame to a CSV file
        filesFolder = os.path.dirname(__file__)
        currentDateTimeAsStr = getCurrentDateTimeAsStr( )
        rankSubmissionfileName = 'fuel_rank_submission' +'_' + currentDateTimeAsStr +'.csv'
        rankSubmissionFilePath = os.path.join(filesFolder , rankSubmissionfileName)
        
        df_predictions.to_csv(rankSubmissionFilePath, na_rep='N/A', sep=';',  index=True)  
        
        end_time = time.time
        #time_difference = end_time - start_time
        print(f"Execution Time: {end_time} - {start_time} seconds")
        return rankSubmissionFilePath
    
    ''' call model_fit '''
    def Build_Model_From_Train(self , train_dataset ):
        
        start_time = time.time() 
        assert train_dataset.shape[0] == self.TrainDataSetRowCount
        
        #train_dataset = dropUnusedColumns(train_dataset , ['idx', 'fuel_kg', 'start' , 'end' , 'flight_id'])
        listOfColumnsToDrop = ['idx-train', 'idx-rank', 'fuel_kg', 'start' , 'end' ,'flight_id','aircraft_type','train_rank']
        train_dataset = dropUnusedColumns(train_dataset , listOfColumnsToDrop)
        print ( list (train_dataset))
        
        ''' use clean outliers with capped quantiles without groupby flight_id nor groupby on aircraft code '''
        train_dataset = self.clean_outliers_capped( train_dataset , self.listOfColumnsWithOutliers)
        #train_dataset = self.clean_outliers_capping_with_groupby(train_dataset , 'aircraft_type' , listOfColumnsWithOutliers)
        print ( list (train_dataset))
        train_dataset = train_dataset.fillna(0.0)
        
        y_columnName = 'fuel_flow_kg_sec'
        X = train_dataset.drop( y_columnName , axis = 1)
        ''' check the stats '''
        #print ( tabulate( train_dataset.describe().transpose() , headers='keys', tablefmt='grid' , showindex=True , ))
        
        ''' scale only the dependent variables - there must be only floats or double not categorical columns - nor absolute DateTime '''
        X = self.scaleDataset( X )
        
        ''' pandas series with only one column -> the y '''
        y = train_dataset[[y_columnName]]
        print ( str ( list (y) ))
        y.shape[0] = self.RankDataSetRowCount
        
        ''' whole array must contain only floats no any string '''
        y = np.asarray(y).astype(np.float32)

        #Neural Networks and Complex Models: For models like neural networks, 
        #scaling the target variable is often necessary to ensure that the loss function operates within a manageable range.
        #y = np.asarray(y).astype(np.float32)
        '''  Split the data (70% train, 20% test)'''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=42)
            
        ''' split data set in 0% train and 20% test '''
        ''' after v25 try to reduce epochs to 200 '''
        ''' after v26 -> set epochs again to 300 '''
        epochs = 300
        model_file_path , currentDateTimeAsString = self.tf_model_fit( X_train, y_train , epochs )
        print ( model_file_path )
            
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        
        ''' generate accuracy text file '''
        self.generateAccuracyTextResults(model_file_path , X_test, y_test, currentDateTimeAsString)
        return model_file_path
