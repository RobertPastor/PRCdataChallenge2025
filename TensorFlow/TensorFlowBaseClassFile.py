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
        
        # Save the plotFlightFeatureVersusTime to a file
        plotFileName = 'results_training_loss_vs_validation_loss' + '_'+ currentDateTimeAsString + '.png'
        filesFolder = os.path.dirname(__file__)
        plotFilePath = os.path.join(filesFolder , plotFileName)
        
        plt.savefig(plotFilePath)  # Save as PNG
        # Close the plotFlightFeatureVersusTime to free memory
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
    
    def generateAccuracyTextResults(self , model_file_path , X_test, y_test, currentDateTimeAsString):
        
        # 5 November after v24 - use rmse again
        with CustomObjectScope({'rmse': self.rmse},{'loss':self.rmse}):
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

    
    ''' 6th November 2025 - common method to make predictions from ranking dataframe '''
    def predictFromRankAndModel(self , modelFilePath , X_rank): 
        
        start_time = time.time

        logging.basicConfig(level=logging.INFO)
        logging.info (' -------------- Rank Fuel -------------')
        
        ''' Save and load a model with the custom activation '''
        with CustomObjectScope({'loss' :  self.rmse}, {'rmse': self.rmse}):
            model = load_model(modelFilePath)
            
        listOfColumnsToDrop = ['idx', 'fuel_kg', 'start' , 'end' ,'flight_id','aircraft_type','train_rank', 'aircraft_icao_code']
        X_rank = dropUnusedColumns(X_rank , listOfColumnsToDrop)
        
        #y_columnName = 'fuel_flow_kg_sec'
        listOfColumnsToDrop = ['fuel_flow_kg_sec']
        X_rank = dropUnusedColumns(X_rank , listOfColumnsToDrop)

        print ( X_rank.isnull().any(axis=1).sum() )
        print ( X_rank.info())
        X_rank = X_rank.fillna(0.0)
        
        ''' DO NOT USE -> do not use groupby flight id to clean outliers '''
        ''' 6th November 2025- not clear if outliers capping is usefull on the ranking dataframe '''
        #X_rank = clean_outliers_capping_with_groupby( X_rank , 'flight_id' , listOfColumnsWithOutliers)
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
    
    ''' this operation reduces the count of rows '''
    def clean_with_z_score(self , df , list_of_columnNames_to_clean):
        for columnName in list_of_columnNames_to_clean:
            zscoreColumnName = columnName + "_zscore"
            df[zscoreColumnName] = ( df[columnName] - df[columnName].mean() ) / df[columnName].std()
            
            df = df[(df[zscoreColumnName] < 3.0) & (df[zscoreColumnName] > -3.0)]
            ''' drop the created column '''
            df = df.drop ( zscoreColumnName , axis=1)
            
        return df 
    
    ''' call model_fit '''
    def Build_Model_From_Train(self , train_dataset ):
        
        start_time = time.time() 
        assert train_dataset.shape[0] == self.TrainDataSetRowCount
        
        #train_dataset = dropUnusedColumns(train_dataset , ['idx', 'fuel_kg', 'start' , 'end' , 'flight_id'])
        listOfColumnsToDrop = ['idx','idx-train', 'idx-rank', 'fuel_kg', 'start' , 'end' ,'flight_id','aircraft_type','train_rank']
        train_dataset = dropUnusedColumns(train_dataset , listOfColumnsToDrop)
        print ( list (train_dataset))
        
        ''' clean with zscore -> this operation reduces the number of rows '''
        train_dataset = self.clean_with_z_score ( train_dataset , self.listOfColumnsWithOutliers)
        
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
        ''' zscore cleaning leads to deleted erroneous rows in the train dataset '''
        #assert len(y) == self.RankDataSetRowCount
        
        ''' whole array must contain only floats no any string '''
        y = np.asarray(y).astype(np.float32)

        #Neural Networks and Complex Models: For models like neural networks, 
        #scaling the target variable is often necessary to ensure that the loss function operates within a manageable range.
        #y = np.asarray(y).astype(np.float32)
        '''  Split the data (70% train, 20% test)'''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=42)
            
        ''' split data set in 0% train and 20% test '''
        ''' after v25 try to reduce epochs to 200 '''
        ''' after v26 -> set epochs again to 300 '''
        epochs = 200
        model_file_path , currentDateTimeAsString = self.tf_model_fit( X_train, y_train , epochs )
        print ( model_file_path )
            
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        
        ''' generate accuracy text file '''
        self.generateAccuracyTextResults(model_file_path , X_test, y_test, currentDateTimeAsString)
        return model_file_path    
    
    ''' 6th November 2025 - use in all scenarios '''
    ''' read the S3 storage to extract all team paruet submissions and find the latest submitted version '''
    def getLatestTeamSubmittedVersion(self):
        # create a client
        client = Minio( endpoint = "s3.opensky-network.org" ,
                        access_key = "HertaMoschenPastor" ,
                        secret_key = "HertaMoschenPastor1&&&xxx" ,
                        secure = True)
                        
        
        print("total buckets : " , len ( client.list_buckets() ) )
        for bucket in client.list_buckets():
            print ( bucket.name , bucket.creation_date )
    
        regexp_pattern = r"[.]"
        listOfVersions = []
        for obj in client.list_objects(bucket_name="prc-2025-understated-zucchini", prefix="understated-zucchini"):
            #print ( object.object_name )
            fileName = obj.object_name
            if str(fileName).endswith("parquet"):
                #print ( fileName )
                fileVersion = str(fileName.split("_")[1])
                #print ( fileVersion )
                fileVersion = re.split(regexp_pattern, fileVersion)
                fileVersion = fileVersion[0]
                #print ( fileVersion )
                listOfVersions.append(int(str(fileVersion)[1:]))
            
        listOfVersions.sort()
        #print ( listOfVersions)
        return max ( listOfVersions )

    ''' compute fuel kg from fuel flow '''
    def computeFuelKg( self , row ):
        return (abs( row['fuel_flow_kg_sec'] ) * row['time_diff_seconds'])
    
    def suppressUTC ( self, row , columnName ):
        from datetime import timezone
        return row[columnName].replace(tzinfo=timezone.utc).astimezone(tz=None)
    
    def uploadTeamParquetFileToS3(self ):
        
        # and secret key.
        client = Minio("s3.opensky-network.org",
            access_key="HertaMoschenPastor",
            secret_key="HertaMoschenPastor1&&&xxx",
        )
        # The file to upload, change this path if needed
        filesFolder = os.path.dirname(__file__)
        
        ''' compute file name to upload '''
        newVersionInt = self.getLatestTeamSubmittedVersion()+1
        fileName_to_upload = "understated-zucchini_v" + str(newVersionInt) + ".parquet"

        filePath_to_upload = os.path.join(filesFolder , fileName_to_upload)
    
        # The destination bucket and filename on the MinIO server
        bucket_name = "prc-2025-understated-zucchini"
        
        # Make the bucket if it doesn't exist.
        found = client.bucket_exists(bucket_name)
        if found:
            print("Bucket", bucket_name, "already exists")

        # Upload the file, renaming it in the process
        client.fput_object(
            bucket_name, fileName_to_upload, filePath_to_upload,
        )
        print(
            fileName_to_upload, "successfully uploaded as object",
            fileName_to_upload, "to bucket", bucket_name,
        )

    ''' generate Team submission parquet file '''
    ''' from a CSV input computed before hand ''' 
    def generateTeamSubmissionParquetFile (self , submissionCsvFileName ,  extendedRankFuelDataFileName):
        logging.info (' -------------- Post Processing to convert fuel flow to fuel kg Fuel -------------')
        
        newSubmissionVersion = self.getLatestTeamSubmittedVersion()+1
        targetTeamParquetFileName = 'understated-zucchini_v' + str(newSubmissionVersion) + ".parquet"
        print ( targetTeamParquetFileName )
            
        filePath = os.path.join( self.javaTrainRankfilesFolder  , extendedRankFuelDataFileName)
        file = Path(filePath )
        print( file.absolute())
        directory = Path(self.javaTrainRankfilesFolder)
        if directory.is_dir() and file.is_file():
            
            print("---- start post processing of CSV predictions -- ")
            
            start_time = time.time()
            X_rank = pd.read_parquet ( filePath )
            
            print ("final shape = " +  str (  X_rank .shape ) ) 
            #assert df.shape[0] == fuelDatabase.getFuelRankDataframeNbRows()
            
            ''' list of columns to keep only '''
            listOfColumnsToKeep = ['idx', 'flight_id', 'start', 'end','time_diff_seconds']
            X_rank =  keepOnlyColumns ( X_rank , listOfColumnsToKeep )
             
            print ( str ( X_rank.shape ))
            assert X_rank.shape[0] == self.RankDataSetRowCount
            
            print ( list ( X_rank ))
            #assert X_train.shape[0] == Count_of_FlightsFiles_to_read
            #print(tabulate(X_rank[:10], headers='keys', tablefmt='grid' , showindex=True , ))
            
            print("input CSV file Warning - with fuel flow = " , submissionCsvFileName)
            filesFolder = os.path.dirname(__file__)
            submissionCsvFilePath  = os.path.join(filesFolder , submissionCsvFileName)
            
            if Path(submissionCsvFilePath).exists() and Path(submissionCsvFilePath).is_file():
                ''' read computed submissions '''
                df_predictions = pd.read_csv(submissionCsvFilePath , sep=';')
    
                # Affichage des 5 premières lignes
                #print(df_predictions.head())
                print(df_predictions.shape)
                assert df_predictions.shape[0] == self.RankDataSetRowCount
                ''' ensure that count of NA is 0 '''
                assert pd.DataFrame(df_predictions).shape[0] == pd.DataFrame(df_predictions).dropna().shape[0]
                
                # Join on index idx 
                df_result = pd.merge(X_rank, df_predictions, left_index=True, right_index=True)
                #print(tabulate(df_result[:10], headers='keys', tablefmt='grid' , showindex=True , ))
                
                ''' compute absolute consumption for the time difference '''
                df_result['fuel_kg'] = df_result.apply ( self.computeFuelKg , axis = 1)
                #print(tabulate(df_result[:10], headers='keys', tablefmt='grid' , showindex=True , ))
                
                df_result = df_result.rename( columns= {'fuel_burn_start':'start','fuel_burn_end':'end' ,'idx_x':'idx'} )
                df_result = df_result.drop ( ['idx_y', 'fuel_flow_kg_sec' , 'time_diff_seconds' ], axis = 1)
                #print(tabulate(df_result[:10], headers='keys', tablefmt='grid' , showindex=True , ))
               
                df_result['start_no_utc'] = df_result.apply ( self.suppressUTC , args = { 'start' }, axis = 1)
                df_result['end_no_utc'] = df_result.apply ( self.suppressUTC , args = { 'end' }, axis = 1)
                #print(tabulate(df_result[:10], headers='keys', tablefmt='grid' , showindex=True , ))
                
                df_result = df_result.drop ( ['start', 'end'  ], axis = 1)
                df_result = df_result.rename( columns= {'start_no_utc':'start','end_no_utc':'end' } )
                #print(tabulate(df_result[:10], headers='keys', tablefmt='grid' , showindex=True , ))
               
                # Rearrange columns order 
                new_order = ['idx', 'flight_id', 'start', 'end','fuel_kg']
                df_result = df_result[new_order]
                
                #print(tabulate(df_result[-10:], headers='keys', tablefmt='grid' , showindex=False , ))
                #print(tabulate(df_result[:10], headers='keys', tablefmt='grid' , showindex=False , ))
                
                ''' write to parquet '''
                
                filesFolder = os.path.dirname(__file__)
                targetTeamParquetFilePath = os.path.join(filesFolder , targetTeamParquetFileName)
                
                print("final submission parquet file = " + targetTeamParquetFilePath)
                df_result.to_parquet(targetTeamParquetFileName)
                
                end_time = time.time()  # Record the end time
                elapsed_time = end_time - start_time
                print(f"Elapsed time: {elapsed_time:.2f} seconds")
                
                print(" submission parquet file <<" + targetTeamParquetFilePath +">> generated correctly")
