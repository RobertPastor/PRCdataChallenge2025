'''
Created on 1 nov. 2025

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

TrainDataSetRowCount = 131530
RankDataSetRowCount = 24289

'''
#listOfColumnsWithOutliers = ["aircraft_altitude_ft_at_fuel_start","aircraft_altitude_ft_at_fuel_end" , 
#                             "aircraft_vertical_rate_ft_min_at_fuel_start","aircraft_vertical_rate_ft_min_at_fuel_end",

#                             "aircraft_mach_at_fuel_start","aircraft_mach_at_fuel_end",

#                             "aircraft_groundspeed_kt_X_at_fuel_start","aircraft_groundspeed_kt_Y_at_fuel_start",
#                             "aircraft_groundspeed_kt_X_at_fuel_end","aircraft_groundspeed_kt_X_at_fuel_end",

#                             "fuel_burnt_start_relative_to_takeoff_sec","fuel_burnt_end_relative_to_takeoff_sec",
#                             "fuel_burnt_end_relative_to_landed_sec",
#                             "aircraft_vertical_rate_ft_min_at_fuel_start","aircraft_vertical_rate_ft_min_at_fuel_end"]

'''
''' clean outliers '''
listOfColumnsWithOutliers = ["aircraft_altitude_ft_at_fuel_start","aircraft_altitude_ft_at_fuel_end" , 
                            "aircraft_vertical_rate_ft_min_at_fuel_start","aircraft_vertical_rate_ft_min_at_fuel_end",
                            "aircraft_computed_vertical_rate_ft_min",
                            "aircraft_mach_at_fuel_start","aircraft_mach_at_fuel_end",
                            "aircraft_TAS_at_fuel_start","aircraft_TAS_at_fuel_end",
                            "aircraft_CAS_at_fuel_start","aircraft_CAS_at_fuel_end",
                            "fuel_burnt_start_relative_to_takeoff_sec","fuel_burnt_end_relative_to_takeoff_sec",
                            "fuel_burnt_end_relative_to_landed_sec",
                            "aircraft_vertical_rate_ft_min_at_fuel_start","aircraft_vertical_rate_ft_min_at_fuel_end"]


class PRCdataChallenge2025Submissions:
    """Exemple de classe simple"""
    
    def __init__(self , extendedFuelTrainDataFileName , extendedRankFuelDataFileName , javaTrainRankfilesFolder):
        self.extendedFuelTrainDataFileName = extendedFuelTrainDataFileName
        self.extendedRankFuelDataFileName = extendedRankFuelDataFileName
        self.javaTrainRankfilesFolder = javaTrainRankfilesFolder
        
    def capped_value( self, row , columnName ):
        if row[columnName] > row['upper_bound']:
            return row['upper_bound']
        if row[columnName] < row['lower_bound']:
            return row['lower_bound']
        else:
            return row[columnName]
        
    def clean_outliers_capped(self , df , list_of_columnNames_to_clean):
        for columnName in list_of_columnNames_to_clean:
            Q1 = df[columnName].quantile(0.25)
            Q3 = df[columnName].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df[columnName] = np.clip(df[columnName], lower_bound, upper_bound)
            return df
    ''' do not use groupby '''
    def clean_outliers_capping_with_groupby ( self,  df , groupByColumnName, list_of_columnNames_to_clean ):
    
    #grouped = df.groupby( groupByColumnName , axis = 1)
        for columnName in list_of_columnNames_to_clean:
            
            df['q1'] = df.groupby(groupByColumnName)[columnName].transform('quantile', (0.25))
            df['q3'] = df.groupby(groupByColumnName)[columnName].transform('quantile', (0.75))
            
            df['IQR'] = df['q3'] - df['q1']
            
            #df['lower_bound'] = df['q1'] - 1.5 * df['IQR']
            df['lower_bound'] = df.apply(lambda row: row['q1'] - ( 1.5 * row['IQR']), axis=1)
            
            #df['upper_bound'] = df['q3'] + 1.5 * df['IQR']
            df['upper_bound'] = df.apply(lambda row: row['q3'] + ( 1.5 * row['IQR']), axis=1)
            
            df[columnName] = df.apply( self.capped_value , axis = 1 , args = [columnName])
            df = dropUnusedColumns( df , ['q1','q3','IQR','lower_bound','upper_bound'])
        return df
    
    def cappedOutliersGroupedByFlightId(self  , df):
        
        ''' use groupby flight id to clean outliers '''
        df = self.clean_outliers_capping_with_groupby( df , 'flight_id' , listOfColumnsWithOutliers)
        print ( list (df))
            
        #print ( tabulate( df.describe().transpose() , headers='keys', tablefmt='grid' , showindex=True , ))
            
        ''' drop column flight id '''
        df = dropUnusedColumns(df , ['flight_id'])
        print( list ( df ))
        return df
 
    def getLatestUploadedSubmission(self):
        
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
            # sort returns nothing 
            listOfVersions.sort()
            #print ( listOfVersions)        
            print( " max submitted proposal = "+ max(listOfVersions))   
            
    ''' possibly use a keras layer '''
    def scaleDataset( self , df ):
    
        '''  Apply MinMaxScaler '''
        columnNameListToScale = []
        for columnName in list ( df ):
            columnNameListToScale.append(columnName)
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        df = scaler.fit_transform(df[columnNameListToScale])
        
        print ( str ( df.shape ))
        return df
                                              
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
        history = model.fit( x = X_train , y = y_train , epochs = epochs , validation_split=0.3, verbose=1)
        
        # Save the entire model to a file
        currentDateTimeAsString = getCurrentDateTimeAsStr()
        modelFileName = "results_model_" +  currentDateTimeAsString + ".h5"

        filesFolder = os.path.dirname(__file__)
        modelFilePath = os.path.join(filesFolder , modelFileName)
        model.save(modelFilePath)  # HDF5 format
        
        self.plot_loss(history = history , y_limit = 0.4 , currentDateTimeAsString=currentDateTimeAsString)
        return modelFilePath , currentDateTimeAsString
    
    def getFlightListMergedWithAircrafts(self , train_rank_str ):
        from trajectory.FlightList.FlightListReader import FlightListDatabase
        flightListDatabase = FlightListDatabase()
        
        if train_rank_str == 'train':
            ''' read flight list '''
            flightListDatabase.readTrainFlightListLite()
            assert flightListDatabase.extendTrainFlightListWithAircraftData() == True
            return flightListDatabase.getTrainFlightListDataframe()
        else:
            flightListDatabase.readRankFlightListLite()
            assert flightListDatabase.extendRankFlightListWithAircraftData() == True
            return flightListDatabase.getRankFlightListDataframe()

    def getFlightListMergedWithAirports(self , train_rank_str ):
        from trajectory.FlightList.FlightListReader import FlightListDatabase
        flightListDatabase = FlightListDatabase()

        if train_rank_str == 'train':
            ''' read flight list '''
            flightListDatabase.readTrainFlightListLite()
            assert flightListDatabase.extendTrainFlightListWithAirportData()
            return flightListDatabase.getTrainFlightListDataframe()
        
        else:
            ''' read flight list '''
            flightListDatabase.readRankFlightListLite()
            assert flightListDatabase.extendRankFlightListWithAirportData()
            return flightListDatabase.getRankFlightListDataframe()
            
    ''' compute distance between departure and arrival airport using great circle '''
    def computeFlightDistanceNauticalMiles( self  , row , 
                                            origin_latitude_deg_columnName , origin_longitude_deg_columnName , 
                                            origin_elevation_meters_columnName , destination_latitude_deg_columnName , 
                                            destination_longitude_deg_columnName , destination_elevation_meters_columnName):
        import sys
        computedDistanceNm = 0.0
        epsilon = sys.float_info.epsilon
        try:
            originLatitudeDeg = max ( float(row[origin_latitude_deg_columnName]) , float(-90.0) + epsilon)
            originLatitudeDeg = min ( originLatitudeDeg , float (+90.0) - epsilon)

            destinationLatitudeDeg = max ( float (row[destination_latitude_deg_columnName]) , float(-90.0) + epsilon)
            destinationLatitudeDeg = min ( destinationLatitudeDeg , float(+90.0) - epsilon)

            ''' 2nd November 2025 - airport ZGOW "Jieyang Chaoshan International Airport" has no elevation_ft value -> it is missing '''
            ''' do not use elevation to compute haversine distance '''
            originGeoPoint = GeographicalPoint ( LatitudeDegrees            = originLatitudeDeg, 
                                                 LongitudeDegrees           = float(row[origin_longitude_deg_columnName]),
                                                 AltitudeMeanSeaLevelMeters = float(0.0))
            
            destinationGeoPoint = GeographicalPoint ( LatitudeDegrees              = destinationLatitudeDeg, 
                                                      LongitudeDegrees             = float(row[destination_longitude_deg_columnName]),
                                                      AltitudeMeanSeaLevelMeters   = float(0.0))
        
            computedDistanceNm = abs(originGeoPoint.computeDistanceMetersTo(destinationGeoPoint) * Meter2NauticalMiles)
        except AssertionError as e:
            raise ValueError ( row['flight_id'] , row[origin_latitude_deg_columnName] , row[destination_latitude_deg_columnName])
            computedDistanceNm = 0.0
        return computedDistanceNm
        
    def computeDistanceBetweenOriginAirportAndAircraftPosition(self , df):
        
        ''' compute haversine distance between origin airport and fuel start aircraft position '''
        df['aircraft_distance_origin_to_fuel_start_Nm'] = df.apply ( self.computeFlightDistanceNauticalMiles , axis = 1 ,
                                                                                               
                 args=('origin_latitude_deg','origin_longitude_deg','origin_elevation_ft',
                              'aircraft_latitude_deg_at_fuel_start','aircraft_longitude_deg_at_fuel_start', 'origin_elevation_ft' ))
            
        ''' compute haversine distance between origin airport and fuel end '''
        df['aircraft_distance_origin_to_fuel_end_Nm'] = df.apply ( self.computeFlightDistanceNauticalMiles , axis = 1 ,
                                                                                             
                args=('origin_latitude_deg','origin_longitude_deg', 'origin_elevation_ft',
                              'aircraft_latitude_deg_at_fuel_end','aircraft_longitude_deg_at_fuel_end','origin_elevation_ft'))
            
        df['aircraft_distance_fuel_start_to_destination_Nm'] = df.apply ( self.computeFlightDistanceNauticalMiles , axis = 1,
                                                                                                  
                        args=('aircraft_latitude_deg_at_fuel_start','aircraft_longitude_deg_at_fuel_start', 'destination_elevation_ft' ,
                              'destination_latitude_deg', 'destination_longitude_deg','destination_elevation_ft'))
            
        df['aircraft_distance_fuel_end_to_destination_Nm'] = df.apply ( self.computeFlightDistanceNauticalMiles , axis = 1,
                                                                                                  
                        args=('aircraft_latitude_deg_at_fuel_end','aircraft_longitude_deg_at_fuel_end', 'destination_elevation_ft' ,
                              'destination_latitude_deg', 'destination_longitude_deg','destination_elevation_ft'))
        return df
    
    def compute_TAS_KnotsfromMach_atFuelStart(self, row ):
        from trajectory.aerocalc.airspeed import mach2tas 
        
        mach = row['aircraft_mach_at_fuel_start']
        aircraft_altitude_ft = row['aircraft_altitude_ft_at_fuel_start']
        return mach2tas ( mach=mach,temp='std',altitude=aircraft_altitude_ft) 
        '''
        #mach = max( row['aircraft_mach_at_fuel_start'], eps_single)
        #if ( row['aircraft_TAS_at_fuel_start'] <= eps_single ) and (mach > 0.0 ):
        #    aircraft_altitude_ft = row['aircraft_altitude_ft_at_fuel_start']
        #    return mach2tas ( mach=mach,temp='std',altitude=aircraft_altitude_ft)
        #else:
        #    return row['aircraft_TAS_at_fuel_start']
        '''

    def compute_TAS_KnotsfromMach_atFuelEnd(self, row ):
        from trajectory.aerocalc.airspeed import mach2tas 
        mach = row['aircraft_mach_at_fuel_end']
        aircraft_altitude_ft = row['aircraft_altitude_ft_at_fuel_end']
        return mach2tas ( mach=mach,temp='std',altitude=aircraft_altitude_ft)
    
        '''
        mach = max( row['aircraft_mach_at_fuel_end'], eps_single)
        if ( row['aircraft_TAS_at_fuel_end'] <= eps_single ) and (mach > 0.0 ):
            aircraft_altitude_ft = row['aircraft_altitude_ft_at_fuel_end']
            return mach2tas ( mach=mach,temp='std',altitude=aircraft_altitude_ft)
        else:
            return row['aircraft_TAS_at_fuel_end']
        '''
            
    def compute_CAS_KnotsfromMach_atFuelStart(self, row ):
        from trajectory.aerocalc.airspeed import  mach_alt2cas
        mach = row['aircraft_mach_at_fuel_start']
        aircraft_altitude_ft = row['aircraft_altitude_ft_at_fuel_start']
        return mach_alt2cas ( mach=mach,altitude=aircraft_altitude_ft)
        '''
        mach = max( row['aircraft_mach_at_fuel_start'], eps_single)
        if ( row['aircraft_CAS_at_fuel_start'] < eps_single ) and (mach > 0.0 ):
            aircraft_altitude_ft = row['aircraft_altitude_ft_at_fuel_start']
            return mach_alt2cas ( mach=mach,altitude=aircraft_altitude_ft)
        else:
            return row['aircraft_CAS_at_fuel_start']
        '''
        
    def compute_CAS_KnotsfromMach_atFuelEnd(self, row ):
        from trajectory.aerocalc.airspeed import  mach_alt2cas
        mach = row['aircraft_mach_at_fuel_end']
        aircraft_altitude_ft = row['aircraft_altitude_ft_at_fuel_end']
        return mach_alt2cas ( mach=mach,altitude=aircraft_altitude_ft)
        '''
        mach = max( row['aircraft_mach_at_fuel_end'], eps_single)
        if ( row['aircraft_CAS_at_fuel_end'] < eps_single ) and (mach > 0.0 ):
            aircraft_altitude_ft = row['aircraft_altitude_ft_at_fuel_end']
            return mach_alt2cas ( mach=mach,altitude=aircraft_altitude_ft)
        else:
            return row['aircraft_CAS_at_fuel_end']
        '''
    
    def computeMissingTASCASfromMach(self, df):
        # Machine epsilon for single precision (32-bit)
        df['aircraft_TAS_at_fuel_start'] = df.apply( self.compute_TAS_KnotsfromMach_atFuelStart , axis = 1)
        df['aircraft_TAS_at_fuel_end']   = df.apply( self.compute_TAS_KnotsfromMach_atFuelEnd , axis = 1)
        
        df['aircraft_CAS_at_fuel_start'] = df.apply( self.compute_CAS_KnotsfromMach_atFuelStart , axis = 1)
        df['aircraft_CAS_at_fuel_end']   = df.apply( self.compute_CAS_KnotsfromMach_atFuelEnd , axis = 1)
        return df
    
    def correctTimeDifferenceFuelBurntStartTakeoff(self , row):
        if row['fuel_burnt_start_relative_to_takeoff_sec'] < eps_single :
            return 0.0
        else:
            return row['fuel_burnt_start_relative_to_takeoff_sec']
        
    def correctTimeDifferenceFuelBurntEndTakeoff(self , row):
        if row['fuel_burnt_end_relative_to_takeoff_sec'] < eps_single :
            return 0.0
        else:
            return row['fuel_burnt_end_relative_to_takeoff_sec']
        
    def correctTimeDifferenceFuelBurntEndLanded(self , row):
        if row['fuel_burnt_end_relative_to_landed_sec'] < eps_single :
            return 0.0
        else:
            return row['fuel_burnt_end_relative_to_landed_sec']
            
    ''' 5th November 2025 - these features should be made available by the java frames '''
    def correctTimeDifferencesFuelBurntStartEnd(self, df):
        df['fuel_burnt_start_relative_to_takeoff_sec'] = df.apply( self.correctTimeDifferenceFuelBurntStartTakeoff , axis = 1  )
        df['fuel_burnt_end_relative_to_takeoff_sec'] = df.apply( self.correctTimeDifferenceFuelBurntEndTakeoff , axis = 1  )
        df['fuel_burnt_end_relative_to_landed_sec'] = df.apply(  self.correctTimeDifferenceFuelBurntEndLanded , axis = 1)
        return df
    
    def addTrainRankUseDifferenciatorColumns(self , df , train_rank_str  ):
        df['train_rank'] = train_rank_str
        idx_train_rank_columnName = 'idx' + "-" + train_rank_str
        df[idx_train_rank_columnName] = df['idx']
        return df
    
    def cleanEmptyAircraftColumnsAndFillInCorrectly(self , df , train_rank_str):
        print ( train_rank_str )
        listOfAircraftColumns = ["Num_Engines","Approach_Speed_knot","Length_ft","Wingspan_ft_without_winglets_sharklets",
                                 "Tail_Height_at_OEW_ft","Wheelbase_ft","Cockpit_to_Main_Gear_ft","Main_Gear_Width_ft",
                                 "MTOW_kg","MALW_kg","Parking_Area_ft2"]
        df = dropUnusedColumns(df, listOfAircraftColumns)
        print ( df.shape )
        ''' merge with flight list data '''
        df_extendedFlightList = self.getFlightListMergedWithAircrafts(train_rank_str)
        print ( list ( df_extendedFlightList))
        ''' need to keep the aircraft type to derive them into dummies '''
        listOfColumnsToKeep = ['flight_id','aircraft_type'] + listOfAircraftColumns
        df_extendedFlightList = keepOnlyColumns ( df_extendedFlightList , listOfColumnsToKeep )
        
        ''' merge train/rank dataframe with flight list extended with airports data '''
        df = pd.merge ( df , df_extendedFlightList , left_on='flight_id', right_on='flight_id', how='inner')
        print ( list ( df ))
        return df
    
    ''' manage a concatenated Train and Rank dataframe to apply the same transformations to both '''
    def concatenateTrainRank (self ):
        ''' manage the train dataframe '''
        filePath = os.path.join( self.javaTrainRankfilesFolder , self.extendedFuelTrainDataFileName)
        file = Path(filePath )
        
        directory = Path(self.javaTrainRankfilesFolder)
        if directory.is_dir() and file.is_file():
            
            #start_time = time.time()
            train_dataset = pd.read_parquet ( filePath )
            # True means it is the train
            train_dataset = self.addTrainRankUseDifferenciatorColumns ( train_dataset , 'train' )
            print ( train_dataset.shape )
            assert train_dataset.shape[0] == TrainDataSetRowCount
            ''' clean aircraft features and reload them from the aircraft database '''
            train_dataset = self.cleanEmptyAircraftColumnsAndFillInCorrectly( train_dataset , 'train')
            print ( train_dataset.shape )
            
            ''' manage the rank dataframe '''
            filePath = os.path.join( self.javaTrainRankfilesFolder , self.extendedRankFuelDataFileName )
            file = Path(filePath )
            
            directory = Path(self.javaTrainRankfilesFolder)
            if directory.is_dir() and file.is_file():
                                
                rank_dataset = pd.read_parquet ( filePath )
                rank_dataset = self.addTrainRankUseDifferenciatorColumns ( rank_dataset , 'rank' )
                print ( rank_dataset.shape )
                assert rank_dataset.shape[0] == RankDataSetRowCount
                
                ''' clean aircraft features and reload them from the aircraft database '''
                rank_dataset = self.cleanEmptyAircraftColumnsAndFillInCorrectly( rank_dataset , 'rank')
                print ( rank_dataset.shape )
                
                # concat Train and Rank
                train_rank_dataset = pd.concat( [ train_dataset , rank_dataset ])
                print ( train_rank_dataset.shape )
                return train_rank_dataset
        return None
    
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

    def Build_Model_From_Train_old(self , extendedFuelTrainDataFileName ):
        
        filePath = os.path.join( self.javaTrainRankfilesFolder , extendedFuelTrainDataFileName)
        file = Path(filePath )
        
        directory = Path(self.javaTrainRankfilesFolder)
        if directory.is_dir() and file.is_file():
            
            start_time = time.time()
            
            train_dataset = pd.read_parquet ( filePath )
            print( train_dataset.shape )
            print ( list (train_dataset))
            assert train_dataset.shape[0] == TrainDataSetRowCount
            
            #train_dataset = dropUnusedColumns(train_dataset , ['idx', 'fuel_kg', 'start' , 'end' , 'flight_id'])
            train_dataset = dropUnusedColumns(train_dataset , ['idx', 'fuel_kg', 'start' , 'end' ])
            ''' drop column with empty values '''
            'Wingspan_ft_without_winglets_sharklets'
            train_dataset = dropUnusedColumns(train_dataset , 'Wingspan_ft_without_winglets_sharklets')
            train_dataset = train_dataset.fillna(0.0)
                        
            ''' clean outliers '''
            listOfColumnNamesToKeep = list ( train_dataset)

            ''' merge with flight list data '''
            flightListExtendedWithAirportsDataFrame = self.getFlightListMergedWithAirports("train")
            print ( list ( flightListExtendedWithAirportsDataFrame))
            
            ''' filter on subset of needed columns '''
            ''' 3rd November 2025 - aircraft_type is needed to perform outliers corrections based upon a groupby this aircraft type '''
            flightListColumnsToKeep = ['flight_id' , 'origin_longitude_deg' , 'origin_latitude_deg' , 'origin_elevation_ft' ,
                                       'destination_longitude_deg', 'destination_latitude_deg', 'destination_elevation_ft']
            
            flightListExtendedWithAirportsDataFrame = keepOnlyColumns ( flightListExtendedWithAirportsDataFrame , flightListColumnsToKeep)
            
            ''' merge train with flight list extended with airports data '''
            train_dataset = pd.merge ( train_dataset , flightListExtendedWithAirportsDataFrame , left_on='flight_id', right_on='flight_id', how='inner')
            #train_dataset = pd.merge( train_dataset )
            print ( list ( train_dataset))
            
            ''' use clean outliers with capping quantiles without groupby flight_id '''
            train_dataset = self.clean_outliers_capped( train_dataset , listOfColumnsWithOutliers)
            #train_dataset = self.clean_outliers_capping_with_groupby(train_dataset , 'aircraft_type' , listOfColumnsWithOutliers)
            print ( list (train_dataset))
            
            #trainFlightListDataframe 
            ''' compute distance from airport origin to each aircraft position at fuel start and fuel end '''
            train_dataset = self.computeDistanceBetweenOriginAirportAndAircraftPosition(train_dataset)
            ''' compute missing speeds from mach '''
            train_dataset = self.computeMissingTASCASfromMach(train_dataset)
            ''' correct time difference between fuel burn start and end from takeoff '''
            #train_dataset = self.correctTimeDifferencesFuelBurntStartEnd (train_dataset)
            ''' see the results '''

            ''' drop column flight id '''
            train_dataset = dropUnusedColumns(train_dataset , ['flight_id','aircraft_type'])
            listOfColumnNamesToKeep = listOfColumnNamesToKeep + ['aircraft_distance_origin_to_fuel_start_Nm','aircraft_distance_origin_to_fuel_end_Nm',
                                                                 'aircraft_distance_fuel_end_to_destination_Nm', 'aircraft_distance_fuel_end_to_destination_Nm']
            train_dataset = keepOnlyColumns( train_dataset , listOfColumnNamesToKeep)
            print( list ( train_dataset ))
            #print(tabulate(train_dataset[-10:], headers='keys', tablefmt='grid' , showindex=True , ))
            #print(tabulate(train_dataset[:10], headers='keys', tablefmt='grid' , showindex=True , ))

            ''' do not scale the independent variable Y '''
            y_columnName = 'fuel_flow_kg_sec'
            X = train_dataset.drop( y_columnName , axis = 1)
            ''' check the stats '''
            #print ( tabulate( train_dataset.describe().transpose() , headers='keys', tablefmt='grid' , showindex=True , ))

            ''' scale only the dependent variables  '''
            X = self.scaleDataset( X )
            
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
            '''  Split the data (80% train, 20% test) hence 0.2'''
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            ''' split data set in 0% train and 20% test '''
            epochs = 300
            model_file_path , currentDateTimeAsString = self.tf_model_fit( X_train, y_train , epochs )
            print ( model_file_path )
            
            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            
            with CustomObjectScope({'rmse': 'mean_absolute_error'}):
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
            
            return model_file_path
    
    ''' 6th November 2025 - common method to make predictions from ranking dataframe '''
    def predictFromRankAndModel(self , modelFilePath , X_rank): 
        
        start_time = time.time

        logging.basicConfig(level=logging.INFO)
        logging.info (' -------------- Rank Fuel -------------')
        
        ''' Save and load a model with the custom activation '''
        with CustomObjectScope({'loss' :  self.rmse}, {'rmse': self.rmse}):
            model = load_model(modelFilePath)
            
        listOfColumnsToDrop = ['idx-train', 'idx-rank', 'fuel_kg', 'start' , 'end' ,'flight_id','aircraft_type','train_rank']
        X_rank = dropUnusedColumns(X_rank , listOfColumnsToDrop)

        print ( X_rank.isnull().any(axis=1).sum() )
        print ( X_rank.info())
        X_rank = X_rank.fillna(0.0)
        
        ''' DO NOT USE -> do not use groupby flight id to clean outliers '''
        ''' 6th November 2025- not clear if outliers capping is usefull on the ranking dataframe '''
        #X_rank = clean_outliers_capping_with_groupby( X_rank , 'flight_id' , listOfColumnsWithOutliers)
        X_rank = self.clean_outliers_capped( X_rank , listOfColumnsWithOutliers)
        assert X_rank.shape[0] == RankDataSetRowCount
        
        print ( list (X_rank ))
        X_rank = self.scaleDataset( X_rank )

        ''' convert True False to float '''
        X_rank = np.asarray(X_rank).astype(np.float32)
            
        ''' generate predictions '''            #predictions = model.predict(X_rank[np.newaxis, ...])
        predictions = model.predict(X_rank)
        print ( predictions.shape )
        assert predictions.shape[0] == RankDataSetRowCount

        ''' ensure that there no empty values '''
        # Convert predictions to a Pandas DataFrame
        y_columnName = 'fuel_flow_kg_sec'
        df_predictions = pd.DataFrame(predictions, columns=[y_columnName])
        print ( df_predictions.shape )
        assert df_predictions.shape[0] == RankDataSetRowCount
        
        print ("number of null values in the predictions = " +  str(df_predictions.isnull().any(axis=1).sum()) )
        ''' empty or N/A submissions are rejected '''
        assert df_predictions.isnull().any(axis=1).sum() == 0

        ''' make all predictions greater than zero '''
        #df_predictions = df_predictions.abs()
        
        #print(tabulate(df_predictions[:10], headers='keys', tablefmt='grid' , showindex=True , ))
        
        assert df_predictions.shape[0] == RankDataSetRowCount
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
    
    ''' do not use anymore '''
    def predictFromRankAndModel_old(self , model_file_name , extendedRankFuelDataFileName ):
        logging.basicConfig(level=logging.INFO)
        
        start_time = time.time
        
        logging.info (' -------------- Rank Fuel -------------')
        
        localfilesFolder = os.path.dirname(__file__)
        filePathModel = os.path.join(localfilesFolder , model_file_name)
        
        # Save and load a model with the custom activation
        with CustomObjectScope({'rmse': self.rmse}):
            model = load_model(filePathModel)
         
        rankFilePath = os.path.join( self.javaTrainRankfilesFolder , extendedRankFuelDataFileName)
        rankFile = Path(rankFilePath )
        
        directory = Path(self.javaTrainRankfilesFolder)
        if directory.is_dir() and rankFile.is_file():
            
            X_rank = pd.read_parquet ( rankFilePath )
            print( X_rank.shape )
            print ( list (X_rank ))
            assert X_rank.shape[0] == RankDataSetRowCount
            
            #print(tabulate(X_rank.describe().transpose()[:10], headers='keys', tablefmt='grid' , showindex=True , ))

            X_rank = dropUnusedColumns(X_rank , ['idx' , 'start' , 'end' ,  'fuel_kg' , 'fuel_flow_kg_sec'])
            'Wingspan_ft_without_winglets_sharklets'
            X_rank = dropUnusedColumns(X_rank , 'Wingspan_ft_without_winglets_sharklets')
             
            ''' we should not have not a number in the fuel_flow_kg_sec column '''
            assert ( X_rank.isnull().any(axis=1).sum() == 0)
            print ( X_rank.info())
            X_rank = X_rank.fillna(0.0)
            
            ''' DO NOT USE -> do not use groupby flight id to clean outliers '''
            #X_rank = clean_outliers_capping_with_groupby( X_rank , 'flight_id' , listOfColumnsWithOutliers)
            #X_rank = self.clean_outliers_capped( X_rank , listOfColumnsWithOutliers)
            listOfColumnNamesToKeep = list ( X_rank)

            ''' merge with flight list data '''
            flightListExtendedWithAirportsDataframe = self.getFlightListMergedWithAirports("rank")
            print ( list ( flightListExtendedWithAirportsDataframe))

            ''' filter on subset of needed columns '''
            flightListColumnsToKeep = ['flight_id' , 'origin_longitude_deg' , 'origin_latitude_deg' , 'origin_elevation_ft' ,
                                       'destination_longitude_deg', 'destination_latitude_deg', 'destination_elevation_ft']
            ''' drop unused columns '''
            flightListExtendedWithAirportsDataframe = keepOnlyColumns ( flightListExtendedWithAirportsDataframe , flightListColumnsToKeep)
            
            ''' merge rank with flight list extended with airports data '''
            X_rank = pd.merge ( X_rank , flightListExtendedWithAirportsDataframe , left_on='flight_id', right_on='flight_id', how='inner')
            print ( X_rank.shape )
            assert X_rank.shape[0] == RankDataSetRowCount
            #train_dataset = pd.merge( train_dataset )
            print ( list ( X_rank))
            
            print ( X_rank.info())
            ''' there are null values in the elevation ft feature '''
            assert X_rank.isnull().any(axis=1).sum() == 0
            
            ''' 2nd October 2025 - 22h48 - add clean outliers on the rank dataset '''
            ''' use clean outliers with capping quantiles without groupby flight_id '''
            X_rank = self.clean_outliers_capped( X_rank , listOfColumnsWithOutliers)
            #train_dataset = self.clean_outliers_capping_with_groupby(train_dataset , 'aircraft_type' , listOfColumnsWithOutliers)
            print ( list (X_rank))
            
            ''' compute distance between airports and aircraft position at fuel start end '''
            X_rank = self.computeDistanceBetweenOriginAirportAndAircraftPosition(X_rank)
            ''' compute missing speeds from mach '''
            X_rank = self.computeMissingTASCASfromMach(X_rank)
            ''' correct time difference between fuel burn start and end from takeoff '''
            #X_rank = self.correctTimeDifferencesFuelBurntStartEnd (X_rank)
            ''' see the results '''
            #print ( tabulate( X_rank.describe().transpose() , headers='keys', tablefmt='grid' , showindex=True , ))

            #trainFlightListDataframe 
            ''' drop column flight id '''
            X_rank = dropUnusedColumns(X_rank , ['flight_id'])
            
            listOfColumnNamesToKeep = listOfColumnNamesToKeep + ['aircraft_distance_origin_to_fuel_start_Nm',
                                                                 'aircraft_distance_origin_to_fuel_end_Nm',
                                                                 'aircraft_distance_fuel_end_to_destination_Nm', 
                                                                 'aircraft_distance_fuel_end_to_destination_Nm']
            X_rank = keepOnlyColumns( X_rank , listOfColumnNamesToKeep)

            print( str ( X_rank.shape ))
            assert X_rank.shape[0] == RankDataSetRowCount
            
            #print(tabulate(X_rank[:10], headers='keys', tablefmt='grid' , showindex=True , ))
            
            print ( list (X_rank ))
            X_rank = self.scaleDataset( X_rank )

            ''' convert True False to float '''
            X_rank = np.asarray(X_rank).astype(np.float32)
            
            ''' generate predictions '''            #predictions = model.predict(X_rank[np.newaxis, ...])
            predictions = model.predict(X_rank)
            print ( predictions.shape )

            # Convert predictions to a Pandas DataFrame
            y_columnName = 'fuel_flow_kg_sec'
            df_predictions = pd.DataFrame(predictions, columns=[y_columnName])
            print ( df_predictions.shape )
            assert df_predictions.shape[0] == RankDataSetRowCount
            
            print ("number of null values in the predictions = " +  str(df_predictions.isnull().any(axis=1).sum()) )
            ''' empty or N/A submissions are rejected '''
            assert df_predictions.isnull().any(axis=1).sum() == 0

            ''' make all predictions greater than zero '''
            df_predictions = df_predictions.abs()
            
            #print(tabulate(df_predictions[:10], headers='keys', tablefmt='grid' , showindex=True , ))
            
            assert df_predictions.shape[0] == RankDataSetRowCount
            
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
            #print(f"Execution Time: {end_time - start_time} seconds")
            return rankSubmissionFilePath
        
    ''' compute fuel kg from fuel flow '''
    def computeFuelKg( self , row ):
        return (abs( row['fuel_flow_kg_sec'] ) * row['time_diff_seconds'])
    
    def suppressUTC ( self, row , columnName ):
        from datetime import timezone
        return row[columnName].replace(tzinfo=timezone.utc).astimezone(tz=None)
    
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
            assert X_rank.shape[0] == RankDataSetRowCount
            
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
                assert df_predictions.shape[0] == RankDataSetRowCount
                
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
    ''' latest version where all transformations are applied to both train and rank dataframes'''
    def extendCorrectTrainRankDataframe(self , concatenatedTrainRankDataset):
        
        assert concatenatedTrainRankDataset.shape[0] == TrainDataSetRowCount + RankDataSetRowCount 
    
        #print(tabulate(concatenatedTrainRankDataset[-10:], headers='keys', tablefmt='grid' , showindex=False , ))
        #print(tabulate(concatenatedTrainRankDataset[:10], headers='keys', tablefmt='grid' , showindex=False , ))
    
        assert concatenatedTrainRankDataset.shape[0] == TrainDataSetRowCount + RankDataSetRowCount 
        
        ''' compute TAS and CAS from mach when mach is not null and TAS or CAS are null '''
        concatenatedTrainRankDataset = prcDataChallenge2025Submissions.ccomputeMissingTASCASfromMachconcatenatedTrainRankDataset)
        #print(tabulate(concatenatedTrainRankDataset[-10:], headers='keys', tablefmt='grid' , showindex=False , ))
        #print(tabulate(concatenatedTrainRankDataset[:10], headers='keys', tablefmt='grid' , showindex=False , ))
 
        concatenatedTrainRankDataset['hasSharklets'] = np.where ( concatenatedTrainRankDataset['Wingspan_ft_without_winglets_sharklets'].isnull() , 0 , 1)
        #print(tabulate(concatenatedTrainRankDataset[-10:], headers='keys', tablefmt='grid' , showindex=False , ))
        #print(tabulate(concatenatedTrainRankDataset[:10], headers='keys', tablefmt='grid' , showindex=False , ))
    
        ''' transform aircraft_type into 0/1 category '''
        ''' 6th November 2025 - v25 -> v26 -> test without dummies '''
        ''' after v26- add dummies again '''
        concatenatedTrainRankDataset = pd.get_dummies( concatenatedTrainRankDataset , columns=['aircraft_type'], dtype = int)
        #print ( list ( concatenatedTrainRankDataset ) )
    
        listOfColumnsToDrop = ['idx','aircraft_type','start','end']
        concatenatedTrainRankDataset = dropUnusedColumns(concatenatedTrainRankDataset, listOfColumnsToDrop)
        
        return concatenatedTrainRankDataset
    
if __name__ == '__main__':
    import platform
    logging.basicConfig(level=logging.INFO)
    print("python version = " + platform.python_version())
    print("tensorflow version = " + tf.__version__)
    print("pandas version = " + pd. __version__)
    print("numpy version = " + np. __version__)
    
    #extendedFuelTrainDataFileName = "ExtendedFuel_train_2025-10-27-18-08-12.parquet"
    #extendedFuelTrainDataFileName = "ExtendedFuel_train_2025-10-31-12-44-23.parquet"
    extendedFuelTrainDataFileName = "ExtendedFuel_train_2025-10-31-12-44-23.parquet"
    extendedFuelTrainDataFileName = "ExtendedFuel_train_2025-11-05-02-26-18.parquet"
    
    #extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-10-26-12-04-34.parquet"
    #extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-10-27-19-52-33.parquet"
    extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-10-31-17-36-58.parquet"
    extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-11-05-03-09-31.parquet"

    javaTrainRankfilesFolder = "C:/Users/rober/eclipse-2025-09/eclipse-jee-2025-09-R-win32-x86_64/Data-Challenge-2025/documents"
    ''' common class instance '''
    prcDataChallenge2025Submissions = PRCdataChallenge2025Submissions(extendedFuelTrainDataFileName , extendedRankFuelDataFileName, javaTrainRankfilesFolder)
    
    ''' in order to apply the same transformations - first concatenate train and rank '''
    concatenatedTrainRankDataset = prcDataChallenge2025Submissions.concatenateTrainRank()
    
    ''' 6 November 2025 - temporary useful because Java activities are currently failing '''
    concatenatedTrainRankDataset = prcDataChallenge2025Submissions.extendCorrectTrainRankDataframe ( concatenatedTrainRankDataset )
    
    ''' df['train_rank'] = train_rank_str '''
    ''' filter again the merged dataset to focus on train only '''
    trainDataSet = concatenatedTrainRankDataset[ concatenatedTrainRankDataset['train_rank'] == 'train']
    #print ( trainDataSet.shape)
    
    ''' build the model '''
    generatedModelFileName = prcDataChallenge2025Submissions.Build_Model_From_Train (trainDataSet)
    #generatedModelFileName = "results_model_2025-11-06-22-40-57.h5"
    
    ''' extract the ranking dataset from the concatenated dataframe '''
    rankingDataset = concatenatedTrainRankDataset[ concatenatedTrainRankDataset['train_rank'] == 'rank']
    print ( rankingDataset.shape )
    
    #listOfColumnsToDrop = ['idx','flight_id','start','end','idx-train','idx-rank','train_rank']
    columnsToDropList = ['idx-train', 'idx-rank', 'fuel_kg', 'start' , 'end' ,'flight_id','aircraft_type','train_rank','Wingspan_ft_without_winglets_sharklets']
    rankingDataset = dropUnusedColumns( rankingDataset, columnsToDropList )
    #generatedModelFileName = "results_model_2025-11-06-08-09-53.h5"
    CsvPredictionsFilePath = prcDataChallenge2025Submissions.predictFromRankAndModel(generatedModelFileName , rankingDataset)
    #print("generated CSV results file path = " + CsvPredictionsFilePath)
    
    #CsvPredictionsFilePath = "fuel_rank_submission_2025-11-01-11-57-32.csv"
    #CsvPredictionsFilePath = "fuel_rank_submission_2025-11-06-08-52-18.csv"

    generatedTeamSubmissionParquetFileName =  prcDataChallenge2025Submissions.generateTeamSubmissionParquetFile(
            CsvPredictionsFilePath , extendedRankFuelDataFileName)
        
    print ( generatedTeamSubmissionParquetFileName )
    
    ''' upload parquet to S3 destination '''
    prcDataChallenge2025Submissions.uploadTeamParquetFileToS3( )
        