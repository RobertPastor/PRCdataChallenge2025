'''
Created on 8 nov. 2025

@author: robert

'''
import platform

import numpy as np
eps_single = np.finfo(np.float32).eps

import pandas as pd
import os
# Set the option to display all columns
pd.options.display.max_columns = None
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

from tabulate import tabulate

''' warning - use tensor flow 2.12.0 not the latest 2.20.0 that is causing DLL problems '''
import tensorflow as tf


from trajectory.Utils.utils import dropUnusedColumns
import logging

from pathlib import Path

from TensorFlow.BBB_TensorFlowBaseClassFile import TensorFlowBaseClass

TrainDataSetRowCount = 131530
RankDataSetRowCount = 24289

''' clean outliers '''
''' specific deal for mach TAS and CAS '''
listOfColumnsWithOutliers = ["aircraft_altitude_ft_at_fuel_start",
                             "aircraft_altitude_ft_at_fuel_end" , 
                             
                            "aircraft_CAS_at_fuel_end" , 
                            "aircraft_CAS_at_fuel_start",
                            
                            "aircraft_computed_vertical_rate_ft_min",
                              
                            "aircraft_delta_altitude_ft_end_destination",
                            
                             "aircraft_delta_altitude_ft_origin_end_start",
                             "aircraft_delta_altitude_ft_origin_fuel_start",
                             "aircraft_delta_altitude_ft_start_destination",

                             "aircraft_distance_flown_origin_end_Nm",
                             "aircraft_distance_flown_origin_start_Nm",
                             "aircraft_distance_flown_start_end_Nm",
                             
                             "aircraft_distance_to_be_flown_end_destination_Nm",
                             "aircraft_distance_to_be_flown_start_destination_Nm",

                             "aircraft_groundspeed_kt_at_fuel_end",
                             "aircraft_groundspeed_kt_at_fuel_start",
                             
                             "aircraft_groundspeed_kt_X_at_fuel_end",
                             "aircraft_groundspeed_kt_X_at_fuel_start",
                             "aircraft_groundspeed_kt_Y_at_fuel_end",
                             "aircraft_groundspeed_kt_Y_at_fuel_start",

                             "aircraft_mach_at_fuel_end",
                             "aircraft_mach_at_fuel_start",
                             
                             "aircraft_TAS_at_fuel_end",
                             "aircraft_TAS_at_fuel_start",
                             
                            "aircraft_track_angle_deg_at_fuel_end",
                            "aircraft_track_angle_deg_at_fuel_start",
                            "aircraft_track_angle_rad_at_fuel_end",
                            "aircraft_track_angle_rad_at_fuel_start",
                             
                            "aircraft_vertical_rate_ft_min_at_fuel_end",
                            "aircraft_vertical_rate_ft_min_at_fuel_start",

                            "fuel_burnt_end_relative_to_landed_sec",
                            "fuel_burnt_end_relative_to_takeoff_sec",
                            "fuel_burnt_start_relative_to_landed_sec",
                            "fuel_burnt_start_relative_to_takeoff_sec",

                            "fuel_burnt_start_relative_to_takeoff_sec",
                            "fuel_burnt_end_relative_to_takeoff_sec",
                            "fuel_burnt_end_relative_to_landed_sec",
                            
                            ]


class PRCdataChallenge2025Submissions(TensorFlowBaseClass):
    """Exemple de classe simple"""
    
    def __init__(self , extendedFuelTrainDataFileName , extendedRankFuelDataFileName , extendedFinalFuelDataFileName ,
                 javaTrainRankfilesFolder):
        self.extendedFuelTrainDataFileName = extendedFuelTrainDataFileName
        self.extendedRankFuelDataFileName = extendedRankFuelDataFileName
        self.extendedFinalFuelDataFileName = extendedFinalFuelDataFileName
        
        self.javaTrainRankfilesFolder = javaTrainRankfilesFolder
        super(PRCdataChallenge2025Submissions, self).__init__(TrainDataSetRowCount , RankDataSetRowCount , listOfColumnsWithOutliers )
        
    def clean_outliers_capped(self , df , list_of_columnNames_to_clip):
        for columnName in list_of_columnNames_to_clip:
            Q1 = df[columnName].quantile(0.25)
            Q3 = df[columnName].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df[columnName] = np.clip(df[columnName], lower_bound, upper_bound)
            
        return df

    ''' used to filter part of the big dataframe containing the train records or the rank records '''
    def addTrainRankUseDifferenciatorColumns(self , df , train_rank_str  ):
        ''' add column with constant content either "rank" or "train" - used to filter afterwards '''
        df['train_rank'] = train_rank_str
        return df
    
    ''' latest version where all transformations are applied to both train and rank dataframes'''
    def extendCorrectTrainRankDataframe(self , concatenatedTrainRankDataset):
        ''' compute TAS and CAS from mach when mach is not null and TAS or CAS are null '''
        ''' if mach is NaN and groundspeed is OK then use groundspeed as TAS value without wind '''
        concatenatedTrainRankDataset = prcDataChallenge2025Submissions.computeTASnCASfromMachOrGroundSpeed(concatenatedTrainRankDataset)
        
        assert concatenatedTrainRankDataset.shape[0] == TrainDataSetRowCount + RankDataSetRowCount

        #print(tabulate(concatenatedTrainRankDataset[-10:], headers='keys', tablefmt='grid' , showindex=False , ))
        #print(tabulate(concatenatedTrainRankDataset[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        ''' set info discriminating whether aircraft has or not winglets or sharklets '''
        concatenatedTrainRankDataset['hasSharklets'] = np.where ( concatenatedTrainRankDataset['Wingspan_ft_without_winglets_sharklets'].isnull() , 0 , 1)
        assert concatenatedTrainRankDataset.shape[0] == TrainDataSetRowCount + RankDataSetRowCount

        ''' use clean outliers with capped quantiles without groupby flight_id nor groupby on aircraft code '''
        ''' after v30 - do not cap / clip the values '''
        #concatenatedTrainRankDataset = self.clean_outliers_capped( concatenatedTrainRankDataset , self.listOfColumnsWithOutliers)
        #assert concatenatedTrainRankDataset.shape[0] == TrainDataSetRowCount + RankDataSetRowCount
        
        #listOfColumnsToDrop = ['idx','flight_id', 'start', 'end' , 'aircraft_ICAO_Code', 'train_rank']
        #df_temp = dropUnusedColumns ( concatenatedTrainRankDataset , listOfColumnsToDrop )
        #print ( tabulate ( df_temp.describe(include='all'), headers='keys', tablefmt='grid' , showindex=False , ) )
        
        ''' 6th November 2025 - v25 -> v26 -> test without dummies for the aircraft type code'''
        ''' after v26- add dummies again '''
        concatenatedTrainRankDataset = pd.get_dummies( concatenatedTrainRankDataset , columns=['aircraft_ICAO_Code'], dtype = int)
        print ( concatenatedTrainRankDataset.shape )
        print ( list ( concatenatedTrainRankDataset ) )
        
        ''' suppress all degrees features '''
        listOfColumnsToDrop = [ 'aircraft_latitude_deg_at_fuel_start', 'aircraft_longitude_deg_at_fuel_start',
                                'aircraft_latitude_deg_at_fuel_end', 'aircraft_longitude_deg_at_fuel_end',
                                 'aircraft_track_angle_deg_at_fuel_start', 'aircraft_track_angle_deg_at_fuel_end',
                                  'Wingspan_ft_without_winglets_sharklets', 'Wingspan_ft_with_winglets_sharklets']
        concatenatedTrainRankDataset = dropUnusedColumns ( concatenatedTrainRankDataset , listOfColumnsToDrop)
        
        assert concatenatedTrainRankDataset.shape[0] == TrainDataSetRowCount + RankDataSetRowCount
        return concatenatedTrainRankDataset
    
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
            
            ''' manage the rank dataframe '''
            filePath = os.path.join( self.javaTrainRankfilesFolder , self.extendedRankFuelDataFileName )
            file = Path(filePath )
            
            directory = Path(self.javaTrainRankfilesFolder)
            if directory.is_dir() and file.is_file():
                                
                rank_dataset = pd.read_parquet ( filePath )
                rank_dataset = self.addTrainRankUseDifferenciatorColumns ( rank_dataset , 'rank' )
                print ( rank_dataset.shape )
                assert rank_dataset.shape[0] == RankDataSetRowCount
                
                # concat Train and Rank
                train_rank_dataset = pd.concat( [ train_dataset , rank_dataset ])
                print ( train_rank_dataset.shape )
                assert train_rank_dataset.shape[0] == TrainDataSetRowCount + RankDataSetRowCount
                print ( list ( train_rank_dataset ) )
                
                return train_rank_dataset
            
        return None

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    print("python version = " + platform.python_version())
    print("tensorflow version = " + tf.__version__)
    print("pandas version = " + pd. __version__)
    print("numpy version = " + np. __version__)
    
    #extendedFuelTrainDataFileName = "ExtendedFuel_train_2025-10-27-18-08-12.parquet"
    #extendedFuelTrainDataFileName = "ExtendedFuel_train_2025-10-31-12-44-23.parquet"
    extendedFuelTrainDataFileName = "ExtendedFuel_train_2025-10-31-12-44-23.parquet"
    extendedFuelTrainDataFileName = "ExtendedFuel_train_2025-11-05-02-26-18.parquet"
    extendedFuelTrainDataFileName = "ExtendedFuel_train_2025-11-08-13-48-02.parquet"
    extendedFuelTrainDataFileName = "ExtendedFuel_train_2025-11-15-22-22-08.parquet"
    extendedFuelTrainDataFileName = "ExtendedFuel_train_2025-11-19-12-46-33.parquet"
    extendedFuelTrainDataFileName = "ExtendedFuel_train_2025-11-20-13-26-25.parquet"
    
    #extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-10-26-12-04-34.parquet"
    #extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-10-27-19-52-33.parquet"
    extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-10-31-17-36-58.parquet"
    extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-11-05-03-09-31.parquet"
    extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-11-08-14-38-59.parquet"
    extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-11-15-23-20-10.parquet"
    extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-11-19-21-58-34.parquet"
    extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-11-20-09-16-03.parquet"
    extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-11-24-19-32-56.parquet"
    extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-11-25-22-18-57.parquet"
    
    extendedFinalFuelDataFileName = "ExtendedFuel_rank_2025-11-DD-HH-MM-SS.parquet"

    javaTrainRankfilesFolder = "C:/Users/rober/eclipse-2025-09/eclipse-jee-2025-09-R-win32-x86_64/Data-Challenge-2025/documents/"
    ''' common class instance '''
    prcDataChallenge2025Submissions = PRCdataChallenge2025Submissions(extendedFuelTrainDataFileName , \
                                                                      extendedRankFuelDataFileName, extendedFinalFuelDataFileName ,\
                                                                      javaTrainRankfilesFolder)
    
    ''' in order to apply the same transformations - first concatenate train and rank '''
    concatenatedTrainRankDataset = prcDataChallenge2025Submissions.concatenateTrainRank()
    print ( concatenatedTrainRankDataset.shape )
    assert concatenatedTrainRankDataset.shape[0] == TrainDataSetRowCount + RankDataSetRowCount
    
    ''' 8 November 2025 - temporary useful because Java activities are not dealing correctly TAS and CAS '''
    concatenatedTrainRankDataset = prcDataChallenge2025Submissions.extendCorrectTrainRankDataframe ( concatenatedTrainRankDataset )
    print ( concatenatedTrainRankDataset.shape )
    
    assert concatenatedTrainRankDataset.shape[0] == TrainDataSetRowCount + RankDataSetRowCount
    print ( tabulate ( concatenatedTrainRankDataset.describe().transpose(), headers='keys', tablefmt='grid' , showindex=False , ))
    
    ''' filter again the merged dataset to extract train data only '''
    trainDataSet = concatenatedTrainRankDataset[ concatenatedTrainRankDataset['train_rank'] == 'train']
    ''' build the model '''
    generatedModelFileName = prcDataChallenge2025Submissions.BuildModelFromTrain (trainDataSet)
    
    #generatedModelFileName = "results_model_v39.h5"
    print ("--> generated model file = " +  generatedModelFileName )
    
    rankingDataset = concatenatedTrainRankDataset[ concatenatedTrainRankDataset['train_rank'] == 'rank']
    
    ''' make the predictions with ranking dataframe '''
    CsvPredictionsFilePath = prcDataChallenge2025Submissions.predictFromRankAndModel(generatedModelFileName , rankingDataset)
    #CsvPredictionsFilePath = "fuel_rank_submission_2025-11-15-23-53-28.csv"
    print ("generated CSV predictions file = " + CsvPredictionsFilePath )
    
    generatedTeamSubmissionParquetFileName =  prcDataChallenge2025Submissions.generateTeamSubmissionParquetFile(CsvPredictionsFilePath , extendedRankFuelDataFileName)
    print ( generatedTeamSubmissionParquetFileName )
    
    ''' upload parquet to S3 destination '''
    ''' no need to provide a version , the version is computed on the fly '''
    prcDataChallenge2025Submissions.uploadTeamParquetFileToS3( )


