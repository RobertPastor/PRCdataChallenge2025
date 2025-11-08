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
from TensorFlow.TensorFlowBaseClassFile import TensorFlowBaseClass

TrainDataSetRowCount = 131530
RankDataSetRowCount = 24289

''' clean outliers '''
''' specific deal for mach TAS and CAS '''
listOfColumnsWithOutliers = ["aircraft_altitude_ft_at_fuel_start","aircraft_altitude_ft_at_fuel_end" , 
                            "aircraft_vertical_rate_ft_min_at_fuel_start","aircraft_vertical_rate_ft_min_at_fuel_end",
                            "aircraft_computed_vertical_rate_ft_min",
                            
                            "fuel_burnt_start_relative_to_takeoff_sec","fuel_burnt_end_relative_to_takeoff_sec",
                            "fuel_burnt_end_relative_to_landed_sec",
                            "aircraft_vertical_rate_ft_min_at_fuel_start","aircraft_vertical_rate_ft_min_at_fuel_end"]


class PRCdataChallenge2025Submissions(TensorFlowBaseClass):
    """Exemple de classe simple"""
    
    def __init__(self , extendedFuelTrainDataFileName , extendedRankFuelDataFileName , javaTrainRankfilesFolder):
        self.extendedFuelTrainDataFileName = extendedFuelTrainDataFileName
        self.extendedRankFuelDataFileName = extendedRankFuelDataFileName
        self.javaTrainRankfilesFolder = javaTrainRankfilesFolder
        
        super(PRCdataChallenge2025Submissions, self).__init__(TrainDataSetRowCount , RankDataSetRowCount , listOfColumnsWithOutliers )
        pass

    ''' used to filter part of the big dataframe containing the train records or the rank records '''
    def addTrainRankUseDifferenciatorColumns(self , df , train_rank_str  ):
        ''' add column with constant content either "rank" or "train" - used to filter afterwards '''
        df['train_rank'] = train_rank_str
        return df
        
    def compute_TAS_KnotsfromMach_atFuelStart(self , row):
        from trajectory.aerocalc.airspeed import mach2tas 
        mach = row['aircraft_mach_at_fuel_start']
        aircraft_altitude_ft = row['aircraft_altitude_ft_at_fuel_start']

        TAS = row['aircraft_TAS_at_fuel_start']
        if TAS != np.nan:
            return TAS
        else:
            #TAS is empty
            if (mach is not None) and (mach != np.nan):
                return  mach2tas ( mach=mach,altitude=aircraft_altitude_ft)
            else:
                return np.nan

    def compute_TAS_KnotsfromMach_atFuelEnd(self , row):
        from trajectory.aerocalc.airspeed import mach2tas 
        mach = row['aircraft_mach_at_fuel_end']
        aircraft_altitude_ft = row['aircraft_altitude_ft_at_fuel_end']

        TAS = row['aircraft_TAS_at_fuel_end']
        if TAS != np.nan:
            return TAS
        else:
            # TAS is empty            
            if (mach is not None) and (mach != np.nan):
                return  mach2tas ( mach=mach,altitude=aircraft_altitude_ft)
            else:
                return np.nan

        ''' compute CAS from mach '''
    def compute_CAS_KnotsfromMach_atFuelStart(self, row ):
        from trajectory.aerocalc.airspeed import  mach_alt2cas
        mach = row['aircraft_mach_at_fuel_start']
        aircraft_altitude_ft = row['aircraft_altitude_ft_at_fuel_start']
        
        CAS = row['aircraft_CAS_at_fuel_start']
        if (CAS != np.nan):
            return CAS
        else:
            if (mach is not None) and (mach != np.nan):
                ''' assumption is that altitude is always provided -> no altitude missings content '''
                return  mach_alt2cas ( mach=mach,altitude=aircraft_altitude_ft)
            else:
                return np.nan
    
    def compute_CAS_KnotsfromMach_atFuelEnd(self, row ):
        from trajectory.aerocalc.airspeed import  mach_alt2cas
        mach = row['aircraft_mach_at_fuel_end']
        aircraft_altitude_ft = row['aircraft_altitude_ft_at_fuel_end']

        CAS = row['aircraft_CAS_at_fuel_end']
        if (CAS != np.nan):
            return CAS
        else:
            if (mach is not None) and (mach != np.nan):
                ''' assumption is that altitude is always provided -> no altitude missings content '''
                return  mach_alt2cas ( mach=mach,altitude=aircraft_altitude_ft)
            else:
                return np.nan
    
    ''' use python aerocal method to compute TAS and CAS from mach when TAS or CAS are null '''
    def computeMissingSpeeds(self, df):
        # Machine epsilon for single precision (32-bit)
        df['aircraft_TAS_at_fuel_start'] = df.apply( self.compute_TAS_KnotsfromMach_atFuelStart , axis = 1)
        df['aircraft_TAS_at_fuel_end']   = df.apply( self.compute_TAS_KnotsfromMach_atFuelEnd , axis = 1)
        
        df['aircraft_CAS_at_fuel_start'] = df.apply( self.compute_CAS_KnotsfromMach_atFuelStart , axis = 1)
        df['aircraft_CAS_at_fuel_end']   = df.apply( self.compute_CAS_KnotsfromMach_atFuelEnd , axis = 1)
        return df

    
    ''' latest version where all transformations are applied to both train and rank dataframes'''
    def extendCorrectTrainRankDataframe(self , concatenatedTrainRankDataset):
        
        ''' compute TAS and CAS from mach when mach is not null and TAS or CAS are null '''
        concatenatedTrainRankDataset = prcDataChallenge2025Submissions.computeMissingSpeeds(concatenatedTrainRankDataset)
        #print(tabulate(concatenatedTrainRankDataset[-10:], headers='keys', tablefmt='grid' , showindex=False , ))
        #print(tabulate(concatenatedTrainRankDataset[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        ''' set info discriminating whether aircraft has or not winglets or sharklets '''
        concatenatedTrainRankDataset['hasSharklets'] = np.where ( concatenatedTrainRankDataset['Wingspan_ft_without_winglets_sharklets'].isnull() , 0 , 1)
        
        ''' 6th November 2025 - v25 -> v26 -> test without dummies '''
        ''' after v26- add dummies again '''
        concatenatedTrainRankDataset = pd.get_dummies( concatenatedTrainRankDataset , columns=['aircraft_ICAO_Code'], dtype = int)
        print ( concatenatedTrainRankDataset.shape )
        print ( list ( concatenatedTrainRankDataset ) )
        
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
                print ( list ( train_rank_dataset ) )
                
                return train_rank_dataset
        return None

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
    extendedFuelTrainDataFileName = "ExtendedFuel_train_2025-11-08-13-48-02.parquet"
    
    #extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-10-26-12-04-34.parquet"
    #extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-10-27-19-52-33.parquet"
    extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-10-31-17-36-58.parquet"
    extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-11-05-03-09-31.parquet"
    extendedRankFuelDataFileName = "ExtendedFuel_rank_2025-11-08-14-38-59.parquet"

    javaTrainRankfilesFolder = "C:/Users/rober/eclipse-2025-09/eclipse-jee-2025-09-R-win32-x86_64/Data-Challenge-2025/documents/11-08-2025-November-08"
    ''' common class instance '''
    prcDataChallenge2025Submissions = PRCdataChallenge2025Submissions(extendedFuelTrainDataFileName , extendedRankFuelDataFileName, javaTrainRankfilesFolder)
    
    ''' in order to apply the same transformations - first concatenate train and rank '''
    concatenatedTrainRankDataset = prcDataChallenge2025Submissions.concatenateTrainRank()
    print ( concatenatedTrainRankDataset.shape )
    assert concatenatedTrainRankDataset.shape[0] == TrainDataSetRowCount + RankDataSetRowCount
    
    ''' 8 November 2025 - temporary useful because Java activities are not dealing correctly TAS and CAS '''
    concatenatedTrainRankDataset = prcDataChallenge2025Submissions.extendCorrectTrainRankDataframe ( concatenatedTrainRankDataset )
    print ( concatenatedTrainRankDataset.shape )
    assert concatenatedTrainRankDataset.shape[0] == TrainDataSetRowCount + RankDataSetRowCount
    
    ''' df['train_rank'] = train_rank_str '''
    ''' filter again the merged dataset to focus on train only '''
    trainDataSet = concatenatedTrainRankDataset[ concatenatedTrainRankDataset['train_rank'] == 'train']
    #print ( trainDataSet.shape)
    
    ''' build the model '''
    #generatedModelFileName = prcDataChallenge2025Submissions.Build_Model_From_Train (trainDataSet)
    
    generatedModelFileName = "results_model_2025-11-08-17-39-43.h5"
    print ( generatedModelFileName )
    
    rankingDataset = concatenatedTrainRankDataset[ concatenatedTrainRankDataset['train_rank'] == 'rank']

    CsvPredictionsFilePath = prcDataChallenge2025Submissions.predictFromRankAndModel(generatedModelFileName , rankingDataset)

    #generatedTeamSubmissionParquetFileName =  prcDataChallenge2025Submissions.generateTeamSubmissionParquetFile(
    #        CsvPredictionsFilePath , extendedRankFuelDataFileName)
        
    #print ( generatedTeamSubmissionParquetFileName )

