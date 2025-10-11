'''
Created on 8 oct. 2025

@author: robert
'''

import logging
import os
from pathlib import Path
import pandas as pd

expectedHeaders =['idx', 'flight_id', 'start', 'end', 'fuel_kg', 'time_diff_seconds']

class FuelDatabase(object):
    
    className = ''
    
    def __init__(self):
        self.className = self.__class__.__name__
        
        self.fileNameFuelTrain = "fuel_train.parquet"
        logging.info(self.fileNameFuelTrain)
        
        self.fileNameFuelRank =  "fuel_rank_submission.parquet"
        logging.info(self.fileNameFuelRank)
        
        self.filesFolder = os.path.dirname(__file__)
        
        self.filePathFuelTrain = os.path.join(self.filesFolder , self.fileNameFuelTrain)
        logging.info(self.filePathFuelTrain)
        
        self.filePathFuelRank = os.path.join(self.filesFolder , self.fileNameFuelRank)
        logging.info(self.filePathFuelRank)
        
    def checkFuelTrainHeaders(self):
        return (set(self.FuelTrainDataframe) == set(expectedHeaders))
    
    def checkFuelRankHeaders(self):
        return (set(self.FuelRankDataframe) == set(expectedHeaders))

    def getFuelTrainDataframe(self):
        return self.FuelTrainDataframe
    
    def getFuelRankDataframe(self):
        return self.FuelRankDataframe
    
    def addTimeDiffSeconds(self , df):
        df['time_diff_seconds'] = (df['end'] - df['start']).dt.total_seconds()
        return df
    
    def readFuelRank(self):
        logging.info(self.filePathFuelRank)
        
        directory = Path(self.filesFolder)
        logging.info(directory)
        
        file = Path(self.filePathFuelRank)
        
        if directory.is_dir() and file.is_file():
            
            logging.info (self.className + "it is a directory - {0}".format(self.filesFolder))
            logging.info (self.className + "it is a file - {0}".format(self.filePathFuelRank))
            
            self.FuelRankDataframe = pd.read_parquet ( self.filePathFuelRank )
            
            ''' Calculate time difference in seconds '''
            self.FuelRankDataframe = self.addTimeDiffSeconds(self.FuelRankDataframe)
            
            logging.info ( str(self.FuelRankDataframe.shape ) )
            logging.info ( str(  list ( self.FuelRankDataframe)) )
        
            return True
        else:
            self.FuelRankDataframe = None
            return False
        
    def readFuelTrain(self):
        
        logging.info(self.filePathFuelTrain)
        
        directory = Path(self.filesFolder)
        logging.info(directory)
        
        file = Path(self.filePathFuelTrain)
        
        if directory.is_dir() and file.is_file():
            
            logging.info (self.className + "it is a directory - {0}".format(self.filesFolder))
            logging.info (self.className + "it is a file - {0}".format(self.filePathFuelTrain))
            
            self.FuelTrainDataframe = pd.read_parquet ( self.filePathFuelTrain )
            
            ''' Calculate time difference in seconds '''
            self.FuelTrainDataframe = self.addTimeDiffSeconds(self.FuelTrainDataframe)

            logging.info ( str(self.FuelTrainDataframe.shape ) )
            logging.info ( str(  list ( self.FuelTrainDataframe)) )
        
            return True
        else:
            self.FuelTrainDataframe = None
            return False
        