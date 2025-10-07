'''
Created on 7 oct. 2025

@author: robert
'''

import logging
import os
import pandas as pd
from pathlib import Path

class FlightListDatabase(object):
    className = ''
    
    
    def __init__(self):
        self.className = self.__class__.__name__
        
        self.fileNameFlightListTrain = "flightlist_train.parquet"
        logging.info(self.fileNameFlightListTrain)
        
        self.fileNameFlightListRank =  "flight_list_rank.parquet"
        logging.info(self.fileNameFlightListRank)
        
        self.filesFolder = os.path.dirname(__file__)
        
        self.filePathFlightListTrain = os.path.join(self.filesFolder , self.fileNameFlightListTrain)
        logging.info(self.filePathFlightListTrain)
        
        self.filePathFlightListRank = os.path.join(self.filesFolder , self.fileNameFlightListRank)
        logging.info(self.filePathFlightListRank)
        
    def readTrainFlightList(self ):
        logging.info(self.filePathFlightListTrain)
        
        directory = Path(self.filesFolder)
        logging.info(directory)
        
        if directory.is_dir():
            
            logging.info (self.className + "it is a directory - {0}".format(self.filesFolder))
            
            df = pd.read_parquet ( self.filePathFlightListTrain )
            logging.info ( str(df.shape ) )
            logging.info ( str(  list ( df)) )
            
            logging.info ( df.head(10) )
        
        return True

        
    def readRankFlightList(self ):
        logging.info(self.filePathFlightListRank)
        
        directory = Path(self.filesFolder)
        logging.info(directory)
        
        if directory.is_dir():
            
            logging.info (self.className + "it is a directory - {0}".format(self.filesFolder))
            
            df = pd.read_parquet ( self.filePathFlightListRank )
            logging.info ( str(df.shape ) )
            logging.info ( str(  list ( df)) )
            
            logging.info ( df.head(10) )
        
        return True