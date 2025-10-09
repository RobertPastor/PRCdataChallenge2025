'''
Created on 7 oct. 2025

@author: robert
'''

import logging
import os
import pandas as pd
from pathlib import Path

expectedHeaders = ['flight_date', 'aircraft_type', 'takeoff', 'landed', 'origin_icao', 'origin_name', 'destination_icao', 'destination_name', 'flight_id']

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
        
    def checkTrainFlightListHeaders(self):
        return (set(self.TrainFlightListDataframe) == set(expectedHeaders))
        
    def checkRankFligthListHeaders(self):
        return (set(self.RankFlightListDataframe) == set(expectedHeaders))
        
    def readTrainFlightList(self ):
        logging.info(self.filePathFlightListTrain)
        
        directory = Path(self.filesFolder)
        logging.info(directory)
        
        file = Path(self.filePathFlightListTrain)
        
        if directory.is_dir() and file.is_file():
            
            logging.info (self.className + "it is a directory - {0}".format(self.filesFolder))
            
            self.TrainFlightListDataframe = pd.read_parquet ( self.filePathFlightListTrain )
            logging.info ( str(self.TrainFlightListDataframe.shape ) )
            logging.info ( str(  list ( self.TrainFlightListDataframe)) )
            
            logging.info ( self.TrainFlightListDataframe.head(10) )
        
            return True
        else:
            return False

        
    def readRankFlightList(self ):
        logging.info(self.filePathFlightListRank)
        
        directory = Path(self.filesFolder)
        logging.info(directory)
        
        file = Path(self.filePathFlightListRank)
        
        if directory.is_dir() and file.is_file():
            
            logging.info (self.className + "it is a directory - {0}".format(self.filesFolder))
            logging.info (self.className + "it is a file - {0}".format(self.filePathFlightListRank))
            
            self.RankFlightListDataframe = pd.read_parquet ( self.filePathFlightListRank )
            logging.info ( str(self.RankFlightListDataframe.shape ) )
            logging.info ( str(  list ( self.RankFlightListDataframe)) )
            
            logging.info ( self.RankFlightListDataframe.head(10) )
        
            return True
        else:
            return False
    
    def collectUniqueAircraftTypes(self):
        pass
        df = self.TrainFlightListDataframe [self.TrainFlightListDataframe['aircraft_type'].notnull()]
        logging.info( df.head (100 ))
    
    def collectUniqueAirports(self):
        
        logging.info(self.className + ": ------- collectUniqueAirports -------- ")
        
        dfTrain = self.TrainFlightListDataframe [self.TrainFlightListDataframe['origin_icao'].notnull()]
        dfTrain = dfTrain['origin_icao']
        logging.info ( str(  list ( dfTrain)) )
        
        dfTrain = dfTrain.rename( 'airport_icao' )
        logging.info ( str(  list ( dfTrain)) )

        logging.info( dfTrain.head (100 ))
        logging.info ( str(dfTrain.shape ) )
        
        dfRank = self.RankFlightListDataframe [self.RankFlightListDataframe['destination_icao'].notnull()]
        dfRank = dfRank['destination_icao']
        dfTrain = dfTrain.rename( 'airport_icao' )

        logging.info ( str(  list ( dfRank)) )

        logging.info( dfRank.head (100 ))
        logging.info ( str(dfRank.shape ) )
        
        dfConcat = pd.concat( [dfTrain , dfRank] )
        logging.info( dfConcat )
        
        logging.info ( str(dfConcat.shape ) )
        dfConcat = dfConcat.unique ( )
        logging.info ( str(dfConcat.shape ) )

        #logging.info( dfConcat.head(100))

        