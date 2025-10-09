'''
Created on 7 oct. 2025

@author: robert
'''

import logging
import os
import pandas as pd
from pathlib import Path

from trajectory.Environment.AirportsDataChallenge.AirportsDataChallengeDatabaseFile import AirportsDataChallengeDatabase

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
            logging.info (self.className + "it is a file - {0}".format(self.filePathFlightListTrain))
            
            self.TrainFlightListDataframe = pd.read_parquet ( self.filePathFlightListTrain )
            logging.info ( str(self.TrainFlightListDataframe.shape ) )
            logging.info ( str(  list ( self.TrainFlightListDataframe)) )
            
            #logging.info ( self.TrainFlightListDataframe.head(10) )
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
            
            #logging.info ( self.RankFlightListDataframe.head(10) )
        
            return True
        else:
            return False
    
    def collectUniqueAircraftTypes(self):
        pass
        df = self.TrainFlightListDataframe [self.TrainFlightListDataframe['aircraft_type'].notnull()]
        logging.info( df.head ())
    
    def collectUniqueAirports(self):
        
        logging.info(self.className + ": ------- collectUniqueAirports -------- ")
        
        self.train = self.TrainFlightListDataframe [self.TrainFlightListDataframe['origin_icao'].notnull()]
        dfTrain = self.train['origin_icao']
        dfTrain = dfTrain.rename( 'airport_icao' )
        logging.info ( str(  list ( dfTrain)) )

        #logging.info( dfTrain.head (100 ))
        logging.info ( str(dfTrain.shape ) )
        
        self.rank  = self.RankFlightListDataframe [self.RankFlightListDataframe['destination_icao'].notnull()]
        dfRank = self.rank ['destination_icao']
        dfRank = dfRank.rename( 'airport_icao' )

        logging.info ( str(  list ( dfRank)) )
        #logging.info( dfRank.head (100 ))
        logging.info ( str(dfRank.shape ) )
        
        dfConcat = pd.concat( [dfTrain , dfRank] )
        logging.info( dfConcat )
        
        logging.info ( str(dfConcat.shape ) )
        dfConcat = dfConcat.unique ( )
        logging.info (self.className + ": size of unique list of airports : " + str(dfConcat.shape ) )
        #logging.info( dfConcat.head(100))
        
    def extendFlightListWithAirportData(self):
        
        logging.info(self.className + ": ---------- extendFlightListWithAirportData ---- ")
        
        airportsDb = AirportsDataChallengeDatabase()
        assert airportsDb.read() == True
        assert airportsDb.checkHeaders() == True
        
        airportsDataframe = airportsDb.getAirportsDataframe()
        
        logging.info( str ( list ( airportsDataframe ) ) )
        logging.info( str ( list ( self.TrainFlightListDataframe ) ) )
        
        ''' extend origin icao '''
        df_flightListExtendedWithAirportData = pd.merge ( self.TrainFlightListDataframe , airportsDataframe , left_on='origin_icao', right_on='icao', how='inner' )
        logging.info( str ( list ( df_flightListExtendedWithAirportData ) ) )

        ''' suppress icao '''
        df_flightListExtendedWithAirportData = df_flightListExtendedWithAirportData.drop( ['icao'] , axis=1 )
        ''' rename extended columns '''
        df_flightListExtendedWithAirportData = df_flightListExtendedWithAirportData.rename(columns= {'latitude':'origin_latitude','longitude':'origin_longitude','elevation':'origin_elevation'})
        logging.info( str ( list ( df_flightListExtendedWithAirportData ) ) )
        
        ''' extend destination icao '''
        df_flightListExtendedWithAirportData = pd.merge ( df_flightListExtendedWithAirportData , airportsDataframe , left_on='destination_icao', right_on='icao', how='inner' )
        
        ''' suppress icao '''
        df_flightListExtendedWithAirportData = df_flightListExtendedWithAirportData.drop( ['icao'] , axis=1 )
        ''' rename extended columns '''
        df_flightListExtendedWithAirportData = df_flightListExtendedWithAirportData.rename(columns= {'latitude':'destination_latitude','longitude':'destination_longitude','elevation':'destination_elevation'})
        logging.info( str ( list ( df_flightListExtendedWithAirportData ) ) )
        
        self.extendedTrainFlightListDataframe = df_flightListExtendedWithAirportData

        #logging.info ( df_flightListExtendedWithAirportData.head(10) )
        