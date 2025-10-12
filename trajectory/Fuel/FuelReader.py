'''
Created on 8 oct. 2025

@author: robert
'''

import logging
import os
from pathlib import Path
import pandas as pd

from pandas.api.types import is_datetime64_any_dtype

from trajectory.FlightList.FlightListReader import FlightListDatabase
from trajectory.utils import keepOnlyColumns

Count_of_FlightsFiles = 1000

expectedHeaders =['idx', 'flight_id', 'takeoff', 'fuel_burn_start', 'fuel_burn_end', 'fuel_kg', 'time_diff_seconds' , 'fuel_flow_kg_sec' , 
                  'fuel_burn_relative_start','fuel_burn_relative_end' ,
                  'origin_longitude', 'origin_latitude', 'origin_elevation', 'destination_longitude', 'destination_latitude', 'destination_elevation',
                  'flight_distance_Nm' , 'flight_duration_sec']

def compute_fuel_flow_kg_sec(row):
    return row['fuel_kg'] / (row['end'] - row['start']).total_seconds() if (row['end'] - row['start']).total_seconds() != 0 else 0

def checkTimeZoneUTC(row):
    if ( row['fuel_burn_start'].tzinfo is None ):
        print("fuel burn start is TimeZone naive")
        
    if ( row['fuel_burn_end'].tzinfo is None ):
        print("fuel burn end is TimeZone naive")
        
    if  row['takeoff'].tzinfo is None :
        print("takeoff is TimeZone naive")
        
    else:
        print("takeoff TimeZone = " + row['takeoff'].tzinfo)


class FuelDatabase(object):
    
    className = ''
    
    def __init__(self):
        logging.basicConfig(level=logging.INFO)

        self.className = self.__class__.__name__
        
        self.fileNameFuelTrain = "fuel_train.parquet"
        #logging.info(self.fileNameFuelTrain)
        self.fileNameFuelRank =  "fuel_rank_submission.parquet"
        #logging.info(self.fileNameFuelRank)
        self.filesFolder = os.path.dirname(__file__)
        self.filePathFuelTrain = os.path.join(self.filesFolder , self.fileNameFuelTrain)
        #logging.info(self.filePathFuelTrain)
        self.filePathFuelRank = os.path.join(self.filesFolder , self.fileNameFuelRank)
        #logging.info(self.filePathFuelRank)
        
    def checkFuelTrainHeaders(self):
        return (set(self.FuelTrainDataframe) == set(expectedHeaders))
    
    def checkFuelRankHeaders(self):
        return (set(self.FuelRankDataframe) == set(expectedHeaders))

    def getFuelTrainDataframe(self):
        return self.FuelTrainDataframe
    
    def getFuelRankDataframe(self):
        return self.FuelRankDataframe
    
        ''' compute difference in seconds between end and start '''
    def addTimeDiffSeconds(self , df):
        df['time_diff_seconds'] = (df['end'] - df['start']).dt.total_seconds()
        return df
    
        ''' convert absolute date time into relative delay in seconds from flight start '''
    def computeRelativeStartEndFromFlightTakeOff(self, df):
        #print(df.info)
        
        df['fuel_burn_relative_start'] = ( df['fuel_burn_start'] - df['takeoff']).dt.total_seconds()
        df['fuel_burn_relative_end'] = ( df['fuel_burn_end'] - df['takeoff']).dt.total_seconds()
        #print(df.info)
        return df
    
    def computeFuelFlowKgSeconds(self , df ):
        df['fuel_flow_kg_sec'] = df.apply( compute_fuel_flow_kg_sec, axis=1 )
        return df
    
    def renameStartEndColumns(self , df ):
        df = df.rename(columns= {'start':'fuel_burn_start','end':'fuel_burn_end'})
        return df
    
    def convertDatetimeToUTC(self, df):
        for columnName in list(df):
            #print(columnName)
            if is_datetime64_any_dtype(df[columnName]):
                #print(self.className + ": column is datetime = " + columnName)
                df[columnName] = pd.to_datetime(df[columnName], utc=True)
        return df
    
    def readFuelRank(self):
        logging.basicConfig(level=logging.INFO)
        #logging.info(self.filePathFuelRank)
        
        directory = Path(self.filesFolder)
        #logging.info(directory)
        file = Path(self.filePathFuelRank)
        
        if directory.is_dir() and file.is_file():
            
            self.FuelRankDataframe = pd.read_parquet ( self.filePathFuelRank )
            self.FuelRankDataframe = self.convertDatetimeToUTC(self.FuelRankDataframe)
            ''' Calculate time difference in seconds '''
            self.FuelRankDataframe = self.addTimeDiffSeconds(self.FuelRankDataframe)
            self.FuelRankDataframe = self.computeFuelFlowKgSeconds(self.FuelRankDataframe)
            self.FuelRankDataframe = self.renameStartEndColumns(self.FuelRankDataframe)
            
            assert self.extendFuelRankWithFlightTakeOff()
            self.FuelRankDataframe = self.computeRelativeStartEndFromFlightTakeOff(self.FuelRankDataframe)
            
            self.FuelRankDataframe = self.FuelRankDataframe.head(Count_of_FlightsFiles)

            return True
        else:
            logging.error (self.className + "it is a directory - {0}".format(self.filesFolder))
            logging.error (self.className + "it is a file - {0}".format(self.filePathFuelRank))
 
            self.FuelRankDataframe = None
            return False
        
    def readFuelTrain(self):
        logging.basicConfig(level=logging.INFO)

        #logging.info(self.filePathFuelTrain)
        directory = Path(self.filesFolder)
        #logging.info(directory)
        file = Path(self.filePathFuelTrain)
        
        if directory.is_dir() and file.is_file():
            
            #logging.info (self.className + "it is a directory - {0}".format(self.filesFolder))
            #logging.info (self.className + "it is a file - {0}".format(self.filePathFuelTrain))
            
            self.FuelTrainDataframe = pd.read_parquet ( self.filePathFuelTrain )
            print("----- origin fuel train dataframe shape = " + str ( self.FuelTrainDataframe.shape ))
            self.FuelTrainDataframe = self.convertDatetimeToUTC(self.FuelTrainDataframe)

            ''' Calculate time difference in seconds '''
            self.FuelTrainDataframe = self.addTimeDiffSeconds(self.FuelTrainDataframe)
            self.FuelTrainDataframe = self.computeFuelFlowKgSeconds(self.FuelTrainDataframe)
            self.FuelTrainDataframe = self.renameStartEndColumns(self.FuelTrainDataframe)

            assert self.extendFuelTrainWithFlightTakeOff()
            self.FuelTrainDataframe = self.computeRelativeStartEndFromFlightTakeOff(self.FuelTrainDataframe)
            
            ''' test mode - keep only first 10 rows '''
            self.FuelTrainDataframe = self.FuelTrainDataframe.head(Count_of_FlightsFiles)

            #logging.info ( str(self.FuelTrainDataframe.shape ) )
            #logging.info ( str(  list ( self.FuelTrainDataframe)) )
        
            return True
        else:
            self.FuelTrainDataframe = None
            return False
        
    def extendFuelRankWithFlightTakeOff(self):    
        
        logging.basicConfig(level=logging.INFO)

        flightListDatabase = FlightListDatabase()
        assert flightListDatabase.readRankFlightList()
        
        df_rankFlightList = flightListDatabase.getRankFlightListDataframe()
        logging.info( self.className + ": ---- Rank flight list = " + str ( list (df_rankFlightList ) ) )
        
        columnNameListToKeep = [ 'flight_id', 'takeoff' ,'origin_longitude', 'origin_latitude', 'origin_elevation', 'destination_longitude', 'destination_latitude', 'destination_elevation']
        df_rankFlightList = keepOnlyColumns( df_rankFlightList , columnNameListToKeep )
        
        logging.info( self.className + ": ---- rank flight list = " + str ( list (df_rankFlightList ) ) )
        logging.info( self.className + ": ---- fuel rank  = " + str ( list (self.FuelRankDataframe ) ) )

        ''' extend in order to obtain flight start date time '''
        self.FuelRankDataframe = pd.merge ( self.FuelRankDataframe , df_rankFlightList , left_on='flight_id', right_on='flight_id', how='inner' )
        logging.info( str ( list ( self.FuelRankDataframe ) ) )
        
        return True
        
    def extendFuelTrainWithFlightTakeOff(self ):
        
        flightListDatabase = FlightListDatabase()
        assert flightListDatabase.readTrainFlightList()
        
        df_trainFlightList = flightListDatabase.getTrainFlightListDataframe()
        logging.info( self.className + ": ---- train flight list = " + str ( list (df_trainFlightList ) ) )
        
        columnNameListToKeep = [ 'flight_id', 'takeoff' ,'origin_longitude', 'origin_latitude', 'origin_elevation', 
                                'destination_longitude', 'destination_latitude', 'destination_elevation',
                                'flight_distance_Nm' , 'flight_duration_sec']
        df_trainFlightList = keepOnlyColumns( df_trainFlightList , columnNameListToKeep )
        
        logging.info( self.className + ": ---- train flight list = " + str ( list (df_trainFlightList ) ) )
        logging.info( self.className + ": ---- fuel train  = " + str ( list (self.FuelTrainDataframe ) ) )

        ''' extend in order to obtain flight start date time '''
        self.FuelTrainDataframe = pd.merge ( self.FuelTrainDataframe , df_trainFlightList , left_on='flight_id', right_on='flight_id', how='inner' )
        logging.info( str ( list ( self.FuelTrainDataframe ) ) )
        
        return True
        