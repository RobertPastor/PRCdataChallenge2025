'''
Created on 8 oct. 2025

@author: robert
'''

import logging
import os
from pathlib import Path
import pandas as pd
from trajectory.FlightList.FlightListReader import FlightListDatabase

expectedHeaders =['idx', 'flight_id', 'takeoff', 'fuel_burn_start', 'fuel_burn_end', 'fuel_kg', 'time_diff_seconds' , 'fuel_flow_kg_sec']

def compute_fuel_flow_kg_sec(row):
    return row['fuel_kg'] / (row['end'] - row['start']).total_seconds() if (row['end'] - row['start']).total_seconds() != 0 else 0



class FuelDatabase(object):
    
    className = ''
    
    def __init__(self):
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
    
    def addTimeDiffSeconds(self , df):
        df['time_diff_seconds'] = (df['end'] - df['start']).dt.total_seconds()
        return df
    
    def computeFuelFlowKgSeconds(self , df ):
        df['fuel_flow_kg_sec'] = df.apply( compute_fuel_flow_kg_sec, axis=1 )
        return df
    
    def renameStartEndColumns(self , df ):
        df = df.rename(columns= {'start':'fuel_burn_start','end':'fuel_burn_end'})
        return df
    
    def readFuelRank(self):
        #logging.info(self.filePathFuelRank)
        
        directory = Path(self.filesFolder)
        #logging.info(directory)
        file = Path(self.filePathFuelRank)
        
        if directory.is_dir() and file.is_file():
            
            self.FuelRankDataframe = pd.read_parquet ( self.filePathFuelRank )
            ''' Calculate time difference in seconds '''
            self.FuelRankDataframe = self.addTimeDiffSeconds(self.FuelRankDataframe)
            self.FuelRankDataframe = self.computeFuelFlowKgSeconds(self.FuelRankDataframe)
            self.FuelRankDataframe = self.renameStartEndColumns(self.FuelRankDataframe)
            
            assert self.extendFuelRankWithFlightTakeOff()

            return True
        else:
            logging.info (self.className + "it is a directory - {0}".format(self.filesFolder))
            logging.info (self.className + "it is a file - {0}".format(self.filePathFuelRank))
 
            self.FuelRankDataframe = None
            return False
        
    def readFuelTrain(self):
        
        #logging.info(self.filePathFuelTrain)
        directory = Path(self.filesFolder)
        #logging.info(directory)
        file = Path(self.filePathFuelTrain)
        
        if directory.is_dir() and file.is_file():
            
            #logging.info (self.className + "it is a directory - {0}".format(self.filesFolder))
            #logging.info (self.className + "it is a file - {0}".format(self.filePathFuelTrain))
            
            self.FuelTrainDataframe = pd.read_parquet ( self.filePathFuelTrain )
            
            ''' Calculate time difference in seconds '''
            self.FuelTrainDataframe = self.addTimeDiffSeconds(self.FuelTrainDataframe)
            self.FuelTrainDataframe = self.computeFuelFlowKgSeconds(self.FuelTrainDataframe)
            self.FuelTrainDataframe = self.renameStartEndColumns(self.FuelTrainDataframe)

            assert self.extendFuelTrainWithFlightTakeOff()
            #logging.info ( str(self.FuelTrainDataframe.shape ) )
            #logging.info ( str(  list ( self.FuelTrainDataframe)) )
        
            return True
        else:
            self.FuelTrainDataframe = None
            return False
        
    def extendFuelRankWithFlightTakeOff(self):    
        
        flightListDatabase = FlightListDatabase()
        assert flightListDatabase.readRankFlightList()
        
        df_rankFlightList = flightListDatabase.getRankFlightListDataframe()
        logging.info( self.className + ": ---- Rank flight list = " + str ( list (df_rankFlightList ) ) )
 
        columnNameListToKeep = [ 'flight_id', 'takeoff']
        for columnName in list(df_rankFlightList):
            if not columnName in columnNameListToKeep:
                df_rankFlightList = df_rankFlightList.drop(columnName, axis=1)
        
        logging.info( self.className + ": ---- train flight list = " + str ( list (df_rankFlightList ) ) )
        
        ''' extend in order to obtain flight start date time '''
        df_FuelDataExtendedWithFlightTakeOff = pd.merge ( self.FuelRankDataframe , df_rankFlightList , left_on='flight_id', right_on='flight_id', how='inner' )
        logging.info( str ( list ( df_FuelDataExtendedWithFlightTakeOff ) ) )
        
        self.FuelRankDataframe = df_FuelDataExtendedWithFlightTakeOff
        return True
        
    def extendFuelTrainWithFlightTakeOff(self ):
        
        flightListDatabase = FlightListDatabase()
        assert flightListDatabase.readTrainFlightList()
        
        df_trainFlightList = flightListDatabase.getTrainFlightListDataframe()
        logging.info( self.className + ": ---- train flight list = " + str ( list (df_trainFlightList ) ) )
        
        columnNameListToKeep = [ 'flight_id', 'takeoff']
        for columnName in list(df_trainFlightList):
            if not columnName in columnNameListToKeep:
                df_trainFlightList = df_trainFlightList.drop(columnName, axis=1)
        
        logging.info( self.className + ": ---- train flight list = " + str ( list (df_trainFlightList ) ) )

        ''' extend in order to obtain flight start date time '''
        df_FuelDataExtendedWithFlightTakeOff = pd.merge ( self.FuelTrainDataframe , df_trainFlightList , left_on='flight_id', right_on='flight_id', how='inner' )
        logging.info( str ( list ( df_FuelDataExtendedWithFlightTakeOff ) ) )
        
        self.FuelTrainDataframe = df_FuelDataExtendedWithFlightTakeOff
        return True
        