'''
Created on 8 oct. 2025

@author: robert
'''

import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np

from pandas.api.types import is_datetime64_any_dtype

from trajectory.FlightList.FlightListReader import FlightListDatabase
from trajectory.utils import keepOnlyColumns
from trajectory.utils import dropUnusedColumns , oneHotEncoderSklearn , getCurrentDateTimeAsStr
from trajectory.Flights.FlightsReader import FlightsDatabase
from tabulate import tabulate


''' load static flight database '''
flightsDatabase = FlightsDatabase()

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
        
        
def extendFuelTrainWithFlightsData( row ):
    #print(''' ------------- row by row loop ------------------''')
    print (  f"Index: {row.name} , Train Flight id: {row['flight_id']} " )
    df_flightData = flightsDatabase.readOneTrainFile( row['flight_id'] )
    #print( str( df_flightData['timestamp'] ))
    df_filtered = df_flightData[ (df_flightData['timestamp'] >= row['fuel_burn_start']) & (df_flightData['timestamp'] <= row['fuel_burn_end'])]
    # keep first row only
    df_filtered = df_filtered.head(1)
    df_filtered = df_filtered.drop ( "flight_id" , axis = 1)
    
    #print(str ( df_filtered.shape ))
    if df_filtered.shape[0] == 0:
        constantValue = np.nan
        constantValue = 0.0
        return pd.Series( { 'timestamp' : (row['fuel_burn_start']) , 'aircraft_type_code' : str('unknown_aircraft_type_code') ,
                         'latitude' : (constantValue) , 'longitude' : (constantValue) ,
                         'altitude' : (constantValue) , 'groundspeed' : (constantValue) , 
                         'track' : (constantValue) , 'vertical_rate' : (constantValue) ,
                         'mach' : (constantValue) , 'TAS' : (constantValue) , 
                         'CAS' : (constantValue) , 'source' : (str('unknown_source'))} )
    else:
        return pd.Series( { 'timestamp' : (df_filtered['timestamp'].iloc[0]) , 'aircraft_type_code' : str(df_filtered['aircraft_type_code'].iloc[0]) ,
                         'latitude' : (df_filtered['latitude'].iloc[0]) , 'longitude' : (df_filtered['longitude'].iloc[0]) ,
                         'altitude' : (df_filtered['altitude'].iloc[0]) , 'groundspeed' : (df_filtered['groundspeed'].iloc[0]) , 
                         'track' : (df_filtered['track'].iloc[0]) , 'vertical_rate' : (df_filtered['vertical_rate'].iloc[0]) ,
                         'mach' : (df_filtered['mach'].iloc[0]) , 'TAS' : (df_filtered['TAS'].iloc[0]) , 
                         'CAS' : (df_filtered['CAS'].iloc[0]) , 'source' : (df_filtered['source'].iloc[0])} )


def extendFuelRankWithFlightsData( row ):
    #print(''' ------------- row by row loop ------------------''')
    print (  f"Index: {row.name} , Rank Flight id: {row['flight_id']} " )
    df_flightData = flightsDatabase.readOneRankFile( row['flight_id'] )
    #print( str( df_flightData['timestamp'] ))
    df_filtered = df_flightData[ (df_flightData['timestamp'] >= row['fuel_burn_start']) & (df_flightData['timestamp'] <= row['fuel_burn_end'])]
    # keep first row only
    df_filtered = df_filtered.head(1)
    df_filtered = df_filtered.drop ( "flight_id" , axis = 1)
    
    #print(str ( df_filtered.shape ))
    if df_filtered.shape[0] == 0:
        constantValue = np.nan
        constantValue = 0.0
        return pd.Series( { 'timestamp' : (row['fuel_burn_start']) , 'aircraft_type_code' : str('unknown') ,
                         'latitude' : (0.0) , 'longitude' : (0.0) ,
                         'altitude' : (0.0) , 'groundspeed' : (0.0) , 
                         'track' : (0.0) , 'vertical_rate' : (0.0) ,
                         'mach' : (0.0) , 'TAS' : (0.0) , 
                         'CAS' : (0.0) , 'source' : ('unknown')} )
    else:
        return pd.Series( { 'timestamp' : (df_filtered['timestamp'].iloc[0]) , 'aircraft_type_code' : str(df_filtered['aircraft_type_code'].iloc[0]) ,
                         'latitude' : (df_filtered['latitude'].iloc[0]) , 'longitude' : (df_filtered['longitude'].iloc[0]) ,
                         'altitude' : (df_filtered['altitude'].iloc[0]) , 'groundspeed' : (df_filtered['groundspeed'].iloc[0]) , 
                         'track' : (df_filtered['track'].iloc[0]) , 'vertical_rate' : (df_filtered['vertical_rate'].iloc[0]) ,
                         'mach' : (df_filtered['mach'].iloc[0]) , 'TAS' : (df_filtered['TAS'].iloc[0]) , 
                         'CAS' : (df_filtered['CAS'].iloc[0]) , 'source' : (df_filtered['source'].iloc[0])} )


class FuelDatabase(object):
    
    className = ''
    
    def __init__(self , count_of_files_to_read):
        logging.basicConfig(level=logging.INFO)

        self.className = self.__class__.__name__
        
        self.count_of_files_to_read = count_of_files_to_read
        
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
        #print(str(set(list(self.FuelTrainDataframe).sort())))
        #print(str(set(expectedHeaders).sort()))
        return set(self.FuelTrainDataframe) == set(expectedHeaders)
    
    def checkFuelRankHeaders(self):
        #print(str(set(list(self.FuelRankDataframe).sort())))
        #print(str(set(expectedHeaders).sort()))
        return set(self.FuelRankDataframe) == set(expectedHeaders)

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

            ''' specify count of files to read ''' 
            if self.count_of_files_to_read and self.count_of_files_to_read > 0:
                self.FuelRankDataframe = self.FuelRankDataframe.head(self.count_of_files_to_read )

            return True
        else:
            logging.error (self.className + " : it is a directory - {0}".format(self.filesFolder))
            logging.error (self.className + " : it is a file - {0}".format(self.filePathFuelRank))
 
            self.FuelRankDataframe = None
            return False
        
    def readFuelTrain(self):
        logging.basicConfig(level=logging.INFO)

        #logging.info(self.filePathFuelTrain)
        directory = Path(self.filesFolder)
        #logging.info(directory)
        file = Path(self.filePathFuelTrain)
        
        if directory.is_dir() and file.is_file():
            
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
            if self.count_of_files_to_read and self.count_of_files_to_read > 0:
                self.FuelTrainDataframe = self.FuelTrainDataframe.head(self.count_of_files_to_read)

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
        
        columnNameListToKeep = [ 'flight_id', 'takeoff' ,'origin_longitude', 'origin_latitude', 'origin_elevation', 
                                'destination_longitude', 'destination_latitude', 'destination_elevation',
                                'flight_distance_Nm' , 'flight_duration_sec']
        
        if flightListDatabase.isExtendedWithAircraftData():
            columnNameListToKeep = columnNameListToKeep + flightListDatabase.getAircraftExtendedListOfCharacteristics()
        df_rankFlightList = keepOnlyColumns( df_rankFlightList , columnNameListToKeep )
        
        logging.info( self.className + ": ---- rank flight list = " + str ( list (df_rankFlightList ) ) )
        logging.info( self.className + ": ---- fuel rank  = " + str ( list (self.FuelRankDataframe ) ) )

        ''' extend in order to obtain flight start date time '''
        self.FuelRankDataframe = pd.merge ( self.FuelRankDataframe , df_rankFlightList , left_on='flight_id', right_on='flight_id', how='inner' )
        logging.info( str ( list ( self.FuelRankDataframe ) ) )
        
        return True
        
    def extendFuelTrainWithFlightTakeOff(self ):
        
        flightListDatabase = FlightListDatabase()
        ''' reading the Flight list -> add aircraft data '''
        assert flightListDatabase.readTrainFlightList()
        
        df_trainFlightList = flightListDatabase.getTrainFlightListDataframe()
        logging.info( self.className + ": ---- train flight list = " + str ( list (df_trainFlightList ) ) )
        
        columnNameListToKeep = [ 'flight_id', 'takeoff' ,'origin_longitude', 'origin_latitude', 'origin_elevation', 
                                'destination_longitude', 'destination_latitude', 'destination_elevation',
                                'flight_distance_Nm' , 'flight_duration_sec']
        
        if flightListDatabase.isExtendedWithAircraftData():
            columnNameListToKeep = columnNameListToKeep + flightListDatabase.getAircraftExtendedListOfCharacteristics()
            
        df_trainFlightList = keepOnlyColumns( df_trainFlightList , columnNameListToKeep )
        
        logging.info( self.className + ": ---- train flight list = " + str ( list (df_trainFlightList ) ) )
        logging.info( self.className + ": ---- fuel train  = " + str ( list (self.FuelTrainDataframe ) ) )

        ''' extend in order to obtain flight start date time '''
        self.FuelTrainDataframe = pd.merge ( self.FuelTrainDataframe , df_trainFlightList , left_on='flight_id', right_on='flight_id', how='inner' )
        logging.info( str ( list ( self.FuelTrainDataframe ) ) )
        
        return True
    
    def extendFuelTrainWithFlightData (self):
        pass
        df = self.getFuelTrainDataframe()
        print("train fuel dataframe shape = " + str( df.shape ))
        
        listOfFlightListColumns = ['timestamp','aircraft_type_code', 'latitude', 'longitude', 'altitude', 'groundspeed', 'track', 
                                    'vertical_rate', 'mach', 'TAS', 'CAS', 'source']
        
        df[listOfFlightListColumns] = df.apply( extendFuelTrainWithFlightsData , axis = 1 )
        
        print ("shape after apply = " +  str ( df.shape ) )
        print ("final list = " +  str ( list ( df )))
        print ("final shape = " +  str (  df .shape ) ) 
        
        ''' drop columns with absolute date time instant '''
        df = dropUnusedColumns( df , ['fuel_burn_start','fuel_burn_end'])
        #print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        
        ''' convert flight data time stamp relative to flight start '''
        df['timestamp_relative_start'] = ( df['timestamp'] - df['takeoff']).dt.total_seconds()
        
        ''' drop absolute date time stamp '''
        df = dropUnusedColumns( df , ['timestamp','takeoff','flight_id'] )
        
        df = dropUnusedColumns( df , ['aircraft_type_code','source'])
        
        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        print(tabulate(df[-10:], headers='keys', tablefmt='grid' , showindex=False , ))
        
        #df = df.dropna()
        df.fillna(df.mean(), inplace=True) # Fill NaNs with column mean
        
        print (self.className + ": final shape = " +  str (  df .shape ) ) 
        self.FuelTrainDataframe = df
        return True
        
    def extendFuelRankWithFlightData(self):
        
        df = self.getFuelRankDataframe()
        print("Rank / Test fuel dataframe shape = " + str( df.shape ))
        
        listOfFlightListColumns = ['timestamp','aircraft_type_code', 'latitude', 'longitude', 'altitude', 'groundspeed', 'track', 
                               'vertical_rate', 'mach', 'TAS', 'CAS', 'source']
        
        df[listOfFlightListColumns] = df.apply( extendFuelRankWithFlightsData , axis = 1 )
        
        print ("shape after apply = " +  str ( df.shape ) )
        print ("final list = " +  str ( list ( df )))
        print ("final shape = " +  str (  df .shape ) ) 
        
        ''' drop columns with absolute date time instant '''
        df = dropUnusedColumns( df , ['fuel_burn_start','fuel_burn_end'])
        
        ''' convert flight data time stamp relative to flight start '''
        df['timestamp_relative_start'] = ( df['timestamp'] - df['takeoff']).dt.total_seconds()
        
        ''' drop absolute date time stamp '''
        df = dropUnusedColumns( df , ['timestamp','takeoff','flight_id'] )
        
        df = dropUnusedColumns( df , ['aircraft_type_code','source'])

        ''' replace nan with mean value '''
        df.fillna(df.mean(), inplace=True) # Fill NaNs with column mean
        #df = df.dropna()
        self.FuelRankDataframe = df
        return True

        