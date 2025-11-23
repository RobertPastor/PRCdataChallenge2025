'''
Created on 7 oct. 2025

@author: robert
'''

import logging
import os
import pandas as pd
from pathlib import Path

from trajectory.Environment.Airports.AirportDatabaseFile import AirportsDatabase
from trajectory.Flights.FlightsReader import FlightsDatabase
from trajectory.Guidance.GeographicalPointFile import GeographicalPoint
from trajectory.Environment.Constants import Meter2NauticalMiles
from trajectory.Guidance.WayPointFile import Airport

from trajectory.Environment.Aircrafts.FAAaircraftDatabaseFile import FaaAircraftDatabase

from tabulate import tabulate

initialHeaders = ['flight_date', 'aircraft_type', 'takeoff', 'landed', 'origin_icao', 'origin_name', 'destination_icao', 'destination_name', 'flight_id']

expectedHeaders = ['flight_date', 'aircraft_type', 'takeoff', 'landed', 'origin_icao', 'origin_name', 'destination_icao', 'destination_name', 'flight_id',
                   'origin_longitude_deg', 'origin_latitude_deg' , 'origin_elevation_ft' , 
                   'destination_longitude_deg' , 'destination_latitude_deg' , 'destination_elevation_ft',
                   'flight_distance_Nm' , 'flight_duration_sec' , 'year' , 'month' , 'day_of_year']

''' compute distance between departure and arrival airport using great circle '''
def computeFlightDistanceNauticalMiles( row ):
    departureAirportGeoPoint = GeographicalPoint ( LatitudeDegrees            = row['origin_latitude_deg'], 
                                                   LongitudeDegrees           = row['origin_longitude_deg'],
                                                   AltitudeMeanSeaLevelMeters = row['origin_elevation_ft'])
    
    arrivalAirportGeoPoint = GeographicalPoint ( LatitudeDegrees              = row['destination_latitude_deg'], 
                                                   LongitudeDegrees           = row['destination_longitude_deg'],
                                                   AltitudeMeanSeaLevelMeters = row['destination_elevation_ft'])
    return departureAirportGeoPoint.computeDistanceMetersTo(arrivalAirportGeoPoint) * Meter2NauticalMiles

''' compute flight duration in seconds '''
def computeFlightDurationSeconds( row ):
    return (row['landed'] - row['takeoff']).total_seconds() 

def extendAircraftCharacteristics( row , characteristicName , aircraftDatabase ):
    characteristicValue = aircraftDatabase.getGenericCaracteristic ( row['aircraft_type'] , characteristicName )
    return characteristicValue

class FlightListDatabase(object):
    className = ''
    
    def __init__(self  , train_rank_final ):
        self.className = self.__class__.__name__
        
        self.train_rank_final = train_rank_final
        
        self.fileNameFlightListTrain = "flightlist_train.parquet"
        self.fileNameFlightListRank =  "flightlist_rank.parquet"
        self.fileNameFlightListFinal =  "flightlist_final.parquet"
        #logging.info(self.fileNameFlightListRank)
        self.filesFolder = os.path.dirname(__file__)
        
        self.filePathFlightListTrain = os.path.join(self.filesFolder , self.fileNameFlightListTrain)
        self.filePathFlightListRank = os.path.join(self.filesFolder , self.fileNameFlightListRank)
        self.filePathFlightListFinal = os.path.join(self.filesFolder , self.fileNameFlightListFinal)
        
        logging.info("train -> " + self.filePathFlightListTrain)
        logging.info("rank -> " + self.filePathFlightListRank)
        logging.info("final -> " + self.filePathFlightListFinal)
        self.flightListExtendedWithAircraftData = False
        
        self.airportsDatabase = AirportsDatabase()
        assert self.airportsDatabase.readAsDict()
        
    def getOriginAirportElevationFeet(self , train_rank, flight_id ):
        
        origin_icao = self.getOriginAirportICAOcode(train_rank, flight_id)
        airport = self.airportsDatabase.getAirportFromICAOCode(origin_icao)
        assert (isinstance(airport, Airport))
        return airport.getElevationMSLFeet()
    
    def getOriginAirportLatitudeDegrees(self , train_rank, flight_id ):
        
        origin_icao = self.getOriginAirportICAOcode(train_rank, flight_id)
        airport = self.airportsDatabase.getAirportFromICAOCode(origin_icao)
        assert (isinstance(airport, Airport))
        return airport.getLatitudeDegrees()
        
    def getOriginAirportLongitudeDegrees(self , train_rank, flight_id ):
        
        origin_icao = self.getOriginAirportICAOcode(train_rank, flight_id)
        airport = self.airportsDatabase.getAirportFromICAOCode(origin_icao)
        assert (isinstance(airport, Airport))
        return airport.getLongitudeDegrees()
        
    def getDestinationAirportElevationFeet(self , train_rank, flight_id ):
        
        destination_icao = self.getDestinationICAOairport(train_rank, flight_id)
        airport = self.airportsDatabase.getAirportFromICAOCode(destination_icao)
        assert (isinstance(airport, Airport))
        return airport.getElevationMSLFeet()

    def getDestinationAirportLatitudeDegrees(self , train_rank, flight_id ):
        
        destination_icao = self.getDestinationICAOairport(train_rank, flight_id)
        airport = self.airportsDatabase.getAirportFromICAOCode(destination_icao)
        assert (isinstance(airport, Airport))
        return airport.getLatitudeDegrees()
        
    def getDestinationAirportLongitudeDegrees(self , train_rank, flight_id ):
        
        destination_icao = self.getDestinationICAOairport(train_rank, flight_id)
        airport = self.airportsDatabase.getAirportFromICAOCode(destination_icao)
        assert (isinstance(airport, Airport))
        return airport.getLongitudeDegrees()
        
    def getOriginAirportICAOcode(self , train_rank, flight_id ):
        pass
        if train_rank == 'train':
            origin_icao = self.TrainFlightListDataframe[self.TrainFlightListDataframe['flight_id'] == flight_id]["origin_icao"].iloc[0]
            return origin_icao
        elif train_rank == 'rank':
            origin_icao = self.RankFlightListDataframe[self.RankFlightListDataframe['flight_id'] == flight_id]["origin_icao"].iloc[0]
            return origin_icao
        else:
            origin_icao = self.FinalFlightListDataframe[self.FinalFlightListDataframe['flight_id'] == flight_id]["origin_icao"].iloc[0]
            return origin_icao
        
    def getOriginICAOairport(self , train_rank, flight_id):
        
        if train_rank == 'train':
            origin_icao = self.TrainFlightListDataframe[self.TrainFlightListDataframe['flight_id'] == flight_id]["origin_icao"].iloc[0]
            logging.info ( str ( origin_icao ) )
            return origin_icao
        elif train_rank == 'rank':
            origin_icao = self.RankFlightListDataframe[self.RankFlightListDataframe['flight_id'] == flight_id]["origin_icao"].iloc[0]
            logging.info ( str ( origin_icao ) )
            return origin_icao
        else:
            origin_icao = self.FinalFlightListDataframe[self.FinalFlightListDataframe['flight_id'] == flight_id]["origin_icao"].iloc[0]
            logging.info ( str ( origin_icao ) )
            return origin_icao
          
    
    def getDestinationICAOairport(self , train_rank, flight_id):
        if train_rank == 'train':
            destination_icao = self.TrainFlightListDataframe[self.TrainFlightListDataframe['flight_id'] == flight_id]["destination_icao"].iloc[0]
            logging.info ( str ( destination_icao ) )
            return destination_icao
        elif train_rank == 'rank':
            destination_icao = self.RankFlightListDataframe[self.RankFlightListDataframe['flight_id'] == flight_id]["destination_icao"].iloc[0]
            logging.info ( str ( destination_icao ) )
            return destination_icao
        else:
            destination_icao = self.FinalFlightListDataframe[self.FinalFlightListDataframe['flight_id'] == flight_id]["destination_icao"].iloc[0]
            logging.info ( str ( destination_icao ) )
            return destination_icao

    def getTakeOffInstant(self , train_rank, flight_id):
        if train_rank == 'train':
            takeoff = self.TrainFlightListDataframe[self.TrainFlightListDataframe['flight_id'] == flight_id]["takeoff"].iloc[0]
            #logging.info (self.className + " - takeoff instant " + str ( takeoff ) )
            return takeoff
        elif train_rank == 'rank':
            takeoff = self.RankFlightListDataframe[self.RankFlightListDataframe['flight_id'] == flight_id]["takeoff"].iloc[0]
            #logging.info (self.className + " - takeoff instant " + str ( takeoff ) )
            return takeoff
        else:
            takeoff = self.FinalFlightListDataframe[self.FinalFlightListDataframe['flight_id'] == flight_id]["takeoff"].iloc[0]
            #logging.info (self.className + " - takeoff instant " + str ( takeoff ) )
            return takeoff
        
    def getLandedInstant(self , train_rank, flight_id):
        if train_rank == 'train':
            landed = self.TrainFlightListDataframe[self.TrainFlightListDataframe['flight_id'] == flight_id]["landed"].iloc[0]
            #logging.info (self.className + " - landed instant " + str ( landed ) )
            return landed
        elif train_rank == 'rank':
            landed = self.RankFlightListDataframe[self.RankFlightListDataframe['flight_id'] == flight_id]["landed"].iloc[0]
            #logging.info (self.className + " - landed instant " + str ( landed ) )
            return landed
        else:
            landed = self.FinalFlightListDataframe[self.FinalFlightListDataframe['flight_id'] == flight_id]["landed"].iloc[0]
            #logging.info (self.className + " - landed instant " + str ( landed ) )
            return landed
        
    def getAircraftICAOcode(self , flight_id ):
        assert self.train_rank_final == 'train' or self.train_rank_final == 'rank' or self.train_rank_final == "final"
        ''' filter on aircraft_type '''
        if self.train_rank_final == 'train':
            df = self.TrainFlightListDataframe [self.TrainFlightListDataframe['aircraft_type'].notnull()]
        elif self.train_rank_final == 'rank':
            df = self.RankFlightListDataframe [self.RankFlightListDataframe['aircraft_type'].notnull()]
        else:
            df = self.FinalFlightListDataframe [self.RankFlightListDataframe['aircraft_type'].notnull()]
        ''' filter on flight_id '''
        df = df[df['flight_id'] == flight_id]
        print(tabulate(df[:1], headers='keys', tablefmt='grid' , showindex=False , ))
        ''' assumption aircraft_type has col index = 1 '''
        aircraft_index = df.columns.get_loc("aircraft_type")
        first_row_index = 0
        ac = df.iloc[first_row_index, aircraft_index]
        return ac
    
    def collectUniqueAircraftTypesFromTrainFlightList(self):
        
        assert self.extendTrainFlightListWithAircraftData() == True
        
        df = self.TrainFlightListDataframe [self.TrainFlightListDataframe['aircraft_type'].notnull()]
        aircraft_codes_list = df['aircraft_type'].unique().tolist()
        
        for aircraft_icao_code in aircraft_codes_list:
            print("aircraft ICAO code = " , str(aircraft_icao_code) )
            if ( self.faaAircraftDatabase.isICAOcodeExisting(aircraft_icao_code)):
                print (" ----> aircraft = " , str(aircraft_icao_code) , " is in Aircraft Database")
        #logging.info( df.head ())
        return True
    
    def checkTrainFlightListHeaders(self):
        return (set(self.TrainFlightListDataframe) == set(expectedHeaders))
        
    def checkRankFligthListHeaders(self):
        return (set(self.RankFlightListDataframe) == set(expectedHeaders))
    
    def getTrainRankFinalFlightListDataframe(self , train_rank_final ):
        if train_rank_final == 'train':
            return self.TrainFlightListDataframe
        elif train_rank_final == 'rank':
            return self.RankFlightListDataframe
        else:
            return self.FinalFlightListDataframe
    
    def getTrainFlightListDataframe(self , train_rank):
        return self.TrainFlightListDataframe
    
    def getRankFlightListDataframe(self):
        return self.RankFlightListDataframe
        
    def readTrainFlightList(self ):
        logging.info(self.filePathFlightListTrain)
        
        directory = Path(self.filesFolder)
        logging.info(directory)
        
        file = Path(self.filePathFlightListTrain)
        
        if directory.is_dir() and file.is_file():
            
            self.TrainFlightListDataframe = pd.read_parquet ( self.filePathFlightListTrain )
            
            assert list(self.TrainFlightListDataframe) == initialHeaders
            
            ''' convert to datetime UTC '''
            self.TrainFlightListDataframe["takeoff"] = pd.to_datetime(self.TrainFlightListDataframe["takeoff"], utc=True)
            self.TrainFlightListDataframe["landed"] = pd.to_datetime(self.TrainFlightListDataframe["landed"], utc=True)
            
            assert self.extendTrainFlightListWithAirportData()
            
            ''' compute distance nautical miles between departure airport and arrival airport '''
            self.TrainFlightListDataframe["flight_distance_Nm"] = self.TrainFlightListDataframe.apply ( computeFlightDistanceNauticalMiles , axis = 1)
            ''' compute flight duration between departure airport and arrival airport '''
            self.TrainFlightListDataframe["flight_duration_sec"] = self.TrainFlightListDataframe.apply ( computeFlightDurationSeconds , axis = 1)
            
            # Extract year
            self.TrainFlightListDataframe['year'] = self.TrainFlightListDataframe['takeoff'].dt.year
            self.TrainFlightListDataframe['month'] = self.TrainFlightListDataframe['takeoff'].dt.month
            
            ''' extract the day number of the year '''
            self.TrainFlightListDataframe['day_of_year'] = self.TrainFlightListDataframe['takeoff'].dt.dayofyear
            ''' extend flight list with aircraft data '''
            assert self.extendTrainFlightListWithAircraftData()

            logging.info ( self.className +  str(self.TrainFlightListDataframe.shape ) )
            logging.info ( self.className +  str(  list ( self.TrainFlightListDataframe)) )
                        
            #logging.info (self.className + str( self.TrainFlightListDataframe.head(10) ) )
            return True
        else:
            logging.error(self.className + " : it is a directory - {0}".format(self.filesFolder))
            logging.error (self.className + " : it is a file - {0}".format(self.filePathFlightListTrain))

            return False
        
    def readTrainRankFinalFlightListLite(self , train_rank_final):
        assert train_rank_final == 'train' or train_rank_final == 'rank' or train_rank_final == 'final'
        if train_rank_final == 'train':
            logging.info(self.filePathFlightListTrain)
            
            directory = Path(self.filesFolder)
            logging.info(directory)
            
            file = Path(self.filePathFlightListTrain)
            if directory.is_dir() and file.is_file():
                
                logging.info (self.className + " : it is a directory - {0}".format(self.filesFolder))
                logging.info (self.className + " : it is a file - {0}".format(self.filePathFlightListTrain))
                
                self.TrainFlightListDataframe = pd.read_parquet ( self.filePathFlightListTrain )
                return True
            
        elif train_rank_final == 'rank':
            logging.info(self.filePathFlightListRank)
            
            directory = Path(self.filesFolder)
            logging.info(directory)
            
            file = Path(self.filePathFlightListRank)
            if directory.is_dir() and file.is_file():
                
                logging.info (self.className + " : it is a directory - {0}".format(self.filesFolder))
                logging.info (self.className + " : it is a file - {0}".format(self.filePathFlightListRank))
                
                self.RankFlightListDataframe = pd.read_parquet ( self.filePathFlightListRank )
                return True
        else:
            ''' final '''
            logging.info(self.filePathFlightListFinal)
            
            directory = Path(self.filesFolder)
            logging.info(directory)
            
            file = Path(self.filePathFlightListFinal)
            if directory.is_dir() and file.is_file():
                
                logging.info (self.className + " : it is a directory - {0}".format(self.filesFolder))
                logging.info (self.className + " : it is a file - {0}".format(self.filePathFlightListFinal))
                
                self.FinalFlightListDataframe = pd.read_parquet ( self.filePathFlightListFinal )
                return True

        return False
        
    def readTrainFlightListLite(self):
        
        logging.info(self.filePathFlightListTrain)
        
        directory = Path(self.filesFolder)
        logging.info(directory)
        
        file = Path(self.filePathFlightListTrain)
        if directory.is_dir() and file.is_file():
            
            logging.info (self.className + "it is a directory - {0}".format(self.filesFolder))
            logging.info (self.className + "it is a file - {0}".format(self.filePathFlightListTrain))
            
            self.TrainFlightListDataframe = pd.read_parquet ( self.filePathFlightListTrain )
            
            assert list(self.TrainFlightListDataframe) == initialHeaders
            return True
        return False
        
    def readRankFlightListLite(self):
        
        logging.info(self.filePathFlightListRank)
        
        directory = Path(self.filesFolder)
        logging.info(directory)
        
        file = Path(self.filePathFlightListRank)
        if directory.is_dir() and file.is_file():
            
            logging.info (self.className + "it is a directory - {0}".format(self.filesFolder))
            logging.info (self.className + "it is a file - {0}".format(self.filePathFlightListRank))
            
            self.RankFlightListDataframe = pd.read_parquet ( self.filePathFlightListRank )
            
            logging.info( str ( list(self.RankFlightListDataframe) ) )
            logging.info( str ( initialHeaders) )
            
            assert list(self.RankFlightListDataframe) == initialHeaders
            return True
            
        assert False

    def readRankFlightList(self ):
        logging.info(self.filePathFlightListRank)
        
        directory = Path(self.filesFolder)
        logging.info(directory)
        
        file = Path(self.filePathFlightListRank)
        
        if directory.is_dir() and file.is_file():
            
            logging.info (self.className + "it is a directory - {0}".format(self.filesFolder))
            logging.info (self.className + "it is a file - {0}".format(self.filePathFlightListRank))
            
            self.RankFlightListDataframe = pd.read_parquet ( self.filePathFlightListRank )
            
            assert list(self.RankFlightListDataframe) == initialHeaders

            ''' convert to datetime UTC '''
            self.RankFlightListDataframe["takeoff"] = pd.to_datetime(self.RankFlightListDataframe["takeoff"], utc=True)
            self.RankFlightListDataframe["landed"] = pd.to_datetime(self.RankFlightListDataframe["landed"], utc=True)
            
            assert self.extendRankFlightListWithAirportData()
            
            ''' compute distance nautical miles between departure airport and arrival airport '''
            self.RankFlightListDataframe["flight_distance_Nm"] = self.RankFlightListDataframe.apply ( computeFlightDistanceNauticalMiles , axis = 1)
            ''' compute flight duration between departure airport and arrival airport '''
            self.RankFlightListDataframe["flight_duration_sec"] = self.RankFlightListDataframe.apply ( computeFlightDurationSeconds , axis = 1)
            
            self.RankFlightListDataframe['year'] = self.RankFlightListDataframe['takeoff'].dt.year
            self.RankFlightListDataframe['month'] = self.RankFlightListDataframe['takeoff'].dt.month
            
            ''' extract the day number of the year '''
            self.RankFlightListDataframe['day_of_year'] = self.RankFlightListDataframe['takeoff'].dt.dayofyear
            
            #assert self.extendRankFlightListWithAircraftData()

            logging.info ( str(self.RankFlightListDataframe.shape ) )
            logging.info ( str(  list ( self.RankFlightListDataframe)) )
            
            #logging.info ( self.RankFlightListDataframe.head(10) )
            return True
        else:
            return False
    
    
    
    def collectUniqueAirports(self):
        
        logging.info(self.className + ": ------- collect Unique Airports -------- ")
        
        self.train = self.TrainFlightListDataframe [self.TrainFlightListDataframe['origin_icao'].notnull()]
        dfTrain = self.train['origin_icao']
        dfTrain = dfTrain.rename( 'airport_icao' )
        #logging.info ( self.className + ": columns = " + str(  list ( dfTrain)) )

        #logging.info( dfTrain.head (100 ))
        logging.info ( self.className + ": shape = " +str(dfTrain.shape ) )
        
        self.rank  = self.RankFlightListDataframe [self.RankFlightListDataframe['destination_icao'].notnull()]
        dfRank = self.rank ['destination_icao']
        dfRank = dfRank.rename( 'airport_icao' )

        #logging.info ( self.className + ": columns = " + str(  list ( dfRank )) )
        #logging.info( dfRank.head (100 ))
        logging.info (self.className +": --- shape = " + str(dfRank.shape ) )
        
        dfConcat = pd.concat( [dfTrain , dfRank] )
        #logging.info( dfConcat )
        
        logging.info ( str(dfConcat.shape ) )
        dfConcat = dfConcat.unique ( )
        
        logging.info (self.className + ": size of unique list of airports : " + str(dfConcat.shape ) )
        #logging.info( dfConcat.head(100))
        
    def extendRankFlightListWithAirportData(self):
        
        logging.info(self.className + ": ---------- extend Flight List With Airport Data ---- ")
        
        airportsDb = AirportsDatabase()
        assert airportsDb.readWithPandas() == True
        assert airportsDb.checkHeaders() == True
        
        airportsDataframe = airportsDb.getAirportsDataframe()
        
        logging.info(self.className + " - " + str ( list ( airportsDataframe ) ) )
        logging.info(self.className + " - " + str ( list ( self.RankFlightListDataframe ) ) )
        logging.info(self.className + " - " + str (  self.RankFlightListDataframe.shape ) )
        
        rankFlightListDataframeRowCount = self.RankFlightListDataframe.shape[0]
        initialFlightListDataframe = self.getRankFlightListDataframe()
        
        ''' extend origin icao '''
        merged_df = pd.merge ( self.RankFlightListDataframe , airportsDataframe , 
                                left_on='origin_icao', right_on='airport_icao', how='inner' )
        logging.info( merged_df.shape )
        ''' not merged airports '''
        
        notInMergedDf = initialFlightListDataframe[~initialFlightListDataframe.isin(merged_df.to_dict(orient='list')).all(axis=1)]
        ''' records in initial flight list not anymore in the merged dataframe merged wuth airorts '''
        print(tabulate(notInMergedDf[:40], headers='keys', tablefmt='grid' , showindex=True , ))
        
        assert merged_df.shape[0] == rankFlightListDataframeRowCount
        logging.info( self.className + "- " +str ( list ( merged_df ) ) )

        ''' suppress icao '''
        merged_df = merged_df.drop( ['airport_icao'] , axis=1 )
        
        ''' rename extended columns '''
        merged_df = merged_df.rename(
            columns= {'airport_latitude_deg' :'origin_latitude_deg',
                      'airport_longitude_deg':'origin_longitude_deg',
                      'airport_elevation_ft' :'origin_elevation_ft'})
        
        logging.info( self.className + " - " + str ( list ( merged_df ) ) )
        
        ''' extend destination icao airport '''
        merged_df = pd.merge ( merged_df , airportsDataframe , left_on='destination_icao', right_on='airport_icao', how='inner' )
        assert merged_df.shape[0] == rankFlightListDataframeRowCount
        
        ''' suppress icao '''
        merged_df = merged_df.drop( ['airport_icao'] , axis=1 )
        
        assert merged_df.shape[0] == rankFlightListDataframeRowCount
        ''' rename extended columns '''
        merged_df = merged_df.rename(
            columns= {'airport_latitude_deg'  :'destination_latitude_deg',
                      'airport_longitude_deg' :'destination_longitude_deg',
                      'airport_elevation_ft'  :'destination_elevation_ft'})
        #logging.info( str ( list ( df_flightListExtendedWithAirportData ) ) )
        
        self.extendedRankFlightListDataframe = merged_df
        self.RankFlightListDataframe = merged_df
        
        return True
        
    def extendTrainFlightListWithAirportData(self):
        
        logging.info(self.className + ": ---------- extend Flight List With Airport Data ---- ")
        
        airportsDb = AirportsDatabase()
        assert airportsDb.readWithPandas() == True
        assert airportsDb.checkHeaders() == True
        
        airportsDataframe = airportsDb.getAirportsDataframe()
        
        logging.info( str ( list ( airportsDataframe ) ) )
        logging.info( str ( list ( self.TrainFlightListDataframe ) ) )
        
        ''' extend origin icao '''
        df_merged = pd.merge ( self.TrainFlightListDataframe , airportsDataframe , 
                                                          left_on='origin_icao', right_on='airport_icao', how='inner' )
        logging.info( str ( list ( df_merged ) ) )

        ''' suppress airport icao '''
        df_merged = df_merged.drop( ['airport_icao'] , axis=1 )
        
        ''' rename extended columns '''
        df_merged = df_merged.rename(
            columns= {'airport_latitude_deg' :'origin_latitude_deg',
                      'airport_longitude_deg':'origin_longitude_deg',
                      'airport_elevation_ft' :'origin_elevation_ft'})
        logging.info( str ( list ( df_merged ) ) )
        
        ''' extend destination icao '''
        df_merged = pd.merge ( df_merged , airportsDataframe , 
                                left_on='destination_icao', right_on='airport_icao', how='inner' )
        
        ''' suppress icao '''
        df_merged = df_merged.drop( ['airport_icao'] , axis=1 )
        ''' rename extended columns '''
        df_merged = df_merged.rename(
            columns= {'airport_latitude_deg':'destination_latitude_deg',
                      'airport_longitude_deg':'destination_longitude_deg',
                      'airport_elevation_ft':'destination_elevation_ft'})
        
        logging.info( str ( list ( df_merged ) ) ) 
        
        self.extendedTrainFlightListDataframe = df_merged
        self.TrainFlightListDataframe = df_merged

        return True
        #logging.info ( df_flightListExtendedWithAirportData.head(10) )
        
    def getTrainFlightDataWithFlightListData(self):
        return self.TrainFlightDataWithFlightListData
        
    def extendTrainFlightDataWithFlightListData(self):
        '''
        loop through the flight ids in the flight list 
        using the flight id , open the flight data file and extend the flight data with the data from the flight list
        '''
        
        flightsDatabase = FlightsDatabase()
        count = 0
        df_concat = None
        
        for index, row in self.TrainFlightListDataframe.iterrows():
            print(f"----- Index: {index} , Name: { row['flight_id'] } ----- ")
            flightName = row['flight_id']
            if count < 10:
                df_flight = flightsDatabase.readOneTrainFile(flightName)
                
                df_join = pd.merge ( df_flight , self.TrainFlightListDataframe , on = 'flight_id' , how = "inner")
                #logging.info("df_shape columns = " + str ( list ( df_join ) ) )
                if count == 0:
                    df_concat = df_join
                else:
                    df_concat = pd.concat( [df_concat, df_join], ignore_index=True)
                    
                #logging.info ("df_concat shape = " +  str(df_concat.shape ) )

                count = count + 1
            else:
                break
            
        self.TrainFlightDataWithFlightListData = df_concat
        df_concat.sample(5)
        return True
    
    def isExtendedWithAircraftData(self):
        return self.flightListExtendedWithAircraftData
    
    def getAircraftExtendedListOfCharacteristics(self):
        if (self.flightListExtendedWithAircraftData == True):
            return self.faaAircraftDatabase.getListOfExtendedCharacteristics()
        else:
            return []
    
    def extendTrainFlightListWithAircraftData(self):
        self.flightListExtendedWithAircraftData = False
        self.faaAircraftDatabase = FaaAircraftDatabase()
        assert self.faaAircraftDatabase.exists()
        
        if ( self.faaAircraftDatabase.read()):
            
            for extendedCharacteristic in self.faaAircraftDatabase.getListOfExtendedCharacteristics():
                self.TrainFlightListDataframe[extendedCharacteristic] = self.TrainFlightListDataframe.apply ( extendAircraftCharacteristics , axis = 1 , args = ( extendedCharacteristic , self.faaAircraftDatabase ))
        
        self.flightListExtendedWithAircraftData = True
        print (self.className + ": ---- train flight list data frame = " , str ( list ( self.TrainFlightListDataframe )))
        return True
    
    def extendRankFlightListWithAircraftData(self):
        self.flightListExtendedWithAircraftData = False
        self.faaAircraftDatabase = FaaAircraftDatabase()
        assert self.faaAircraftDatabase.exists()
        
        if ( self.faaAircraftDatabase.read()):
            
            for extendedCharacteristic in self.faaAircraftDatabase.getListOfExtendedCharacteristics():
                self.RankFlightListDataframe[extendedCharacteristic] = self.RankFlightListDataframe.apply ( extendAircraftCharacteristics , axis = 1 , args = ( extendedCharacteristic , self.faaAircraftDatabase ))
            
        self.flightListExtendedWithAircraftData = True
        print (self.className + ": ---- rank flight list data frame = " ,  str ( list ( self.RankFlightListDataframe )))
        return True