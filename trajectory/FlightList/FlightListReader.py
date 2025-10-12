'''
Created on 7 oct. 2025

@author: robert
'''

import logging
import os
import pandas as pd
from pathlib import Path
from tabulate import tabulate

from trajectory.Environment.AirportsDataChallenge.AirportsDataChallengeDatabaseFile import AirportsDataChallengeDatabase
from trajectory.Flights.FlightsReader import FlightsDatabase

expectedHeaders = ['flight_date', 'aircraft_type', 'takeoff', 'landed', 'origin_icao', 'origin_name', 'destination_icao', 'destination_name', 'flight_id',
                   'origin_longitude', 'origin_latitude' , 'origin_elevation' , 'destination_longitude' , 'destination_latitude' , 'destination_elevation']

class FlightListDatabase(object):
    className = ''
    
    def __init__(self):
        self.className = self.__class__.__name__
        
        self.fileNameFlightListTrain = "flightlist_train.parquet"
        #logging.info(self.fileNameFlightListTrain)
        
        self.fileNameFlightListRank =  "flight_list_rank.parquet"
        #logging.info(self.fileNameFlightListRank)
        
        self.filesFolder = os.path.dirname(__file__)
        
        self.filePathFlightListTrain = os.path.join(self.filesFolder , self.fileNameFlightListTrain)
        #logging.info(self.filePathFlightListTrain)
        
        self.filePathFlightListRank = os.path.join(self.filesFolder , self.fileNameFlightListRank)
        #logging.info(self.filePathFlightListRank)
        
    def checkTrainFlightListHeaders(self):
        return (set(self.TrainFlightListDataframe) == set(expectedHeaders))
        
    def checkRankFligthListHeaders(self):
        return (set(self.RankFlightListDataframe) == set(expectedHeaders))
    
    def getTrainFlightListDataframe(self):
        return self.TrainFlightListDataframe
    
    def getRankFlightListDataframe(self):
        return self.RankFlightListDataframe
        
    def readTrainFlightList(self ):
        logging.info(self.filePathFlightListTrain)
        
        directory = Path(self.filesFolder)
        logging.info(directory)
        
        file = Path(self.filePathFlightListTrain)
        
        if directory.is_dir() and file.is_file():
            
            logging.info (self.className + "it is a directory - {0}".format(self.filesFolder))
            logging.info (self.className + "it is a file - {0}".format(self.filePathFlightListTrain))
            
            self.TrainFlightListDataframe = pd.read_parquet ( self.filePathFlightListTrain )
            
            ''' convert to datetime UTC '''
            self.TrainFlightListDataframe["takeoff"] = pd.to_datetime(self.TrainFlightListDataframe["takeoff"], utc=True)
            self.TrainFlightListDataframe["landed"] = pd.to_datetime(self.TrainFlightListDataframe["landed"], utc=True)
            
            assert self.extendTrainFlightListWithAirportData()

            logging.info ( self.className +  str(self.TrainFlightListDataframe.shape ) )
            logging.info ( self.className +  str(  list ( self.TrainFlightListDataframe)) )
            
            #logging.info (self.className + str( self.TrainFlightListDataframe.head(10) ) )
            return True
        else:
            logging.error(self.className + "it is a directory - {0}".format(self.filesFolder))
            logging.error (self.className + "it is a file - {0}".format(self.filePathFlightListTrain))

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
            ''' convert to datetime UTC '''
            self.RankFlightListDataframe["takeoff"] = pd.to_datetime(self.RankFlightListDataframe["takeoff"], utc=True)
            self.RankFlightListDataframe["landed"] = pd.to_datetime(self.RankFlightListDataframe["landed"], utc=True)
            
            assert self.extendRankFlightListWithAirportData()

            logging.info ( str(self.RankFlightListDataframe.shape ) )
            logging.info ( str(  list ( self.RankFlightListDataframe)) )
            
            #logging.info ( self.RankFlightListDataframe.head(10) )
        
            return True
        else:
            return False
    
    def collectUniqueAircraftTypes(self):
        pass
        df = self.TrainFlightListDataframe [self.TrainFlightListDataframe['aircraft_type'].notnull()]
        
        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        #logging.info( df.head ())
    
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
        
        airportsDb = AirportsDataChallengeDatabase()
        assert airportsDb.read() == True
        assert airportsDb.checkHeaders() == True
        
        airportsDataframe = airportsDb.getAirportsDataframe()
        
        logging.info( str ( list ( airportsDataframe ) ) )
        logging.info( str ( list ( self.RankFlightListDataframe ) ) )
        
        ''' extend origin icao '''
        df_flightListExtendedWithAirportData = pd.merge ( self.RankFlightListDataframe , airportsDataframe , left_on='origin_icao', right_on='icao', how='inner' )
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
        #logging.info( str ( list ( df_flightListExtendedWithAirportData ) ) )
        
        self.extendedTrainFlightListDataframe = df_flightListExtendedWithAirportData
        self.RankFlightListDataframe = df_flightListExtendedWithAirportData

        return True
        
    def extendTrainFlightListWithAirportData(self):
        
        logging.info(self.className + ": ---------- extend Flight List With Airport Data ---- ")
        
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
        #logging.info( str ( list ( df_flightListExtendedWithAirportData ) ) )
        
        self.extendedTrainFlightListDataframe = df_flightListExtendedWithAirportData
        self.TrainFlightListDataframe = df_flightListExtendedWithAirportData

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
            if count < 100:
                df_flight = flightsDatabase.readOneFile(flightName)
                
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