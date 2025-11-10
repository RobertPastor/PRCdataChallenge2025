'''
Created on 9 nov. 2025

@author: rober
'''

import sys
import platform
import logging
import pandas as pd
import os
import math
import numpy as np
from trajectory.aerocalc.airspeed import mach2tas , mach_alt2cas

from trajectory.Flights.FlightsReader import FlightsDatabase
from trajectory.FlightList.FlightListReader import FlightListDatabase
from trajectory.Guidance.WayPointFile import WayPoint , Airport

from trajectory.Fuel.FuelReader import FuelDatabase
from tabulate import tabulate
from trajectory.Environment.Airports.AirportDatabaseFile import AirportsDatabase

from trajectory.utils import keepOnlyColumns

class FlightsInterpolated(object):
    
    flight_id = None
    
    def __init__(self , train_rank , nbFlights , flight_id_filtered):
        self.train_rank = train_rank
        self.nbFlights = nbFlights
        self.flight_id_filtered = flight_id_filtered
        
        ''' dictionnary with takeoff and landed instant for each flight id '''
        self.TakeOffInstantDict = {}
        self.LandedInstantDict = {}
        
        self.OriginICAODict = {}
        self.DestinationICAODict = {}
        
        self.OriginLatitudeDegreesDict = {}
        self.OriginLongitudeDegreesDict = {}
        self.OriginElevationFeetDict = {}
        
        self.DestinationLatitudeDegreesDict = {}
        self.DestinationLongitudeDegreesDict = {}
        self.DestinationElevationFeetDict = {}
        
        self.airports = AirportsDatabase()
        assert self.airports.readWithPandas() == True
 

    def fill_Fuel_Frame_with_empty_columns_for_interpolation(self, df ):
    
        listOfColumns = ['latitude', 'longitude', 'altitude', 'groundspeed', 'track', 'vertical_rate', 'mach', 'TAS', 'CAS']
        # Add an empty column (filled with NaN)
        for columnName in listOfColumns:
            df[columnName] = np.nan
        
        return df

    def build_Fuel_Dataframe_with_start_end_timestamps( self, fuelTrainDataframe ):
        
        print ("-----focusing on the training flights data files -----")
        print ("-----extend fuel dataframe timestamps with fuel start and end timestamps  -----")
        print ("-----prepare for interpolating flights timestamps added from fuel start and end timestamps  -----")
        
        if ( self.train_rank == 'train'):
            fuelStartDataframe = fuelTrainDataframe.copy()
            listOfColumnNamesToKeep = ['flight_id', 'start']
            fuelStartDataframe = keepOnlyColumns( fuelStartDataframe, listOfColumnNamesToKeep)
            fuelStartDataframe = fuelStartDataframe.rename(columns={'start': 'timestamp'})
            print ( list ( fuelStartDataframe ))
            print ( fuelStartDataframe.shape  )
        #print ( tabulate( fuelStartDataframe[:10] , headers='keys', tablefmt='grid' , showindex=False , ))
        
            fuelEndDataframe = fuelTrainDataframe.copy()
            listOfColumnNamesToKeep = ['flight_id', 'end']
            fuelEndDataframe = keepOnlyColumns( fuelEndDataframe, listOfColumnNamesToKeep)
            fuelEndDataframe = fuelEndDataframe.rename(columns={'end': 'timestamp'})
            print ( list ( fuelEndDataframe ))
            print ( fuelEndDataframe.shape  )
            #print ( tabulate( fuelEndDataframe[:10] , headers='keys', tablefmt='grid' , showindex=False , ))
            ''' concat start wit end '''
            return pd.concat( [fuelStartDataframe , fuelEndDataframe] )
    
    def prepare_Fuel_for_interpolation(self ):

        if self.train_rank ==  'train':

            fuelDatabase = FuelDatabase(self.nbFlights)
            fuelTrainDataframe = fuelDatabase.readFuelTrainLite()
            print ( list ( fuelTrainDataframe ))
            ''' retreve fuel with only start end timestamp '''
            fuelTrainDataframe = self.build_Fuel_Dataframe_with_start_end_timestamps (fuelTrainDataframe)
            ''' drop duplicates ''' 
            fuelTrainDataframe = fuelTrainDataframe.drop_duplicates()
            print ( fuelTrainDataframe.shape )
            print ( list ( fuelTrainDataframe ) )
            
            ''' filter on one flight id '''
            if self.flight_id_filtered:
                print("--------- filtering on flight_id -----------")
                fuelTrainDataframe = fuelTrainDataframe[fuelTrainDataframe['flight_id'] == self.flight_id_filtered]
            ''' sort using timestamps '''
            fuelTrainDataframe = fuelTrainDataframe.sort_values(by='timestamp')

            #print(tabulate(fuelTrainDataframe[:10], headers='keys', tablefmt='grid' , showindex=False , ))
            #print(tabulate(fuelTrainDataframe[-10:], headers='keys', tablefmt='grid' , showindex=False , ))
            
            print("="*90)
            return fuelTrainDataframe
            
    def append_TakeOff_Landed_to_Fuel(self , flight_id):
        takeOffRow = pd.DataFrame ( { "flight_id" : flight_id , "timestamp" : self.TakeOffInstant})
        df = pd.concat( [self.fuelTrainDataframe , takeOffRow] , ignore_index=True) 
        landedRow = pd.DataFrame ( { "flight_id" : flight_id , "timestamp" : self.LandedInstant})
        df = pd.concat( [df , landedRow] , ignore_index=True)
        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        self.fuelTrainDataframe = df
        return True
    
    def computeOrigin2DestinationTrackAngleDegrees(self , flight_id ):
        if flight_id:
            
            self.flightList = FlightListDatabase()
            assert self.flightList.readTrainRankFlightListLite(self.train_rank) == True
            
            origin_icao = self.flightList.getOriginICAOairport(self.train_rank, flight_id)
            origin_airport = self.airports.getAirportFromICAOCode(origin_icao)
            assert (isinstance(origin_airport, Airport))
            print ( str ( origin_airport))
        
            destination_icao = self.flightList.getDestinationICAOairport(self.train_rank, flight_id)
            destination_airport = self.airports.getAirportFromICAOCode(destination_icao)
            assert (isinstance(destination_airport, Airport))
            print ( str ( destination_airport ) )

            bearing_angle_degrees = origin_airport.getBearingDegreesTo(destination_airport)
            print("from origin = " + origin_icao + " -> to destination = " + 
                  destination_icao + " -> bearing angle = " + str(bearing_angle_degrees) + " degrees")
            return bearing_angle_degrees

        raise ValueError("flight_id must be set !!!")

    def retrieve_FlightList_TakeOff_Landed(self , flight_id):
        
        self.flightList = FlightListDatabase()
        assert self.flightList.readTrainRankFlightListLite(self.train_rank) == True
 
        assert flight_id != None
        if flight_id:
            
            ''' origin ''' 
            
            takeOffInstant = self.flightList.getTakeOffInstant(self.train_rank , flight_id)
            self.TakeOffInstantDict [flight_id] = takeOffInstant
            #print("flight id = " + flight_id + " -> takeoff = " + str( takeOffInstant ) )

            origin_icao = self.flightList.getOriginICAOairport(self.train_rank , flight_id)
            self.OriginICAODict [flight_id] = origin_icao
            #print("flight id = " + flight_id + " -> origin airport = " + str( origin_icao ) )
            
            originLatitudeDegrees = self.flightList.getOriginAirportLatitudeDegrees(self.train_rank , flight_id)
            self.OriginLatitudeDegreesDict [flight_id] = originLatitudeDegrees
            #print("flight id = " + flight_id + " -> origin latitude = " + str( originLatitudeDegrees ) )
            
            originLongitudeDegrees = self.flightList.getOriginAirportLongitudeDegrees(self.train_rank , flight_id)
            self.OriginLongitudeDegreesDict [flight_id] = originLongitudeDegrees
            #print("flight id = " + flight_id + " -> origin longitude = " + str( originLongitudeDegrees ) )
            
            originElevationFeet = self.flightList.getOriginAirportElevationFeet(self.train_rank , flight_id)
            self.OriginElevationFeetDict [flight_id] = originElevationFeet
            #print("flight id = " + flight_id + " -> origin elevation feet = " + str( originElevationFeet ) )

            ''' destination '''
            
            landedInstant = self.flightList.getLandedInstant(self.train_rank , flight_id)
            self.LandedInstantDict [flight_id] = landedInstant
            #print("flight id = " + flight_id + " -> landed = " + str ( landedInstant ) )

            destination_icao = self.flightList.getDestinationICAOairport(self.train_rank , flight_id)
            self.DestinationICAODict [flight_id] = destination_icao
            #print("flight id = " + flight_id + " -> destination = " + str ( destination_icao ) )
            
            destinationLatitudeDegrees = self.flightList.getDestinationAirportLatitudeDegrees(self.train_rank , flight_id)
            self.DestinationLatitudeDegreesDict [flight_id] = destinationLatitudeDegrees
            #print("flight id = " + flight_id + " -> destination latitude = " + str( destinationLatitudeDegrees ) )
            
            destinationLongitudeDegrees = self.flightList.getDestinationAirportLongitudeDegrees(self.train_rank , flight_id)
            self.DestinationLongitudeDegreesDict [flight_id] = destinationLongitudeDegrees
            #print("flight id = " + flight_id + " -> destination longitude = " + str( destinationLongitudeDegrees ) )

            destinationElevationFeet = self.flightList.getDestinationAirportElevationFeet(self.train_rank , flight_id)
            self.DestinationElevationFeetDict [flight_id] = destinationElevationFeet
            #print("flight id = " + flight_id + " -> destination elevation feet = " + str( destinationElevationFeet ) )
            
            return True

        raise ValueError("flight id must be provided to retrieve a takeoff instant and a landed instant")
    
    def getFlightIdTakeOffInstant(self , flight_id ):
        return self.TakeOffInstantDict[flight_id]   
    
    def getFlightIdLandedInstant(self , flight_id ):
        return self.LandedInstantDict[flight_id]
    
    def getFlightIdOriginICAO(self , flight_id):
        return self.OriginICAODict[flight_id]
    
    def getFlightIdDestinationICAO(self, flight_id):
        return self.DestinationICAODict[flight_id]
    
    def compute_TAS_Knots_from_Mach(self , row):
        mach = row['mach']
        if ( mach is None) or math.isnan ( mach ) or ( mach == np.nan) or (mach < sys.float_info.epsilon):
            return row['TAS']
 
        aircraft_altitude_ft = row['altitude']
        if ( aircraft_altitude_ft is None) or math.isnan(aircraft_altitude_ft) or ( aircraft_altitude_ft == np.nan) or (aircraft_altitude_ft < sys.float_info.epsilon):
            return row['TAS']
        
        TAS = row['TAS']
        if not ( math.isnan(TAS) ) and ( TAS != np.nan) and (TAS > sys.float_info.epsilon):
            return TAS
        else:
            #TAS is empty
            try:
                return  mach2tas ( mach = mach , altitude = aircraft_altitude_ft , alt_units = 'ft')
            except ValueError:
                return np.nan
            
    def compute_CAS_Knots_from_Mach(self, row):
        mach = row['mach']
        if ( mach is None) or math.isnan ( mach ) or ( mach == np.nan) or (mach < sys.float_info.epsilon):
            return row['CAS']
        
        aircraft_altitude_ft = row['altitude']
        if ( aircraft_altitude_ft is None) or math.isnan(aircraft_altitude_ft) or ( aircraft_altitude_ft == np.nan) or (aircraft_altitude_ft < sys.float_info.epsilon):
            return row['CAS']
        
        CAS = row['CAS']
        if not ( math.isnan(CAS) ) and ( CAS != np.nan) and (CAS > sys.float_info.epsilon):
            return CAS
        else:
            #CAS is empty
            try:
                return  mach_alt2cas ( mach=mach , altitude=aircraft_altitude_ft , alt_units='ft')
            except ValueError:
                return np.nan

        
    ''' use python aerocal method to compute TAS and CAS from mach when TAS or CAS are null '''
    def computeMissingSpeeds(self, df):
        print("=========== compute missing speeds ===============")
        # Machine epsilon for single precision (32-bit)
        df['TAS'] = df.apply( self.compute_TAS_Knots_from_Mach, axis = 1)
        df['CAS'] = df.apply( self.compute_CAS_Knots_from_Mach , axis = 1)
        return df
    
    def interpolate_one_flight_data(self , flight_id, flightsDatabase):
        
        fileName = flight_id + ".parquet"
        
        flightDataframe = flightsDatabase.readOneFlightFileLite(self.train_rank, fileName)
        print ( flightDataframe.shape )
            
        ''' filter fuel on flight id and perform concat '''
        fuelDataframe = self.prepare_Fuel_for_interpolation( )
        
        ''' in the newly inserted records, need to add the aircraft ICAO code '''
        firstNonNullTypeCode = flightsDatabase.getFirstNonNullValueInColumn(flightDataframe, 'typecode')
        print(" ----> first non null Aircraft ICAO code value = " + firstNonNullTypeCode)
            
        
        ''' in order for the fuel start and end to exist as new rows in the flight dataframe '''                    
        print ( fuelDataframe.shape )
            
        ''' concat the dataframe '''
        flightDataframe = pd.concat ( [flightDataframe , fuelDataframe])
        print ( flightDataframe.shape )
            
        ''' set hard code values for type code and for source '''
        flightDataframe['typecode'] = firstNonNullTypeCode
        flightDataframe['source'] = 'interpolated'
        
        ''' show rows with nan in typecode column '''
        df_with_nan_in_typecode = flightDataframe[flightDataframe['typecode'].isna()]
        #print(tabulate(df_with_nan_in_typecode[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        assert df_with_nan_in_typecode.shape[0] == 0
        
        df_with_not_nan_in_mach = flightDataframe[flightDataframe['mach'].notna()]
        print(tabulate(df_with_not_nan_in_mach[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        
        ''' track is equivalent to heading if there is no wind '''
        bearingAngleFromOrigin2DestinationDegrees = self.computeOrigin2DestinationTrackAngleDegrees(flight_id)
        
        ''' append the takeoff and landed records '''
        takeOffRow = pd.DataFrame ( { "timestamp"   : [self.TakeOffInstantDict[flight_id]] , 
                                        "flight_id"     : [flight_id] , 
                                        "typecode"      : [firstNonNullTypeCode] , 
                                        "latitude"      : [self.OriginLatitudeDegreesDict[flight_id]] ,
                                        "longitude"     : [self.OriginLongitudeDegreesDict[flight_id]] ,
                                        "altitude"      : [self.OriginElevationFeetDict[flight_id]]  ,
                                        "groundspeed"   : [(0.0)],
                                        "track"         : [bearingAngleFromOrigin2DestinationDegrees],
                                        "vertical_rate" : [(0.0)],
                                        "mach"          : [(0.0)],
                                        "TAS"           : [(np.nan)],
                                        "CAS"           : [(np.nan)],
                                        "source"        : ["interpolated"]
                                        } )
        #print ( flightTrainDataframe.info() )
        #print ( takeOffRow.info() )
        ''' insert at the end index of dataframe '''
        flightDataframe = pd.concat( [ flightDataframe , takeOffRow])
        flightDataframe.reset_index(inplace=True)
        ''' track is equivalent to heading if there is no wind '''
        landedRow = pd.DataFrame ( { "timestamp"    : [self.LandedInstantDict[flight_id]] , 
                                        "flight_id"     : [flight_id] , 
                                        "typecode"      : [firstNonNullTypeCode] , 
                                        "latitude"      : [self.DestinationLatitudeDegreesDict[flight_id]] ,
                                        "longitude"     : [self.DestinationLongitudeDegreesDict[flight_id]] ,
                                        "altitude"      : [self.DestinationElevationFeetDict[flight_id]]  ,
                                        "groundspeed"   : [(0.0)],
                                        "track"         : [bearingAngleFromOrigin2DestinationDegrees],
                                        "vertical_rate" : [(0.0)],
                                        "mach"          : [(0.0)],
                                        "TAS"           : [(np.nan)],
                                        "CAS"           : [(np.nan)],
                                        "source"        : ["interpolated"]
                                        } )
        #print ( flightTrainDataframe.info() )
        #print ( landedRow.info() )
        flightDataframe = pd.concat( [ flightDataframe , landedRow])
        flightDataframe.reset_index(drop=True, inplace=True)
            
        ''' convert timestamps to UTC '''
        flightDataframe['timestamp'] = pd.to_datetime(flightDataframe['timestamp'], utc=True)
            
        ''' sort values acording to increasing timestamps '''
        flightDataframe = flightDataframe.sort_values(by='timestamp')
        flightDataframe.reset_index(drop=True, inplace=True)
            
        #print(tabulate(flightTrainDataframe[:10], headers='keys', tablefmt='grid' , showindex=True , ))
        #print(tabulate(flightTrainDataframe[-10:], headers='keys', tablefmt='grid' , showindex=True , ))
        
        print ( flightDataframe.info() )
        
        listOfColumnsToInterpolate = ['latitude','longitude','altitude','groundspeed','track','vertical_rate','mach','TAS','CAS']
        for columnToInterpolate in listOfColumnsToInterpolate:
            countOfNulls = flightDataframe[columnToInterpolate].isna().sum()
            print ( "column = " + columnToInterpolate + " count of nulls = " + 
                    str(countOfNulls) + " - versus = " + str(flightDataframe.shape[0]))

        flightDataframe = self.computeMissingSpeeds(flightDataframe)
        
        ''' show dataframe informations '''
        print ( flightDataframe.shape )
        print ( flightDataframe.info() )
            
        ''' main interpolation '''
        for columnToInterpolate in listOfColumnsToInterpolate:
            countOfNulls = flightDataframe[columnToInterpolate].isna().sum()
            print ( "column = " + columnToInterpolate + " count of nulls = " + 
                    str(countOfNulls) + " - versus = " + str(flightDataframe.shape[0]))
            
        print("=========================== interpolate ========================")
        flightDataframe = flightDataframe.interpolate(limit_direction='forward')
        
        flightDataframe = flightDataframe.drop( ['index'] , axis=1 )
 
        print(tabulate(flightDataframe[:10], headers='keys', tablefmt='grid' , showindex=True , ))
        #print(tabulate(flightTrainDataframe.describe().transpose(), headers='keys', tablefmt='grid' , showindex=False , ))
        print(tabulate(flightDataframe[-10:], headers='keys', tablefmt='grid' , showindex=True , ))

        print ( "-"*80 )

    def interpolate_all_flights_data(self):
        '''loop through the files '''
        flightsDatabase = FlightsDatabase()
        
        directory = flightsDatabase.getTrainRankFinalFlightsFolderPathStr( self.train_rank)
        count = 0
        for fileName in os.listdir(directory):
            if count < self.nbFlights:
                if fileName.endswith(".parquet"): # Filter specific file types
                    filePath = os.path.join(directory, fileName)
                    #print ( filePath )
                    flight_id = fileName.split(".")[0]
                    #print("flight_id = " + flight_id)
                    if flight_id and ( flight_id == self.flight_id_filtered):
                        ''' there is a filtered flight_id condition '''
                        self.interpolate_one_flight_data( flight_id , flightsDatabase )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("python version = " + platform.python_version())
    print("pandas version = " + pd. __version__)
    
    train_rank = "train"
    flight_id = "prc770864956"
    nbFlights = 10
    flightsInterpolated = FlightsInterpolated( train_rank, nbFlights , flight_id)
    #flightsInterpolated.prepare_Fuel_for_interpolation()
    
    assert flightsInterpolated.retrieve_FlightList_TakeOff_Landed(flight_id) == True
    ''' add takeoff and landed to the fuel start end dataframe '''
    
    fuelTrainDataframe = flightsInterpolated.interpolate_all_flights_data()


    