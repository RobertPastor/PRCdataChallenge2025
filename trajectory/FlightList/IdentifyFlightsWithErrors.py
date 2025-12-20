'''
Created on 20 déc. 2025

@author: robert

'''

import logging
import unittest
import pandas as pd
from trajectory.FlightList.FlightListReader import FlightListDatabase
from tabulate import tabulate

from trajectory.Environment.Airports.AirportDatabaseFile import AirportsDatabase
from trajectory.Guidance.WayPointFile import Airport
from trajectory.Guidance.GeographicalPointFile import GeographicalPoint
from trajectory.Environment.Constants import Meter2NauticalMiles

def computeFlightDistanceNauticalMiles( origin_latitude_deg , origin_longitude_deg ,  origin_elevation_ft ,
                                        destination_latitude_deg , destination_longitude_deg ,  destination_elevation_ft):
    departureAirportGeoPoint = GeographicalPoint ( LatitudeDegrees            = origin_latitude_deg, 
                                                       LongitudeDegrees           = origin_longitude_deg,
                                                       AltitudeMeanSeaLevelMeters = origin_elevation_ft)
        
    arrivalAirportGeoPoint = GeographicalPoint ( LatitudeDegrees              = destination_latitude_deg, 
                                                       LongitudeDegrees           = destination_longitude_deg,
                                                       AltitudeMeanSeaLevelMeters = destination_elevation_ft)
    
    return departureAirportGeoPoint.computeDistanceMetersTo(arrivalAirportGeoPoint) 


#============================================
class Test_Main(unittest.TestCase):
    
        
    def test_Rank_Flight_Ids(self):
        
        self.airportsDatabase = AirportsDatabase()
        assert self.airportsDatabase.readAsDict()
        
        print("------------test_main_two----------------")

        train_rank_final = "rank"
        flightList = FlightListDatabase(train_rank_final)
        
        assert flightList.readRankFlightListLite()
        flight_ids_list = flightList.collectFlightIdsForOneAircraftType('A320')
        print ( flight_ids_list )
        print ( len( flight_ids_list ))
        
        max_distance_meters = 0.0
        max_flight_id = "prcxxxxxxx"
        max_origin_icao = "AAA"
        max_destination_icao = "AAA"
        for flight_id in flight_ids_list:
            
            print("-----------------------")
            print (flight_id)
            
            origin_icao = flightList.getOriginAirportICAOcode(train_rank_final, flight_id)
            print(origin_icao)
            destination_icao = flightList.getDestinationICAOairport(train_rank_final, flight_id)
            print(destination_icao)
            print( flight_id + " - origin = " + origin_icao + " - destination = " + destination_icao)
            
            origin_airport = self.airportsDatabase.getAirportFromICAOCode(origin_icao)
            assert (isinstance(origin_airport, Airport))
            
            destination_airport = self.airportsDatabase.getAirportFromICAOCode(destination_icao)
            assert (isinstance(destination_airport, Airport))

            origin_elevation_meters = origin_airport.getElevationMSLFeet()
            origin_latitude_degrees = origin_airport.getLatitudeDegrees()
            origin_longitude_degrees = origin_airport.getLongitudeDegrees()
            
            destination_elevation_meters = destination_airport.getElevationMSLFeet()
            destination_latitude_degrees = destination_airport.getLatitudeDegrees()
            destination_longitude_degrees = destination_airport.getLongitudeDegrees()
            
            distance_meters = computeFlightDistanceNauticalMiles( origin_latitude_degrees , origin_longitude_degrees , origin_elevation_meters , \
                                                                  destination_latitude_degrees , destination_longitude_degrees , destination_elevation_meters)
            print(distance_meters)
            if ( distance_meters > max_distance_meters):
                max_distance_meters = distance_meters
                max_flight_id = flight_id
                max_origin_icao = origin_icao
                max_destination_icao = destination_icao
                
        print("-------------")
        print(" flight id = " + max_flight_id + " - origin = " + max_origin_icao + " - destination = " + destination_icao)
        print(" flight id = " + max_flight_id + " - max distance = {0} meters".format(max_distance_meters))

        
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()