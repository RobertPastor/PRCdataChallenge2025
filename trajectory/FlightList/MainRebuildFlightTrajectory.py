'''
Created on 22 nov. 2025

@author: robert
'''

import platform
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import logging
from tabulate import tabulate

from trajectory.FlightList.FlightListReader import FlightListDatabase
from trajectory.Flights.FlightsReader import FlightsDatabase
from trajectory.Guidance.WayPointFile import Airport
from trajectory.Environment.Airports.AirportDatabaseFile import AirportsDatabase
import matplotlib.pyplot as plt

from trajectory.Environment.Runways.RunWaysDatabaseFile import RunWayDataBase

class FlightTrajectoryReBuild(object):
    
    flight_id  = "prc812317830"
    train_rank_final = "rank"
    columnNameList = ['latitude', 'longitude','altitude','groundspeed','track','vertical_rate', 'mach', 'TAS', 'CAS']

    def __init__(self , train_rank_final , flight_id ):
        self.train_rank_final
        assert self.train_rank_final == "train" or self.train_rank_final == "rank" or self.train_rank_final == "final"
    
    def extractAircraftICAOcode(self):
        return self.flightListDatabase.getAircraftICAOcode(self.flight_id)
    
    def extractDepartureAirport(self):
        return self.flightListDatabase.getOriginAirportICAOcode(self.train_rank_final , self.flight_id)
    
    def extractDestinationAirport(self):
        return self.flightListDatabase.getDestinationICAOairport(self.train_rank_final , self.flight_id)
    
    def extractTakeOffInstant(self):
        return self.flightListDatabase.getTakeOffInstant( self.train_rank_final , self.flight_id )
    
    def extractLandedInstant(self):
        return self.flightListDatabase.getLandedInstant( self.train_rank_final , self.flight_id )
    
    def readFlightListDatabase(self):
        logging.info("---------Read Flight List <<" + self.train_rank_final +">> ------------")

        self.flightListDatabase = FlightListDatabase(self.train_rank_final)
        if self.flightListDatabase.readTrainRankFinalFlightListLite(self.train_rank_final):
            logging.info("train rank final flight list read correctly")
            
            self.flightListDataframe = self.flightListDatabase.getTrainRankFinalFlightListDataframe(self.train_rank_final)
            print(tabulate(self.flightListDataframe[:10], headers='keys', tablefmt='grid' , showindex=False , ))
            logging.info(self.flightListDataframe.shape)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    print("python version = " + platform.python_version())
    print("tensorflow version = " + tf.__version__)
    print("pandas version = " + pd. __version__)
    print("numpy version = " + np. __version__)
        
    logging.basicConfig(level=logging.DEBUG)
    train_rank_final = "rank"
    flight_id = "prc812317830"
    
    flightTrajectoryReBuild = FlightTrajectoryReBuild( train_rank_final , flight_id)
    flightTrajectoryReBuild.readFlightListDatabase()
    
    print ( "aircraft = " + flightTrajectoryReBuild.extractAircraftICAOcode() )
    ac = flightTrajectoryReBuild.extractAircraftICAOcode()
    print ( "aircraft = " + ac  )

    originAirportICAOcode = flightTrajectoryReBuild.extractDepartureAirport()
    print( "origin airport = " +   originAirportICAOcode)  
    destinationAirportICAOcode =   flightTrajectoryReBuild.extractDestinationAirport()
    print( "destination airport = "+  destinationAirportICAOcode)
    
    print ( flightTrajectoryReBuild.extractTakeOffInstant())
    print ( flightTrajectoryReBuild.extractLandedInstant())
    
    runWaysDataBase = RunWayDataBase()
    runWaysDataBase.read()
    print (f"are there runways for the airport -> {originAirportICAOcode}  -> {runWaysDataBase.hasRunWays(originAirportICAOcode)}" )
    print (f"are there runways for this airport -> {destinationAirportICAOcode}  -> {runWaysDataBase.hasRunWays(destinationAirportICAOcode)}" )
    
    print("-"*90)
    for runway in runWaysDataBase.getRunWays(originAirportICAOcode):
        print(runway)
    print("-"*90)
    for runway in runWaysDataBase.getRunWays(destinationAirportICAOcode):
        print(runway)
        
    airportsDatabase = AirportsDatabase()
    airportsDatabase.readWithPandas()
    
    assert airportsDatabase.isAirportICAOcodeInDB(originAirportICAOcode)
    assert airportsDatabase.isAirportICAOcodeInDB(destinationAirportICAOcode)
    
    departureAirport = Airport(Name = originAirportICAOcode, 
                            LatitudeDegrees = airportsDatabase.getAirportLatitudeDegrees(originAirportICAOcode) , 
                            LongitudeDegrees = airportsDatabase.getAirportLongitudeDegrees(originAirportICAOcode) , 
                            fieldElevationAboveSeaLevelMeters = airportsDatabase.getAirportElevationMeters(originAirportICAOcode))
    
    arrivalAirport = Airport(Name = destinationAirportICAOcode , 
                             LatitudeDegrees = airportsDatabase.getAirportLatitudeDegrees(destinationAirportICAOcode) , 
                             LongitudeDegrees = airportsDatabase.getAirportLongitudeDegrees(destinationAirportICAOcode) , 
                             fieldElevationAboveSeaLevelMeters = airportsDatabase.getAirportElevationMeters(destinationAirportICAOcode) )
    
    ''' best departure run-way is the one with minimal distance between end of 5 nautical climb ramp and first point of the route '''
    bestDepartureRunway = runWaysDataBase.computeBestDepartureRunway( departureAirport, arrivalAirport )    
    bestArrivalRunway = runWaysDataBase.computeBestDepartureRunway( departureAirport, arrivalAirport )