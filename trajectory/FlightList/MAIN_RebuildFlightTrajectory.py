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
from trajectory.Guidance.WayPointFile import Airport ,WayPoint
from trajectory.Environment.Airports.AirportDatabaseFile import AirportsDatabase
import matplotlib.pyplot as plt
from trajectory.Environment.Earth.EarthFile import Earth
from trajectory.Environment.Atmosphere.AtmosphereFile import Atmosphere
from trajectory.Openap.AircraftMainFile import OpenapAircraft

from trajectory.Environment.Runways.RunWaysDatabaseFile import RunWayDataBase
from trajectory.GuidanceOpenap.FlightPathOpenapFile import FlightPathOpenap

class FlightTrajectoryReBuild(object):
    
    flight_id  = "prc812317830"
    train_rank_final = "rank"
    columnNameList = ['latitude','longitude','altitude','groundspeed','track','vertical_rate', 'mach', 'TAS', 'CAS']

    def __init__(self , train_rank_final , flight_id ):
        self.train_rank_final = train_rank_final
        self.flight_id = flight_id
        assert self.train_rank_final == "train" or self.train_rank_final == "rank" or self.train_rank_final == "final"
        
    def readRunways(self):
        self.runWaysDataBase = RunWayDataBase()
        assert self.runWaysDataBase.read()
        print("-"*90)
        for runway in self.runWaysDataBase.getRunWays(originAirportICAOcode):
            print(runway)
        print("-"*90)
        for runway in self.runWaysDataBase.getRunWays(destinationAirportICAOcode):
            print(runway)
            
    def readAirports(self):
        self.airportsDatabase = AirportsDatabase()
        assert self.airportsDatabase.readWithPandas()
        
        print(f"airport {originAirportICAOcode} is in airports database -> {self.airportsDatabase.isAirportICAOcodeInDB(self.DepartureAirportICAOCode)}")
        print(f"airport {destinationAirportICAOcode} is in airports database -> {self.airportsDatabase.isAirportICAOcodeInDB(self.ArrivalAirportICAOCode)}")

    def extractAircraftICAOcode(self):
        self.aircraftICAOcode =  self.flightListDatabase.getAircraftICAOcode(self.flight_id)
        return self.aircraftICAOcode
    
    def extractDepartureAirport(self):
        self.DepartureAirportICAOCode = self.flightListDatabase.getOriginAirportICAOcode(self.train_rank_final , self.flight_id)
        return self.DepartureAirportICAOCode
    
    def extractDestinationAirport(self):
        self.ArrivalAirportICAOCode = self.flightListDatabase.getDestinationICAOairport(self.train_rank_final , self.flight_id)
        return self.ArrivalAirportICAOCode
    
    def extractTakeOffInstant(self):
        self.takeOffInstant = self.flightListDatabase.getTakeOffInstant( self.train_rank_final , self.flight_id )
        return self.takeOffInstant
    
    def extractLandedInstant(self):
        self.landedInstant = self.flightListDatabase.getLandedInstant( self.train_rank_final , self.flight_id )
        return self.landedInstant
    
    def readFlightListDatabase(self):
        logging.info("---------Read Flight List <<" + self.train_rank_final +">> ------------")

        self.flightListDatabase = FlightListDatabase(self.train_rank_final)
        if self.flightListDatabase.readTrainRankFinalFlightListLite(self.train_rank_final):
            logging.info("train rank final flight list read correctly")
            
            self.flightListDataframe = self.flightListDatabase.getTrainRankFinalFlightListDataframe(self.train_rank_final)
            print(tabulate(self.flightListDataframe[:10], headers='keys', tablefmt='grid' , showindex=False , ))
            logging.info(self.flightListDataframe.shape)
            
    def computeBestRunways(self):
        
        print (f"are there runways for the origin airport -> {originAirportICAOcode}  -> {self.runWaysDataBase.hasRunWays(originAirportICAOcode)}" )
        print (f"are there runways for the destination airport -> {destinationAirportICAOcode}  -> {self.runWaysDataBase.hasRunWays(destinationAirportICAOcode)}" )

        ''' best departure run-way is the one with minimal distance between end of 5 nautical climb ramp and first point of the route '''
        departureAirport = Airport(Name = originAirportICAOcode, 
                            LatitudeDegrees = self.airportsDatabase.getAirportLatitudeDegrees(originAirportICAOcode) , 
                            LongitudeDegrees = self.airportsDatabase.getAirportLongitudeDegrees(originAirportICAOcode) , 
                            fieldElevationAboveSeaLevelMeters = self.airportsDatabase.getAirportElevationMeters(originAirportICAOcode),
                            ICAOcode = originAirportICAOcode)
        print(departureAirport)
        arrivalAirport = Airport(Name = destinationAirportICAOcode , 
                             LatitudeDegrees = self.airportsDatabase.getAirportLatitudeDegrees(destinationAirportICAOcode) , 
                             LongitudeDegrees = self.airportsDatabase.getAirportLongitudeDegrees(destinationAirportICAOcode) , 
                             fieldElevationAboveSeaLevelMeters = self.airportsDatabase.getAirportElevationMeters(destinationAirportICAOcode) ,
                             ICAOcode = destinationAirportICAOcode)
        print (arrivalAirport)
        self.bestDepartureRunway = self.runWaysDataBase.computeBestDepartureRunway( departureAirport, arrivalAirport )
        self.bestArrivalRunway = self.runWaysDataBase.computeBestArrivalRunway( departureAirport, arrivalAirport )
            
    def getAdepRouteAsString(self, AdepRunWayName = None):
        strRoute = "ADEP/" + self.DepartureAirportICAOCode 
        if (AdepRunWayName):
            strRoute += "/" + AdepRunWayName
        return strRoute
    
    def getAdesRouteAsString(self, AdesRunWayName = None ):
        strRoute = ""
        if (AdesRunWayName):
            strRoute += "ADES/" + self.ArrivalAirportICAOCode
            strRoute += "/" + AdesRunWayName
        return strRoute
            
    def computeDirectRouteAsString(self):
        strRoute = self.getAdepRouteAsString(AdepRunWayName = self.bestDepartureRunway.getName())
        strRoute += "-"
        strRoute += self.getAdesRouteAsString(AdesRunWayName = self.bestArrivalRunway.getName())
        strRoute = str(strRoute).replace("--", "-")
        print(f"direct route = {strRoute}")
        return strRoute
        
    def computeFlightProfile(self):
        
        self.aircraft = OpenapAircraft( aircraftICAOcode     = str(self.aircraftICAOcode).lower() , 
                                        earth                = Earth() , 
                                        atmosphere           = Atmosphere() , 
                                        initialMassKilograms = None)
        
        if not (self.aircraft is None) :
            routeAsString = self.computeDirectRouteAsString()
            print ( routeAsString )
            flightPath = FlightPathOpenap(
                    route = routeAsString, 
                    aircraftICAOcode     = self.aircraftICAOcode.lower(),
                    RequestedFlightLevel = float( self.aircraft.getMaxCruiseFlightLevel() ), 
                    cruiseMach           = float( self.aircraft.getMaximumSpeedMmoMach() ), 
                    takeOffMassKilograms = float( self.aircraft.getReferenceMassKilograms() ) ,
                    reducedClimbPowerCoeff = 0.0 ,
                    directRoute            = True)
            try:
                    flightPath.computeFlight(deltaTimeSeconds = 1.0)
                    abortedFlight = flightPath.abortedFlight
                    #csvAltitudeMSLTimeGroundTrack = flightPath.createCsvAltitudeTimeProfile()
                    #flightPath.createStateVectorHistoryFile()
                    print(f"flight_id = {self.flight_id} - takeoff instant = {self.takeOffInstant}")
                    pd_df = flightPath.getAircraft().createPRCdataChallengeFlightDataframe(abortedFlight , self.aircraftICAOcode ,
                                                                                   self.flight_id , self.takeOffInstant)
                    #flightPath.createKmlXmlDocument()
                    
    
            except Exception as e:
                    logging.error("Trajectory Compute Wrap - Exception = {0}".format( str(e ) ) )
        else:
            raise ValueError(f"cannot find aircraft {self.aircraftICAOcode} in Openap database ")

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
    
    ac = flightTrajectoryReBuild.extractAircraftICAOcode()
    print ("aircraft = " + ac  )

    originAirportICAOcode = flightTrajectoryReBuild.extractDepartureAirport()
    print( f"origin airport =>  {originAirportICAOcode}" )  
    destinationAirportICAOcode =   flightTrajectoryReBuild.extractDestinationAirport()
    print( f"destination airport =>  {destinationAirportICAOcode}" )
    
    print ( flightTrajectoryReBuild.extractTakeOffInstant())
    print ( flightTrajectoryReBuild.extractLandedInstant())
    
    flightTrajectoryReBuild.readAirports()
    flightTrajectoryReBuild.readRunways()
    flightTrajectoryReBuild.computeBestRunways()
    flightTrajectoryReBuild.computeFlightProfile()
    