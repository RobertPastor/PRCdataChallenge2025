'''
Created on 1 déc. 2025

@author: robert

'''


import platform
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import logging
from tabulate import tabulate
from pathlib import Path

from trajectory.FlightList.FlightListReader import FlightListDatabase
from trajectory.Flights.FlightsReader import FlightsDatabase
from trajectory.Guidance.WayPointFile import Airport 
from trajectory.Environment.Airports.AirportDatabaseFile import AirportsDatabase
from trajectory.Environment.Earth.EarthFile import Earth
from trajectory.Environment.Atmosphere.AtmosphereFile import Atmosphere
from trajectory.Openap.AircraftMainFile import OpenapAircraft

from trajectory.Environment.Runways.RunWaysDatabaseFile import RunWaysDataBase
from trajectory.GuidanceOpenap.FlightPathOpenapFile import FlightPathOpenap
from trajectory.Environment.WayPoints.WayPointsDatabaseFile import WayPointsDatabase

''' with a list of flight_id to rebuild the trajectories '''
class FlightIdsToRebuild(object):
    pass
    def __init__(self , train_rank_final ):
        
        assert isinstance ( train_rank_final , str )
        assert train_rank_final == "train" or train_rank_final == "rank" or train_rank_final == "final"
        self.train_rank_final = train_rank_final
        self.flightsDatabase = FlightsDatabase()
        
        self.earth = Earth()
        self.atmosphere = Atmosphere()
        
        self.runwaysDatabase = RunWaysDataBase()
        assert self.runwaysDatabase.read()
        
        self.airportsDatabase = AirportsDatabase()
        assert self.airportsDatabase.readWithPandas()
        
        self.wayPointsDatabase = WayPointsDatabase()
        assert self.wayPointsDatabase.exists()
        assert self.wayPointsDatabase.read() 
        logging.info ("number of way-points = {0}".format(self.wayPointsDatabase.getNumberOfWaypoints()) )
        pass
        self.errorsDict = {} 
        
    def getFlighIdsToRebuildFilteredByAircraft(self, aircraftICAOcodeToFilter):
        print("="*120)
        print("aircraft = " + aircraftICAOcodeToFilter)
        print("="*120)

        self.flightListDatabase = FlightListDatabase(self.train_rank_final)
        assert self.flightListDatabase.readTrainRankFinalFlightListLite(self.train_rank_final)
        logging.info("train rank final flight list read correctly")
            
        df = self.flightListDatabase.getTrainRankFinalFlightListDataframe(self.train_rank_final)
        df = df[df["aircraft_type"] == aircraftICAOcodeToFilter]
        dfSeries = df["flight_id"]
        
        flight_ids_list = []
        for index, flight_id in dfSeries.items():
            print(f"Index: {index}, Value: {flight_id}")
            flight_ids_list.append(flight_id)
        
        print(flight_ids_list)
        return flight_ids_list
        
        
    def readFlighIdsToRebuild(self):
        pass
        self.filesFolder = os.path.dirname(__file__)
        self.fileName = self.train_rank_final + "_" "FlightIdsToRebuild" + ".xlsx"
        self.filePath = os.path.join ( self.filesFolder, self.fileName)
        logging.info ( self.filePath )
        self.df_flight_ids = pd.read_excel ( io = self.filePath  , sheet_name = "flight_ids" , names = ["flight_id"] )
        logging.info ( self.df_flight_ids.shape )
        return True
    
    def getFlightIdsListToRebuild(self):
        flight_ids_list = []
        for index , row in self.df_flight_ids.iterrows():
            flight_id = row["flight_id"]
            #logging.info (str(index) + " -> " +  flight_id)
            
            folder = self.flightsDatabase.getTrainRankFinalFlightsComputedFolderPathStr(self.train_rank_final)
            fileName = flight_id + ".parquet"
            filePathStr = os.path.join ( folder , fileName)
            if os.path.exists(filePathStr) and os.path.isfile(filePathStr):
                logging.info ( "file path ->" + filePathStr + " has been already computed")
            else:
                flight_ids_list.append(flight_id)
        return flight_ids_list
    
    def getFlightIdsToRebuildForOneAircraft(self , aircraftICAOcodeToFilter):
        flight_ids_list = []
        for index , row in self.df_flight_ids.iterrows():
            flight_id = row["flight_id"]
            #logging.info (str(index) + " -> " +  flight_id)
            
            folder = self.flightsDatabase.getTrainRankFinalFlightsComputedFolderPathStr(self.train_rank_final)
            fileName = flight_id + ".parquet"
            filePathStr = os.path.join ( folder , fileName)
            if os.path.exists(filePathStr) and os.path.isfile(filePathStr):
                logging.info ( "file path ->" + filePathStr + " has been already computed")
            else:
                
                flight_ids_list.append(flight_id)
        return flight_ids_list

    def rebuildAllFlightIds(self ,aircraftICAOcode):
        counter = 0
        for index , row in self.df_flight_ids.iterrows():
            
            flight_id = row["flight_id"]
            logging.info (str(index) + " -> " +  flight_id)
            
            folder = self.flightsDatabase.getTrainRankFinalFlightsComputedFolderPathStr(self.train_rank_final)
            fileName = flight_id + ".parquet"
            filePathStr = os.path.join ( folder , fileName)
            if os.path.exists(filePathStr) and os.path.isfile(filePathStr):
                logging.info ( "file path ->" + filePathStr + " has been already computed")
            else:
                counter = counter + 1
                if counter > 500:
                    break
                try:
                    self.rebuildOneFlightId(flight_id , aircraftICAOcode)
                except Exception as e:
                    self.errorsDict[flight_id] = "{0}".format(e)
                    logging.error("{0}".format(e))
                    
        return self.errorsDict
                    
    def rebuildOneFlightId(self , argumentList ):
        
        flight_id = argumentList[0]
        assert isinstance( flight_id , str )

        aircraftICAOcode = argumentList[1]
        if aircraftICAOcode == None:
            ''' no filter on the aircraft type '''
            pass
        else:
            assert isinstance( aircraftICAOcode , str )
        print (" rebuildOneFlightId = " + flight_id + " -> aircraft ICAO code = " + aircraftICAOcode)
        try:
            print("--> rebuild flight = " + flight_id)
            flightTrajectoryReBuild = FlightTrajectoryReBuild( self.train_rank_final , flight_id , 
                                                               self.earth , self.atmosphere, 
                                                               self.airportsDatabase, self.runwaysDatabase , 
                                                               self.wayPointsDatabase)
            flightTrajectoryReBuild.readFlightListDatabase(aircraftICAOcode)
            ac = flightTrajectoryReBuild.extractAircraftICAOcode()
            logging.info ("aircraft = " + ac  )
            if ((ac) and (aircraftICAOcode == None)) or ((ac) and (str(ac).upper() == str(aircraftICAOcode).upper())):
                
                originAirportICAOcode = flightTrajectoryReBuild.extractDepartureAirport()
                logging.info( f"origin airport =>  {originAirportICAOcode}" )  
                destinationAirportICAOcode =   flightTrajectoryReBuild.extractDestinationAirport()
                logging.info( f"destination airport =>  {destinationAirportICAOcode}" )
                
                logging.info (f"take-off instant =  {flightTrajectoryReBuild.extractTakeOffInstant()}")
                logging.info (f"landed instant = {flightTrajectoryReBuild.extractLandedInstant()}")
                
                flightTrajectoryReBuild.readAirports(originAirportICAOcode,destinationAirportICAOcode)
                if flightTrajectoryReBuild.readRunways(originAirportICAOcode,destinationAirportICAOcode):
                    flightTrajectoryReBuild.computeBestRunways(originAirportICAOcode,destinationAirportICAOcode)
                    flightTrajectoryReBuild.computeFlightProfile()
            else:
                logging.info("aircraft not found -> " + ac + " - or it has been filtered")
                #raise ValueError("aircraft not found -> " + ac)
        except Exception as e:
            print (e)

class FlightTrajectoryReBuild(object):
    
    flight_id  = "prc812317830"
    train_rank_final = "rank"
    columnNameList = ['latitude','longitude','altitude','groundspeed','track','vertical_rate', 'mach', 'TAS', 'CAS']

    def __init__(self , train_rank_final , flight_id , earth , atmosphere, airportsDatabase, runwaysDataBase , waypointsDatabase):
        self.train_rank_final = train_rank_final
        self.flight_id = flight_id
        assert self.train_rank_final == "train" or self.train_rank_final == "rank" or self.train_rank_final == "final"
        
        self.flightsDatabase = FlightsDatabase()
        
        assert (isinstance (earth , Earth))
        self.earth = earth
        
        assert (isinstance (atmosphere , Atmosphere))
        self.atmosphere = atmosphere
        
        assert ( isinstance ( airportsDatabase , AirportsDatabase))
        self.airportsDatabase = airportsDatabase
        
        assert ( isinstance ( runwaysDataBase , RunWaysDataBase))
        self.runwaysDataBase = runwaysDataBase
        
        assert isinstance ( waypointsDatabase , WayPointsDatabase )
        self.waypointsDatabase = waypointsDatabase
        logging.info( "size of waypoints database = {0}".format(self.waypointsDatabase.getNumberOfWaypoints()))
        pass
        
    def readRunways(self , originAirportICAOcode , destinationAirportICAOcode):
        logging.info("-"*90)
        originAirportRunwaysFound = False
        for runway in self.runwaysDataBase.getRunWays(originAirportICAOcode):
            #logging.info(runway)
            originAirportRunwaysFound = True
        assert ( originAirportRunwaysFound == True)
            
        destinationAirportRunwaysFound = False
        logging.info("-"*90)
        for runway in self.runwaysDataBase.getRunWays(destinationAirportICAOcode):
            #logging.info(runway)
            destinationAirportRunwaysFound = True
        assert ( destinationAirportRunwaysFound == True)
        return True
        
    def readAirports(self , originAirportICAOcode , destinationAirportICAOcode):
        
        assert self.airportsDatabase.isAirportICAOcodeInDB(originAirportICAOcode)
        assert self.airportsDatabase.isAirportICAOcodeInDB(destinationAirportICAOcode)
        logging.info(f"airport {originAirportICAOcode} is in airports database -> {self.airportsDatabase.isAirportICAOcodeInDB(originAirportICAOcode)}")
        logging.info(f"airport {destinationAirportICAOcode} is in airports database -> {self.airportsDatabase.isAirportICAOcodeInDB(destinationAirportICAOcode)}")

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
    
    def readFlightListDatabase(self , aircraftICAOcode):
        logging.info("---------Read Flight List <<" + self.train_rank_final +">> ------------")
        self.flightListDatabase = FlightListDatabase(self.train_rank_final)
        if self.flightListDatabase.readTrainRankFinalFlightListLite(self.train_rank_final):
            logging.info("train rank final flight list read correctly")
            
            self.flightListDataframe = self.flightListDatabase.getTrainRankFinalFlightListDataframe(self.train_rank_final)
            if aircraftICAOcode:
                ''' filter on aircraft type = A359 '''
                self.flightListDataframe = self.flightListDataframe[self.flightListDataframe['aircraft_type'] == str(aircraftICAOcode.upper())] 
            
            logging.info(tabulate(self.flightListDataframe[:10], headers='keys', tablefmt='grid' , showindex=False , ))
            logging.info(self.flightListDataframe.shape)
            
    def computeBestRunways(self , originAirportICAOcode , destinationAirportICAOcode):
        
        logging.info (f"are there runways for the origin airport -> {originAirportICAOcode}  -> {self.runwaysDataBase.hasRunWays(originAirportICAOcode)}" )
        logging.info (f"are there runways for the destination airport -> {destinationAirportICAOcode}  -> {self.runwaysDataBase.hasRunWays(destinationAirportICAOcode)}" )

        ''' best departure run-way is the one with minimal distance between end of 5 nautical climb ramp and first point of the route '''
        self.departureAirport = Airport(Name = originAirportICAOcode, 
                            LatitudeDegrees = self.airportsDatabase.getAirportLatitudeDegrees(originAirportICAOcode) , 
                            LongitudeDegrees = self.airportsDatabase.getAirportLongitudeDegrees(originAirportICAOcode) , 
                            fieldElevationAboveSeaLevelMeters = self.airportsDatabase.getAirportElevationMeters(originAirportICAOcode),
                            ICAOcode = originAirportICAOcode)
        logging.info(f"departure airport = {self.departureAirport}")
        assert isinstance ( self.departureAirport , Airport )
        self.arrivalAirport = Airport(Name = destinationAirportICAOcode , 
                             LatitudeDegrees = self.airportsDatabase.getAirportLatitudeDegrees(destinationAirportICAOcode) , 
                             LongitudeDegrees = self.airportsDatabase.getAirportLongitudeDegrees(destinationAirportICAOcode) , 
                             fieldElevationAboveSeaLevelMeters = self.airportsDatabase.getAirportElevationMeters(destinationAirportICAOcode) ,
                             ICAOcode = destinationAirportICAOcode)
        logging.info (f"arrival airport = {self.arrivalAirport}")
        assert isinstance ( self.arrivalAirport , Airport )
        
        self.bestDepartureRunway = self.runwaysDataBase.computeBestDepartureRunway(self.departureAirport,self.arrivalAirport)
        logging.info(f"best departure runway = {self.bestDepartureRunway}")
        assert not self.bestDepartureRunway is None
        
        self.bestArrivalRunway = self.runwaysDataBase.computeBestArrivalRunway(self.departureAirport,self.arrivalAirport)
        logging.info(f"best arrival runway = {self.bestArrivalRunway}")
        assert not self.bestArrivalRunway is None
        
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
        logging.info(f"direct route = {strRoute}")
        return strRoute
        
    def computeFlightProfile(self ):
        self.aircraft = None
        if (self.aircraftICAOcode):
            self.aircraft = OpenapAircraft( aircraftICAOcode = str(self.aircraftICAOcode).lower() , 
                                        earth                = self.earth , 
                                        atmosphere           = self.atmosphere ,
                                        initialMassKilograms = None)
            logging.info(self.aircraft)
        if not (self.aircraft is None) :
            routeAsString = self.computeDirectRouteAsString()
            logging.info ( routeAsString )
            
            ''' 24th November 2025 - use maximum mass instead of reference mass '''
            flightPath = FlightPathOpenap(
                    strRoute               = routeAsString, 
                    aircraftICAOcode       = self.aircraftICAOcode.lower(),
                    RequestedFlightLevel   = float( self.aircraft.getMaxCruiseFlightLevel() ), 
                    cruiseMach             = float( self.aircraft.getMaximumSpeedMmoMach() ), 
                    takeOffMassKilograms   = float( self.aircraft.getReferenceMassKilograms() ) ,
                    reducedClimbPowerCoeff = 0.0 ,
                    earth                  = self.earth,
                    atmosphere             = self.atmosphere,
                    airportsDatabase       = self.airportsDatabase,
                    runwaysDataBase        = self.runwaysDataBase,
                    waypointsDatabase      = self.waypointsDatabase,
                    directRoute            = True)
            
            flightPath.computeFlight(deltaTimeSeconds = 1.0)
            abortedFlight = flightPath.abortedFlight
            #csvAltitudeMSLTimeGroundTrack = flightPath.createCsvAltitudeTimeProfile()
            #flightPath.createStateVectorHistoryFile()
            logging.info(f"flight_id = {self.flight_id} - takeoff instant = {self.takeOffInstant}")
            finalRoute = flightPath.finalRoute
            df = flightPath.getAircraft().createPRCdataChallengeFlightDataframe(finalRoute, abortedFlight , self.aircraftICAOcode ,
                                                                                       self.flight_id , self.takeOffInstant)
            #flightPath.createKmlXmlDocument()

            folder = self.flightsDatabase.getTrainRankFinalFlightsComputedFolderPathStr(self.train_rank_final)
            fileName = self.flight_id + ".parquet"
            path = os.path.join ( folder , fileName)
            df.to_parquet( path , index=False)
            #flightPath.createKmlXmlDocument()
                
        else:
            logging.info("aircraft not found -> " + self.aircraftICAOcode.lower())