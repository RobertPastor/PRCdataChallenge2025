'''
Created on 5 oct. 2025

@author: rober
'''



from time import time
import csv
import unittest
import os
import sys
import math
import logging

from trajectory.Environment.Earth.EarthFile import Earth
from trajectory.Environment.Atmosphere.AtmosphereFile import Atmosphere

from trajectory.Environment.Airports.AirportDatabaseFile import AirportsDatabase
from trajectory.Guidance.WayPointFile import Airport

from trajectory.Environment.Runways.RunWaysDatabaseFile import RunWayDataBase
from openap import prop

from trajectory.Openap.AircraftMainFile import OpenapAircraft

from trajectory.GuidanceOpenap.FlightPathOpenapFile import FlightPathOpenap

#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        
        airportsDatabase = AirportsDatabase()
        if airportsDatabase.read():
            print ("Airports database read correctly" )
            #for country in airportsDatabase.getCountries():
            #    print ( country )
                
            ICAOcode = "LFPG"
            airport = airportsDatabase.getAirportFromICAOCode(ICAOcode)
            
            assert (isinstance(airport, Airport))
            print("Airport = " + airport.getName())
            
    def test_main_two(self):
        
        earth = Earth()
        atmosphere = Atmosphere()
        
        start_time = time()
        
        ''' warning : wrap aircraft code letters must be in lower case '''
        aircraftICAOcode = 'a320'
        #aircraftICAOcode = 'a332'
        
        logging.info("Trajectory Compute Wrap - " + aircraftICAOcode)
        Adep = "KATL"
        Ades = "KLAX"
        route = 'KATL-KLAX'
        #route = "MMMX-KSEA"
        
        AdepRunway = "27R"
        AdesRunway = "07L"
        
        runwaysDB = RunWayDataBase()
        if (runwaysDB.exists()):
            print("runwaysDB exists")
            ret = runwaysDB.read()
            assert ret == True
            logging.info("------------ show Adep runways ------------")
            for runway in runwaysDB.getRunWays(Adep):
                print(runway)
                
            logging.info("------------ show Ades runways ------------")

            for runway in runwaysDB.getRunWays(Ades):
                print(runway)
        else:
            sys.exit()
        
        available_acs = prop.available_aircraft(use_synonym=True)

        if aircraftICAOcode in available_acs:
        
            ac = OpenapAircraft( aircraftICAOcode , earth , atmosphere , initialMassKilograms = None)
            logging.info( ac.getAircraftName())

            takeOffWeightKg = ac.getReferenceMassKilograms()
            logging.info("take off weight = {0:.2f} kg".format( takeOffWeightKg ))
            cruiseFlightLevel = ac.getMaxCruiseAltitudeFeet() / 100.0
            logging.info("cruise level FL = {0:.2f} ".format ( cruiseFlightLevel ) )
            
            targetCruiseMach = ac.getMaximumSpeedMmoMach()
            logging.info( "target cruise mach = {0:.2f} ".format( targetCruiseMach ) )
            
            if not ( aircraftICAOcode in prop.available_aircraft(use_synonym=True) ):
                logging.error( "Aircraft code = {0} not in openap Wrap".format( aircraftICAOcode ))
            else:
            
                    strRoute = "ADEP" + "/" + Adep + "/" + AdepRunway 
                    strRoute += "-" + "VUZ" + "-" + "ABQ" + "-" + "TNP" 
                    strRoute += "-" + "ADES" + "/" + Ades + "/" + AdesRunway 
                    logging.info(strRoute)
                    flightPath = FlightPathOpenap(
                            route                = strRoute, 
                            aircraftICAOcode     = aircraftICAOcode,
                            RequestedFlightLevel = cruiseFlightLevel, 
                            cruiseMach           = targetCruiseMach, 
                            takeOffMassKilograms = takeOffWeightKg)
                    try:
                        flightPath.computeFlight(deltaTimeSeconds = 1.0)
                        
                        end_time = time()
                        seconds_elapsed = end_time - start_time
                    
                        hours, rest = divmod(seconds_elapsed, 3600)
                        minutes, seconds = divmod(rest, 60)
                        logging.info ( "hours = {0} - minutes = {1} - seconds = {2:.2f}".format( hours, minutes, seconds))
                        
                        
                        #flightPath.createStateVectorHistoryFile()
                        #flightPath.createKmlXmlDocument()
                    
                    except Exception as e:
                        logging.error("Trajectory Compute Wrap - Exception = {0}".format( str(e ) ) )
