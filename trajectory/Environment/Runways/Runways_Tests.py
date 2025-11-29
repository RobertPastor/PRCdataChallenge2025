'''
Created on 5 oct. 2025

@author: rober
'''

import time
import csv
import unittest
import os
import math
import logging
from trajectory.Environment.Earth.EarthFile import Earth
from trajectory.Environment.Airports.AirportDatabaseFile import AirportsDatabase
from trajectory.Guidance.WayPointFile import Airport
from trajectory.Environment.Runways.RunWaysDatabaseFile import RunWaysDataBase

#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        logging.basicConfig(level=logging.INFO)

        airportsDatabase = AirportsDatabase()
        ret = airportsDatabase.readAsDict()
        logging.info ("Airports database read status = " + str(ret) )
        #for country in airportsDatabase.getCountries():
        #    logging.info ( country )
            
        ICAOcode = "LFPG"
        airport = airportsDatabase.getAirportFromICAOCode(ICAOcode)
        
        assert (isinstance(airport, Airport))
        logging.info("Airport = " + airport.getName())
        
    def test_main_two(self):
        logging.basicConfig(level=logging.INFO)

        runwaysDB = RunWaysDataBase()
        if (runwaysDB.exists()):
            logging.info("runwaysDB exists")
            ret = runwaysDB.read()
            logging.info ("read runways database result = {0}".format(ret))
        else:
            logging.info("runwaysDB does not exists")
            
    def test_main_three(self):
        logging.basicConfig(level=logging.INFO)

        airportsDatabase = AirportsDatabase()
        if airportsDatabase.readAsDict():
            index = 0
            for country in airportsDatabase.getCountries():
                index = index + 1
                if index>10:
                    break
                logging.info ("country = " + country )
                
            airportICAOcode = "LFPG"
            airportLFPG = airportsDatabase.getAirportFromICAOCode(airportICAOcode)
            logging.info(airportLFPG)
            
            runwaysDB = RunWaysDataBase()
            if (runwaysDB.exists()):
                logging.info("runwaysDB exists")
                if runwaysDB.read():
                    
                    for runway in runwaysDB.getRunWays(airportICAOcode):
                        logging.info(runway)
                
        
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()