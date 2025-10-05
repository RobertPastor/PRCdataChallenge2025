'''
Created on 5 oct. 2025

@author: rober
'''

import time
import csv
import unittest
import os
import math

from trajectory.Environment.Earth.Earth import Earth
from trajectory.Environment.Airports.AirportDatabaseFile import AirportsDatabase
from trajectory.Guidance.WayPointFile import Airport

from trajectory.Environment.Runways.RunWaysDatabaseFile import RunWayDataBase

#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        
        airportsDatabase = AirportsDatabase()
        ret = airportsDatabase.read()
        print ("Airports database read status = " + str(ret) )
        #for country in airportsDatabase.getCountries():
        #    print ( country )
            
        ICAOcode = "LFPG"
        airport = airportsDatabase.getAirportFromICAOCode(ICAOcode)
        
        assert (isinstance(airport, Airport))
        print("Airport = " + airport.getName())
        
    def test_main_two(self):
        
        runwaysDB = RunWayDataBase()
        if (runwaysDB.exists()):
            print("runwaysDB exists")
            ret = runwaysDB.read()
            print ("read runways database result = {0}".format(ret))
        else:
            print("runwaysDB does not exists")
            
    def test_main_three(self):
        
        airportsDatabase = AirportsDatabase()
        ret = airportsDatabase.read()
        if ret:
            print ("Airports database read status = " + str(ret) )
            #for country in airportsDatabase.getCountries():
            #    print ( country )
                
            airportICAOcode = "LFPG"
            airportLFPG = airportsDatabase.getAirportFromICAOCode(airportICAOcode)
            
            runwaysDB = RunWayDataBase()
            
            if (runwaysDB.exists()):
                print("runwaysDB exists")
                if runwaysDB.read():
                    
                    for runway in runwaysDB.getRunWays(airportICAOcode):
                        print(runway)
                
        
        
if __name__ == '__main__':
    unittest.main()