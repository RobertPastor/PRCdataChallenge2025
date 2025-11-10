'''
Created on 5 oct. 2025

@author: robert
'''


import logging
import unittest
from trajectory.Guidance.WayPointFile import Airport
from trajectory.Environment.Airports.AirportDatabaseFile import AirportsDatabase

#============================================
class Test_Main(unittest.TestCase):
    
    def test_main_two(self):
        logging.basicConfig(level=logging.DEBUG)
        
        print( "-------------------- test with pandas reader ---------------")

        airportsDatabase = AirportsDatabase()
        assert airportsDatabase.readWithPandas()

    def test_main_one(self):
        logging.basicConfig(level=logging.DEBUG)
    
        airportsDatabase = AirportsDatabase()
        ret = airportsDatabase.readAsDict()
        print ( ret )
        #for country in airportsDatabase.getCountries():
        
        ICAOcode = "LFPG"
        airport = airportsDatabase.getAirportFromICAOCode(ICAOcode)
        
        assert (isinstance(airport, Airport))
    
        if airport: 
            print ( "airport = {0} - elevation = {1} meters".format( airport.Name , airport.fieldElevationAboveSeaLevelMeters))
            
        ICAOcode = "MMMX"
        airport = airportsDatabase.getAirportFromICAOCode(ICAOcode)
        if airport: 
            print ( "airport = {0} - ICAO code = {1} ".format( airport.getName(), airport.getICAOcode()))
            print ( "airport = {0} - elevation = {1} meters".format( airport.Name , airport.fieldElevationAboveSeaLevelMeters))
            
            print ( "airport = {0} - latitude = {1} degrees".format( airport.Name , airport.getLatitudeDegrees()))        
            print ( "airport = {0} - longitude = {1} degrees".format( airport.Name , airport.getLongitudeDegrees()))
            
        
if __name__ == '__main__':
    unittest.main()