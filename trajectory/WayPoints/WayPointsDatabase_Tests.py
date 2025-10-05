'''
Created on 5 oct. 2025

@author: rober
'''


import logging
import unittest
from trajectory.Guidance.WayPointFile import Airport, WayPoint

from trajectory.Environment.Airports.AirportDatabaseFile import AirportsDatabase
from trajectory.WayPoints.WayPointsDatabaseFile import WayPointsDatabase

#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        logging.basicConfig(level=logging.DEBUG)
        
        wayPointsdb = WayPointsDatabase()
        assert ( wayPointsdb.exists() == True )
        if wayPointsdb.exists():
            assert ( wayPointsdb.read() == True)
            
            wayPointName = "JAMSHEDPUR"
            logging.info(wayPointName)
            wayPoint = wayPointsdb.getWayPoint(wayPointName)
            logging.info( wayPoint )
            assert isinstance ( wayPoint , WayPoint) 
        
        
if __name__ == '__main__':
    unittest.main()