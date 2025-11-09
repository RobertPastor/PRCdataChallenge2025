'''
Created on 9 nov. 2025

@author: robert
'''
from trajectory.Flights.FlightsReader import FlightsDatabase

import os

import logging
import unittest
import pandas as pd
import os
from trajectory.Flights.FlightsReader import FlightsDatabase
from tabulate import tabulate
from trajectory.Fuel.FuelReader import FuelDatabase


#============================================
class Test_Main(unittest.TestCase):
    
    def test_main_interpolate_all(self):
        
        train_rank = "train"
        ''' open fuel database '''
        fuelDatabase = FuelDatabase()
        fuelTrainDataframe = fuelDatabase.fuelDatabasereadFuelTrainLite()
        print ( list ( fuelTrainDataframe ))
        
        '''loop through the files '''
        flightsDatabase = FlightsDatabase()
        
        directory = flightsDatabase.getTrainFlightsFolderPathStr()
        for fileName in os.listdir(directory):
            if fileName.endswith(".parquet"): # Filter specific file types
                filePath = os.path.join(directory, fileName)
                print ( filePath )
                
                flight_id = fileName.split("\\.")[0]
                print("flight_id = " + flight_id)
                
                flightTrainDataframe = flightsDatabase.readOneTrainFileLite(fileName)
                print ( list ( flightTrainDataframe ))
        
      
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(pd. __version__)
    
    unittest.main()