'''
Created on 9 oct. 2025

@author: rober
'''


import logging
import unittest
import pandas as pd
import os
from trajectory.Flights.FlightsReader import FlightsDatabase

#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        logging.basicConfig(level=logging.INFO)

        print("---------------- test_main_one  ----------------")

        logging.basicConfig(level=logging.INFO)
        
        logging.info("Read Flight data")
        flightsDatabase = FlightsDatabase()
        assert flightsDatabase.readSomeFiles(testMode = True) == True
        
        
    def test_main_two(self):
        logging.basicConfig(level=logging.INFO)

        print("---------------- test_main_two  ----------------")

        logging.basicConfig(level=logging.INFO)
        
        logging.info("Read Flight data")
        flightsDatabase = FlightsDatabase()
        assert flightsDatabase.readSomeFiles(testMode = True) == True
        
        assert flightsDatabase.checkFlightsTrainHeaders()
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(pd. __version__)
    
    unittest.main()
        