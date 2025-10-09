'''
Created on 7 oct. 2025

@author: rober
'''
import logging
import unittest
import pandas as pd
import os
from trajectory.FlightList.FlightListReader import FlightListDatabase
#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        
        print("---------------- test_main_one  ----------------")

        logging.basicConfig(level=logging.INFO)
        
        logging.info("Read Flight List")
        
        flightList = FlightListDatabase()
        if flightList.readTrainFlightList():
            logging.info("train flight list read correctly")
            
            
    def test_main_two(self):
        
        print("---------------- test_main_two  ----------------")

        logging.basicConfig(level=logging.INFO)
        
        logging.info("Read Flight List")
        
        flightList = FlightListDatabase()
        if flightList.readRankFlightList():
            logging.info("rank flight list read correctly")
            
    def test_main_three(self):
        
        print("---------------- test_main_three  ----------------")

        flightList = FlightListDatabase()
        if flightList.readRankFlightList():
            flightList.checkRankFligthListHeaders()
            assert flightList.checkRankFligthListHeaders() ==  True
            
        if flightList.readTrainFlightList():
            flightList.checkTrainFlightListHeaders()
            assert flightList.checkTrainFlightListHeaders() ==  True

    def test_main_four(self):
        logging.basicConfig(level=logging.INFO)

        print("---------------- test main four ----------------")
        
        flightList = FlightListDatabase()
        assert ( flightList.readRankFlightList() == True )
        assert ( flightList.readTrainFlightList() == True )
        
        flightList.collectUniqueAirports()
            
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(pd. __version__)
    
    unittest.main()