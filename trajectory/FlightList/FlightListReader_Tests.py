'''
Created on 7 oct. 2025

@author: rober
'''
import logging
import unittest

from trajectory.FlightList.FlightListReader import FlightListDatabase
#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        logging.basicConfig(level=logging.DEBUG)
        
        logging.info("Read Flight List")
        
        flightList = FlightListDatabase()
        if flightList.readTrainFlightList():
            logging.info("train flight list read correctly")
            
            
    def test_main_two(self):
        logging.basicConfig(level=logging.DEBUG)
        
        logging.info("Read Flight List")
        
        flightList = FlightListDatabase()
        if flightList.readRankFlightList():
            logging.info("rank flight list read correctly")
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()