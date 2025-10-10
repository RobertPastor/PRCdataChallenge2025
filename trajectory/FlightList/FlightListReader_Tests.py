'''
Created on 7 oct. 2025

@author: robert
'''
import logging
import unittest

from trajectory.FlightList.FlightListReader import FlightListDatabase
#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        
        print("---------------- test_main_one  ----------------")

        logging.basicConfig(level=logging.DEBUG)
        
        logging.info("Read Flight List")
        
        flightList = FlightListDatabase()
        if flightList.readTrainFlightList():
            logging.info("train flight list read correctly")
            
            
    def test_main_two(self):
        
        print("---------------- test_main_two  ----------------")

        logging.basicConfig(level=logging.DEBUG)
        
        logging.info("Read Flight List")
        
        flightList = FlightListDatabase()
        if flightList.readRankFlightList():
            logging.info("rank flight list read correctly")
            
    def test_main_three(self):
        
        logging.basicConfig(level=logging.DEBUG)

        print("---------------- test_main_three  ----------------")

        flightList = FlightListDatabase()
        if flightList.readRankFlightList():
            flightList.checkRankFligthListHeaders()
            assert flightList.checkRankFligthListHeaders() ==  True
            
        if flightList.readTrainFlightList():
            flightList.checkTrainFlightListHeaders()
            assert flightList.checkTrainFlightListHeaders() ==  True

    def test_main_four(self):
        
        logging.basicConfig(level=logging.DEBUG)

        print("---------------- test main four ----------------")
        
        flightList = FlightListDatabase()
        assert ( flightList.readRankFlightList() == True )
        assert ( flightList.readTrainFlightList() == True )
        
        flightList.collectUniqueAirports()
        
    def test_main_five(self):
        
        logging.basicConfig(level=logging.DEBUG)

        print("---------------- test main five ----------------")
        
        flightList = FlightListDatabase()
        assert ( flightList.readTrainFlightList() == True )
        assert ( flightList.readRankFlightList() == True )
        
        assert flightList.checkTrainFlightListHeaders() == True
        assert flightList.checkRankFligthListHeaders() == True
        #flightList.collectUniqueAirports()
        
        flightList.extendTrainFlightListWithAirportData()
        
    def test_main_six(self):
            
        logging.basicConfig(level=logging.INFO)

        print("---------------- test main six ----------------")
        
        flightList = FlightListDatabase()
        assert ( flightList.readTrainFlightList() == True )
        
        flightList.extendTrainFlightDataWithFlightListData()

        
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    unittest.main()