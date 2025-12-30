'''
Created on 7 oct. 2025

@author: robert
'''
import logging
import unittest
from tabulate import tabulate

from trajectory.FlightList.FlightListReader import FlightListDatabase
#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        
        print("---------------- Read Train flight list  ----------------")
        logging.basicConfig(level=logging.DEBUG)
        
        logging.info("Read Flight List")
        
        flightList = FlightListDatabase()
        if flightList.readTrainFlightList():
            logging.info("train flight list read correctly")
            
        df = flightList.getTrainFlightListDataframe()
        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))

            
    def test_main_two(self):
        
        print("---------------- Read Rank Flight list  ----------------")
        logging.basicConfig(level=logging.DEBUG)
        
        logging.info("Read Flight List")
        
        flightList = FlightListDatabase()
        if flightList.readRankFlightList():
            logging.info("rank flight list read correctly")
            
        df = flightList.getRankFlightListDataframe()
        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))

    def test_main_three(self):
        
        logging.basicConfig(level=logging.DEBUG)

        print("---------------- check headers  ----------------")

        flightList = FlightListDatabase()
        if flightList.readRankFlightList():
            flightList.checkRankFligthListHeaders()
            assert flightList.checkRankFligthListHeaders() ==  True
            
        if flightList.readTrainFlightList():
            flightList.checkTrainFlightListHeaders()
            assert flightList.checkTrainFlightListHeaders() ==  True

    def test_main_four(self):
        
        logging.basicConfig(level=logging.DEBUG)

        print("---------------- collect unique airports ICAO codes  ----------------")
        
        flightList = FlightListDatabase()
        assert ( flightList.readRankFlightList() == True )
        assert ( flightList.readTrainFlightList() == True )
        
        flightList.collectUniqueAirports()
        
        
    def test_main_five(self):
        
        logging.basicConfig(level=logging.DEBUG)

        print("---------------- extend flight list with airport data ----------------")
        
        flightList = FlightListDatabase()
        assert ( flightList.readTrainFlightList() == True )
        assert ( flightList.readRankFlightList() == True )
        
        assert flightList.checkTrainFlightListHeaders() == True
        assert flightList.checkRankFligthListHeaders() == True
        #flightList.collectUniqueAirports()
        
        flightList.extendTrainFlightListWithAirportData()
        
    def test_main_six(self):
            
        logging.basicConfig(level=logging.INFO)

        print("---------------- extend flight data with flight list data ??? ----------------")
        
        flightList = FlightListDatabase()
        assert ( flightList.readTrainFlightList() == True )
        
        flightList.extendTrainFlightDataWithFlightListData()
        
    def test_main_seven(self):

        logging.basicConfig(level=logging.INFO)

        print("---------------- extend flight list with aircraft data----------------")
        
        flightList = FlightListDatabase()
        assert ( flightList.readTrainFlightList() == True )
        
        #assert flightList.checkTrainFlightListHeaders()
        
        print(tabulate(flightList.getTrainFlightListDataframe()[:10], headers='keys', tablefmt='grid' , showindex=True , ))
        
        assert flightList.collectUniqueAircraftTypesFromTrainFlightList() == True

        
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    unittest.main()