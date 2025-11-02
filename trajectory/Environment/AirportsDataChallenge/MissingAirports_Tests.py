'''
Created on 2 nov. 2025

@author: robert
'''



import logging
import unittest
import pandas as pd
from trajectory.Environment.AirportsDataChallenge.AirportsDataChallengeDatabaseFile import AirportsDataChallengeDatabase
from trajectory.FlightList.FlightListReader import FlightListDatabase
from tabulate import tabulate

#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        print("------------test_main_one----------------")

        logging.basicConfig(level=logging.INFO)
        
        logging.info("Read Data Challenge Airports")
        
        airportsDb = AirportsDataChallengeDatabase()
        assert airportsDb.read() == True
        
        missingsAirportsList = ["EGHQ","LTDB","MMSM","OKKK","SEQM","SPJC","ZUTF"]
        missingsAirportsList = ["ZGOW" ,"VTBS"]
        for airportICAOcode in missingsAirportsList:
            print( airportICAOcode )
            print (airportsDb.isAirportInDatabase(airportICAOcode) )
        
        
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()