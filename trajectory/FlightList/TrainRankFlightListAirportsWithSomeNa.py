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

from trajectory.FlightList.FlightListReader import FlightListDatabase
from tabulate import tabulate

#============================================
class Test_Main(unittest.TestCase):

    def test_Train_flightlist_not_in_airport_DropNa(self):
        print("------------test_main_one----------------")

        logging.basicConfig(level=logging.INFO)
        
        logging.info("Read Data Challenge Airports")
        
        airportsDb = AirportsDataChallengeDatabase()
        assert airportsDb.read() == True
        
        airportsDataframe = airportsDb.getAirportsDataframe()
        
        trainRankFlightList = FlightListDatabase()
        assert ( trainRankFlightList.readTrainFlightListLite() )
        assert ( trainRankFlightList.readRankFlightListLite() )
        
        print ( trainRankFlightList.getTrainFlightListDataframe().shape )
        print ( trainRankFlightList.getRankFlightListDataframe().shape )
        
        df = pd.concat( [ trainRankFlightList.getTrainFlightListDataframe() , trainRankFlightList.getRankFlightListDataframe()] , axis=0)
        
        
        
        


        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
    