'''
Created on 25 déc. 2025

@author: robert
'''
import unittest

import logging
import unittest
import pandas as pd
from trajectory.Environment.AirportsDataChallenge.AirportsDataChallengeDatabaseFile import AirportsDataChallengeDatabase
from trajectory.FlightList.FlightListReader import FlightListDatabase
from tabulate import tabulate


#============================================
class Test_Main(unittest.TestCase):

    def test_Rank_flightlist_4_AircraftCode(self):
        print("------------test_main rank----------------")
        
        train_rank_final = "rank"
        flightList = FlightListDatabase(train_rank_final)
        assert flightList.readRankFlightListLite()
        
        flight_ids_list = flightList.collectFlightIdsForOneAircraftType('A320')
        print ( flight_ids_list )
        print ( len(flight_ids_list) )
        
        
    def test_Train_flightlist_4_AircraftCode(self):
        print("------------test_main train----------------")
        
        train_rank_final = "train"
        flightList = FlightListDatabase(train_rank_final)
        assert flightList.readTrainFlightListLite()
        
        flight_ids_list = flightList.collectFlightIdsForOneAircraftType('A320')
        print ( flight_ids_list )
        print ( len(flight_ids_list) )



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    unittest.main()
    

