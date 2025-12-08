'''
Created on 7 déc. 2025

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

    def test_Train_flightlist_2_AircraftCodes(self):
        print("------------test_main_one----------------")
        
        train_rank_final = "train"
        trainFlightList = FlightListDatabase(train_rank_final)
        
        assert trainFlightList.readTrainFlightListLite()
        df = trainFlightList.collectUniqueAircrafts ()
        print(df)

        
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    unittest.main()
