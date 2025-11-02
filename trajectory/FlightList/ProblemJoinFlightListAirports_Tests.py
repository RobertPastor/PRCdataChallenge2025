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

    def test_main_one(self):
        print("------------test_main_one----------------")

        logging.basicConfig(level=logging.INFO)
        
        logging.info("Read Data Challenge Airports")
        
        airportsDb = AirportsDataChallengeDatabase()
        assert airportsDb.read() == True
        
        airportsDataframe = airportsDb.getAirportsDataframe()
        
        rankFlightList = FlightListDatabase()
        assert ( rankFlightList.readRankFlightListLite() )
        
        rankFligthListDataframe = rankFlightList.getRankFlightListDataframe()
        initialFlightListDataframe = rankFlightList.getRankFlightListDataframe()
        
        print ( rankFligthListDataframe.shape )
        
        ''' ------------merge on origin icao ----- '''
        merged_df = pd.merge ( rankFligthListDataframe , airportsDataframe , 
                                left_on='origin_icao', right_on='airport_icao', how='inner' )
        
        print ( merged_df.shape )
        
        # Filter rows not in df2
        result = initialFlightListDataframe[~initialFlightListDataframe.isin(merged_df.to_dict(orient='list')).all(axis=1)]
        print ( tabulate( result , headers='keys', tablefmt='grid' , showindex=True , ))

        ''' ------ merge on destination icao '''
        merged_df = pd.merge ( rankFligthListDataframe , airportsDataframe , 
                                left_on='destination_icao', right_on='airport_icao', how='inner' )
        
        print ( merged_df.shape )
        
        # Filter rows not in df2
        result = initialFlightListDataframe[~initialFlightListDataframe.isin(merged_df.to_dict(orient='list')).all(axis=1)]
        print ( tabulate( result , headers='keys', tablefmt='grid' , showindex=True , ))


        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()