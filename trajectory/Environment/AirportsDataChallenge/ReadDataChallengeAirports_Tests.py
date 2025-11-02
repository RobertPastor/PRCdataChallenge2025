'''
Created on 7 oct. 2025

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
        
        if airportsDb.read():
            logging.info("Data Challenge Airports - correctly read")
        else:
            logging.error("Data Challenge Airports - read failed")
            
    def test_main_two(self):
        logging.basicConfig(level=logging.INFO)

        print("------------test_main_two----------------")

        airportsDb = AirportsDataChallengeDatabase()
        if airportsDb.read():
            ParisCDG = "LFPG"
            
            logging.info( airportsDb.getAirPort( ParisCDG ))
            assert ( not ( airportsDb.getAirPort( ParisCDG ) is None ))
            
            NewquayCornwallAirport = "EGHQ"
            
            print ( airportsDb.isAirportInDatabase(NewquayCornwallAirport))
            print ( airportsDb.isAirportInDatabase("LTDB"))
            
            
            logging.info( airportsDb.getAirPort( NewquayCornwallAirport ))
            assert ( not ( airportsDb.getAirPort( NewquayCornwallAirport ) is None ))
            
    def test_main_three(self):
        
        logging.basicConfig(level=logging.INFO)

        print("------------test_main_three----------------")
        airportsDb = AirportsDataChallengeDatabase()
        if airportsDb.read():
            assert airportsDb.checkHeaders () == True
            logging.info("both expected and read column list are identical")
            
    def test_airports_in_flightlist_not_in_airports(self):
        
        airportsDb = AirportsDataChallengeDatabase()
        assert airportsDb.read() == True
        assert airportsDb.checkHeaders() == True
        
        airportsDataframe = airportsDb.getAirportsDataframe()
        
        print("------------test airports not in flight list----------------")
        flightListDatabase = FlightListDatabase()
        flightListDatabase.readRankFlightList()
        
        df_flightList = flightListDatabase.getRankFlightListDataframe()
        print ( df_flightList.shape )

        rankFlightListDataframeRowCount = df_flightList.shape[0]
        
        df_flightListMerged = pd.merge ( df_flightList , airportsDataframe , left_on='origin_icao', right_on='airport_icao', how='inner' )

        ''' not merged airports '''
        notMergedDf = df_flightList[~df_flightList.isin(df_flightListMerged)].dropna()
        print ( notMergedDf.shape )
        for airport in notMergedDf['origin_icao']:
            print ( airport )
        print(tabulate(notMergedDf[:40], headers='keys', tablefmt='grid' , showindex=True , ))
        
        assert df_flightListMerged.shape[0] == rankFlightListDataframeRowCount


        
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()