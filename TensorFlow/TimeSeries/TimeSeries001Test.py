'''
Created on 25 déc. 2025

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

    def test_Rank_flightlist_4_AircraftCode(self):
        print("------------test_main rank----------------")
        
        aircraft_type_code = "A320"
        
        train_rank_final = "rank"
        flightList = FlightListDatabase(train_rank_final)
        assert flightList.readRankFlightListLite()
        
        df = flightList.getRankFlightListDataframe()
        df = df[df['aircraft_type']==aircraft_type_code]
        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        
        print("------------------ departure airport -----------------------")
        
        origin_airport_list = df['origin_icao'].unique().tolist()
        print ( origin_airport_list )
        
        for airport_code in origin_airport_list:
            df_airport = df[df['origin_icao']==airport_code]
            #print(tabulate(df_airport[:10], headers='keys', tablefmt='grid' , showindex=False , ))

        ''' most used departure airports '''
        airports_origin_destination = [
            {"adep":"KLAX","ades":"MSLP"},{"adep":"MMSP","ades":"KDFW"},
            {"adep":"KMDW","ades":"MMLO"},{"adep":"KMIA","ades":"SKBO"},{"adep":"MMSD","ades":"KOAK"},
            {"adep":"MMUN","ades":"SKBO"},{"adep":"MMLO","ades":"KORD"},{"adep":"SKBO","ades":"KIAD"},
            {"adep":"MNMG","ades":"KMIA"},{"adep":"KJFK","ades":"MSLP"},{"adep":"KORD","ades":"MMGL"},
            {"adep":"KIAH","ades":"MMGL"},{"adep":"SKRG","ades":"KJFK"},{"adep":"MSLP","ades":"KIAD"}]
        for airport_pair in airports_origin_destination:
            print ("adep = {0} - ades = {1}".format( airport_pair["adep"] , airport_pair["ades"] ) )
            #===================================================================
            # ]
            #===================================================================

        print("------------------ flight ids for A320-----------------------")

        flight_ids_list = flightList.collectFlightIdsForOneAircraftType(aircraft_type_code)
        print ( flight_ids_list )
        print ( "flight is list size = {0}".format( len(flight_ids_list) ) )
        
        
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    unittest.main()