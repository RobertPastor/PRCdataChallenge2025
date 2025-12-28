'''
Created on 25 déc. 2025

@author: robert
'''


import logging
import unittest
import pandas as pd
from trajectory.FlightList.FlightListReader import FlightListDatabase
from tabulate import tabulate

#============================================
class Test_Main(unittest.TestCase):

    def test_Rank_flightlist_4_AircraftCode(self):
        
        print("------------test_main train----------------")
        
        ''' use train dataset to beneficiate from fuel consumption data '''
        aircraft_type_code = "A320"
        
        train_rank_final = "train"
        flightList = FlightListDatabase(train_rank_final)
        assert flightList.readTrainFlightListLite()
        
        df = flightList.getTrainFlightListDataframe(train_rank_final)
        ''' filter on unique aircraft '''
        df = df[df['aircraft_type']==aircraft_type_code]
        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        
        
        print("------------------ departure airport -----------------------")
        
        origin_airport_list = df['origin_icao'].unique().tolist()
        print ( origin_airport_list )
        
        results = {}
        count = 0
        for origin_airport_code in origin_airport_list:
            df_origin_airports = df[df['origin_icao']==origin_airport_code]
            #print(tabulate(df_origin_airports[:10], headers='keys', tablefmt='grid' , showindex=False , ))
            ''' find top 10 routes with most availables flight data / trajectories '''
            
            for index, row in df_origin_airports.iterrows():
                print(row)
                print(f"Index: {index}, origin: {row['origin_icao']}, Age: {row['destination_icao']}")

                origin_icao = row['origin_icao']
                destination_icao = row['destination_icao']
                route = origin_icao + "-" + destination_icao
                print(route)
                if ( route in results):
                    count = results[route]["count"]
                    results[route] = {"route": route , "count":count+1}
                else:
                    results[route] = {"route": route , "count":1}

        top_most_flown_route = ""
        top_most_flown_routes_max = 0
        for result in results:
            #print(result + " - " + str( results[result]) )
            if results[result]["count"] > top_most_flown_routes_max:
                top_most_flown_route = result
                top_most_flown_routes_max = results[result]["count"]
            
        print ( "most flown route -> {0} - max = {1}".format(top_most_flown_route , top_most_flown_routes_max))
        for result in results:
            if ( result == top_most_flown_route):
                print(result + " - " + str( results[result]) )
                origin_airport = str(result).split("-")[0]
                print( origin_airport )
        
        ''' most used departure airports '''
        

        print("------------------ flight ids for A320-----------------------")

        flight_ids_list = flightList.collectFlightIdsForOneAircraftType(aircraft_type_code)
        print ( flight_ids_list )
        print ( "flight is list size = {0}".format( len(flight_ids_list) ) )
        
        
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    unittest.main()