'''
Created on 19 nov. 2025

@author: robert

'''

import os
import logging
import unittest
from tabulate import tabulate

from trajectory.FlightList.FlightListReader import FlightListDatabase
from trajectory.Flights.FlightsReader import FlightsDatabase

#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        
        print("---------------- Read Train flight list  ----------------")
        
        logging.basicConfig(level=logging.DEBUG)
        train_rank_final = "train"
        logging.info("Read Flight List")
        
        flightList = FlightListDatabase()
        if flightList.readTrainRankFinalFlightListLite(train_rank_final):
            logging.info("train flight list read correctly")
            
        df_flightList = flightList.getTrainFlightListDataframe()
        print(tabulate(df_flightList[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        
        count = 0
        for index, row in df_flightList.iterrows():
            #count = count + 1
            if count > 100:
                break
            
            print(f"----- Index: {index} , Name: { row['flight_id'] } ----- ")
            flight_id = row['flight_id']
            print(flight_id)
            
            flightsData = FlightsDatabase()
            df_flightsData  =  flightsData.readOneFlightFileLite(train_rank_final , flight_id )
            
            interpolatedColumnList = ['latitude', 'longitude','altitude','groundspeed','track','vertical_rate', 'mach', 'TAS', 'CAS']
            
            for column in interpolatedColumnList:
                nan_count_col1 = df_flightsData[column].isna().sum()
                print(f"NaN values in {column}: {nan_count_col1}")
            print("-"*90)
                
            df_flightsData[interpolatedColumnList] = df_flightsData[interpolatedColumnList] \
                            .interpolate(method='linear', axis=0, Direction='forward', limit_area="inside")
                            
            df_flightsData[interpolatedColumnList] = df_flightsData[interpolatedColumnList] \
                            .interpolate(method='linear', axis=0, Direction='backward', limit_area="inside")

            for column in interpolatedColumnList:
                nan_count_col1 = df_flightsData[column].isna().sum()
                print(f"NaN values in {column}: {nan_count_col1}")
                
            ''' generate a parquet file '''
            path = flightsData.getFlightsInterpolatedFolderPathStr(train_rank_final)
            path = os.path.join(path , flight_id + ".parquet")
            print ( path )
            df_flightsData.to_parquet(path, index = False)
            print("-"*90)
            print("-"*90)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()