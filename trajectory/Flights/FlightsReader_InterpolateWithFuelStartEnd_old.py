'''
Created on 9 nov. 2025

@author: robert
'''
from trajectory.Flights.FlightsReader import FlightsDatabase

import os

import logging
import unittest
import pandas as pd
import os
import numpy as np
from trajectory.Flights.FlightsReader import FlightsDatabase
from tabulate import tabulate
from trajectory.Fuel.FuelReader import FuelDatabase
from tabulate import tabulate

from trajectory.utils import keepOnlyColumns

def fill_Fuel_Frame_with_empty_columns_for_interpolation(df):
    
    listOfColumns = ['latitude', 'longitude', 'altitude', 'groundspeed', 'track', 'vertical_rate', 'mach', 'TAS', 'CAS']
    # Add an empty column (filled with NaN)
    for columnName in listOfColumns:
        df[columnName] = np.nan
    
    return df

def build_Fuel_Dataframe_from_start_end( fuelTrainDataframe ):
    
    fuelStartDataframe = fuelTrainDataframe.copy()
    listOfColumnNamesToKeep = ['flight_id', 'start']
    fuelStartDataframe = keepOnlyColumns( fuelStartDataframe, listOfColumnNamesToKeep)
    fuelStartDataframe = fuelStartDataframe.rename(columns={'start': 'timestamp'})
    print ( list ( fuelStartDataframe ))
    print ( fuelStartDataframe.shape  )
    
    fuelEndDataframe = fuelTrainDataframe.copy()
    listOfColumnNamesToKeep = ['flight_id', 'end']
    fuelEndDataframe = keepOnlyColumns( fuelEndDataframe, listOfColumnNamesToKeep)
    fuelEndDataframe = fuelEndDataframe.rename(columns={'end': 'timestamp'})
    print ( list ( fuelEndDataframe ))
    print ( fuelEndDataframe.shape  )

    return fuelStartDataframe , fuelEndDataframe

#============================================
class Test_Main(unittest.TestCase):
    
    def test_main_interpolate_all(self):
        
        train_rank = "train"
        ''' open fuel database '''
        count_of_files = 10
        fuelDatabase = FuelDatabase(count_of_files)
        
        fuelTrainDataframe = fuelDatabase.readFuelTrainLite()
        print ( list ( fuelTrainDataframe ))
        
        fuelTrainReducedWithStartDataframe , fuelTrainReducedWithEndDataframe = build_Fuel_Dataframe_from_start_end (fuelTrainDataframe)
        fuelTrainDataframe = pd.concat( [fuelTrainReducedWithStartDataframe , fuelTrainReducedWithEndDataframe] )
        print ( fuelTrainDataframe.shape )
        print ( list ( fuelTrainDataframe ) )
        
        ''' filter on one flight id '''
        fuelTrainDataframe = fuelTrainDataframe[fuelTrainDataframe['flight_id'] == "prc770864956"]
        fuelTrainDataframe = fuelTrainDataframe.sort_values(by='timestamp')

        print(tabulate(fuelTrainDataframe[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        
        '''loop through the files '''
        flightsDatabase = FlightsDatabase()
        
        directory = flightsDatabase.getTrainFlightsFolderPathStr()
        maxCount = 10
        count = 0
        for fileName in os.listdir(directory):
            if count < maxCount:
                if fileName.endswith(".parquet"): # Filter specific file types
                    filePath = os.path.join(directory, fileName)
                    print ( filePath )
                    
                    flight_id = fileName.split(".")[0]
                    print("flight_id = " + flight_id)
                    if ( flight_id == "prc770864956"):
                    
                        flightTrainDataframe = flightsDatabase.readOneTrainFileLite(fileName)
                        print ( flightTrainDataframe.shape )
                        
                        ''' filter fuel on flight id and perform concat '''
                        ''' in order for the fuel start and end to exist as new rows in the flight dataframe '''                    
                        filteredFuelDataframe = fuelTrainDataframe[fuelTrainDataframe['flight_id'] == flight_id]
                        print ( filteredFuelDataframe.shape )
    
                        ''' concat the dataframe '''
                        flightTrainDataframe = pd.concat ( [flightTrainDataframe , filteredFuelDataframe])
                        print ( flightTrainDataframe.shape )
                        
                        flightTrainDataframe = flightTrainDataframe.sort_values(by='timestamp')
                        
                        print(tabulate(flightTrainDataframe[:10], headers='keys', tablefmt='grid' , showindex=False , ))
                        #print(tabulate(flightTrainDataframe.describe().transpose(), headers='keys', tablefmt='grid' , showindex=False , ))
    
                        #null_rows = flightTrainDataframe[flightTrainDataframe['typecode'].isnull()]
                        #print(tabulate(null_rows, headers='keys', tablefmt='grid' , showindex=False , ))
                        
                        flightTrainDataframe = flightTrainDataframe.interpolate(limit_direction='both')
                        #print(tabulate(flightTrainDataframe[:10], headers='keys', tablefmt='grid' , showindex=False , ))
                        print(tabulate(flightTrainDataframe[-10:], headers='keys', tablefmt='grid' , showindex=False , ))

                        print ( "-"*80 )
                    
                    
            count = count + 1
        
      
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(pd. __version__)
    
    unittest.main()