'''
Created on 12 oct. 2025

@author: robert


exploring regression with tensor flow
https://www.youtube.com/watch?v=XJFSXH8E6CA 
'''

import pandas as pd
from tensorflow.python.ops import inplace_ops
# Set the option to display all columns
pd.options.display.max_columns = None

import numpy as np 
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

from tabulate import tabulate

''' warning - use tensor flow 2.12.0 not the latest 2.20.0 that is causing DLL problems '''
import tensorflow as tf

from trajectory.Fuel.FuelReader import FuelDatabase
from trajectory.Flights.FlightsReader import FlightsDatabase

import logging
import unittest

''' load static flight database '''
flightsDatabase = FlightsDatabase()

def extendFuelTrainWithFlightsData( row ):
    
    print(''' ------------- row by row loop ------------------''')
    print ( row['flight_id'] )
    df_flightData = flightsDatabase.readOneTrainFile( row['flight_id'] )
    #print( str( df_flightData['timestamp'] ))
    df_filtered = df_flightData[ (df_flightData['timestamp'] >= row['fuel_burn_start']) & (df_flightData['timestamp'] <= row['fuel_burn_end'])]
    # keep first row only
    df_filtered = df_filtered.head(1)
    df_filtered = df_filtered.drop ( "flight_id" , axis = 1)
    
    print(str ( df_filtered.shape ))
    print(str ( list ( df_filtered ) ) )
    
    #return  pd.Series( df_filtered['timestamp'] ).reset_index(drop=True)
    #return  pd.Series(df_filtered['timestamp']).reset_index(drop=True)

    #return pd.concat( [ pd.Series(df_filtered['timestamp']) , pd.Series(df_filtered['aircraft_type_code']) ] , axis = 1).reset_index(drop=True, inplace=True)
    #return df_filtered[ ['timestamp','aircraft_type_code'] ].reset_index( drop = False , inplace = True)
    #return pd.Series( { 'timestamp' : df_filtered['timestamp'] , 'aircraft_type_code' : df_filtered['aircraft_type_code'] } ).reset_index(drop=True,inplace=True)
    series = pd.Series( { 'timestamp' : (df_filtered['timestamp'].iloc[0]) , 'aircraft_type_code' : str(df_filtered['aircraft_type_code'].iloc[0]) } )
    
    return series


#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        logging.basicConfig(level=logging.INFO)
        
        
        logging.info (' -------------- Train Fuel database -------------')
        
        fuelDatabase = FuelDatabase()
                
        assert fuelDatabase.readFuelTrain() == True
        assert fuelDatabase.checkFuelTrainHeaders() == True
        
        df = fuelDatabase.getFuelTrainDataframe()
        
        print("train fuel dataframe shape = " + str( df.shape ))
        
        # Pretty print the DataFrame
        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        
#       for index, row in df.iterrows():
#            print(f"Index: {index}, flight id: {row['flight_id']} ")
        print ("final shape = " +  str (  df .shape ) ) 
        
        #df['timestamp'] = df.apply( extendFuelTrainWithFlightsData , axis = 1)
        listOfColumns = ['timestamp','aircraft_type_code', 'latitude', 'longitude', 'altitude', 'groundspeed', 'track', 'vertical_rate', 'mach', 'TAS', 'CAS', 'source']
        #df[listOfColumns] = df.apply( extendFuelTrainWithFlightsData , axis = 1, result_type='expand')
        
        df[['timestamp','aircraft_type_code']] = df.apply( extendFuelTrainWithFlightsData , axis = 1 )
        print ( "after apply = " +  str ( df.shape ) )
        #df[ ['timestamp','aircraft_type_code'] ] = 
        #applied_df = df.apply( extendFuelTrainWithFlightsData , axis = 1 , result_type='expand')
        
        print (  str ( df.shape ) )
        print ( str ( list  (  df )))
        
        
        print ("final list = " +  str ( list ( df )))
        print ("final shape = " +  str (  df .shape ) ) 
        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("tensorflow version = " + tf.__version__)
    print("pandas version = " + pd. __version__)
    
    unittest.main()
