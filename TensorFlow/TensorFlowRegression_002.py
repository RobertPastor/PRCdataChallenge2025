'''
Created on 12 oct. 2025

@author: robert


exploring regression with tensor flow
https://www.youtube.com/watch?v=XJFSXH8E6CA 
'''


from functools import lru_cache

import pandas as pd
import time
from sklearn.preprocessing._encoders import OneHotEncoder
# Set the option to display all columns
pd.options.display.max_columns = None

import numpy as np 
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

from tabulate import tabulate
from trajectory.utils import dropUnusedColumns , oneHotEncodeSource

''' warning - use tensor flow 2.12.0 not the latest 2.20.0 that is causing DLL problems '''
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler

from trajectory.Fuel.FuelReader import FuelDatabase
from trajectory.Flights.FlightsReader import FlightsDatabase

import logging
import unittest



''' Root mean square between prediction and actual values '''
def rmse(y_true, y_pred):
    return backend.sqrt( backend.mean (backend.square(y_pred - y_true)))

transformer = make_column_transformer( 
    (MinMaxScaler(), [ 'time_diff_seconds', 'fuel_flow_kg_sec', 'origin_longitude', 'origin_latitude', 'origin_elevation', 
    'destination_longitude', 'destination_latitude', 'destination_elevation', 'flight_distance_Nm', 'flight_duration_sec', 
    'fuel_burn_relative_start', 'fuel_burn_relative_end',  'latitude', 'longitude', 'altitude', 'groundspeed', 'track', 'vertical_rate', 'mach', 'TAS', 'CAS'] ) ,
    (OneHotEncoder(handle_unknown='ignore') , ['aircraft_type_code' , 'source'] ) )

''' load static flight database '''
flightsDatabase = FlightsDatabase()


def extendFuelTrainWithFlightsData( row ):
    
    #print(''' ------------- row by row loop ------------------''')
    print ( row['flight_id'] )
    df_flightData = flightsDatabase.readOneTrainFile( row['flight_id'] )
    #print( str( df_flightData['timestamp'] ))
    df_filtered = df_flightData[ (df_flightData['timestamp'] >= row['fuel_burn_start']) & (df_flightData['timestamp'] <= row['fuel_burn_end'])]
    # keep first row only
    df_filtered = df_filtered.head(1)
    df_filtered = df_filtered.drop ( "flight_id" , axis = 1)
    
    #print(str ( df_filtered.shape ))
    if df_filtered.shape[0] == 0:
        
        return pd.Series( { 'timestamp' : (row['fuel_burn_start']) , 'aircraft_type_code' : str('unknown') ,
                         'latitude' : (0.0) , 'longitude' : (0.0) ,
                         'altitude' : (0.0) , 'groundspeed' : (0.0) , 
                         'track' : (0.0) , 'vertical_rate' : (0.0) ,
                         'mach' : (0.0) , 'TAS' : (0.0) , 
                         'CAS' : (0.0) , 'source' : (0.0)} )
    else:
        return pd.Series( { 'timestamp' : (df_filtered['timestamp'].iloc[0]) , 'aircraft_type_code' : str(df_filtered['aircraft_type_code'].iloc[0]) ,
                         'latitude' : (df_filtered['latitude'].iloc[0]) , 'longitude' : (df_filtered['longitude'].iloc[0]) ,
                         'altitude' : (df_filtered['altitude'].iloc[0]) , 'groundspeed' : (df_filtered['groundspeed'].iloc[0]) , 
                         'track' : (df_filtered['track'].iloc[0]) , 'vertical_rate' : (df_filtered['vertical_rate'].iloc[0]) ,
                         'mach' : (df_filtered['mach'].iloc[0]) , 'TAS' : (df_filtered['TAS'].iloc[0]) , 
                         'CAS' : (df_filtered['CAS'].iloc[0]) , 'source' : (df_filtered['source'].iloc[0])} )


#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        logging.basicConfig(level=logging.INFO)
        start_time = time.time()
        logging.info (' -------------- Train Fuel database -------------')
                
        fuelDatabase = FuelDatabase()
                
        assert fuelDatabase.readFuelTrain() == True
        assert fuelDatabase.checkFuelTrainHeaders() == True
        
        df = fuelDatabase.getFuelTrainDataframe()
        
        print("train fuel dataframe shape = " + str( df.shape ))
        
        #df['timestamp'] = df.apply( extendFuelTrainWithFlightsData , axis = 1)
        listOfFlightListColumns = ['timestamp','aircraft_type_code', 'latitude', 'longitude', 'altitude', 'groundspeed', 'track', 'vertical_rate', 'mach', 'TAS', 'CAS', 'source']
        #df[listOfColumns] = df.apply( extendFuelTrainWithFlightsData , axis = 1, result_type='expand')
        
        df[listOfFlightListColumns] = df.apply( extendFuelTrainWithFlightsData , axis = 1 )
        
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        
        print ( "shape after apply = " +  str ( df.shape ) )
        print ("final list = " +  str ( list ( df )))
        print ("final shape = " +  str (  df .shape ) ) 
        
        ''' drop columns with absolute date time instant '''
        df = dropUnusedColumns( df , ['fuel_burn_start','fuel_burn_end'])
        #print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        
        ''' convert flight data time stamp relative to flight start '''
        df['timestamp_relative_start'] = ( df['timestamp'] - df['takeoff']).dt.total_seconds()
        
        ''' drop absolute date time stamp '''
        df = dropUnusedColumns( df , ['timestamp','takeoff','flight_id'] )

        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        
        #df = df.dropna(axis = 'index' , how = 'any')
        print ("final shape = " +  str (  df .shape ) ) 
        
        listOfColumnsToDrop = ['idx','fuel_kg']
        X_train = df.drop( listOfColumnsToDrop, axis=1)
        y_train = df['fuel_kg']
        
        ''' one hot encoder for aircraft_type_code  and source '''
        for columnName in ['aircraft_type_code' , 'source']:
            X_train = oneHotEncodeSource( X_train , columnName)
            
        ''' fill not a number with zeros '''
        X_train = X_train.fillna(0)
            
        print(" ---- after one hot encoder ------")
        print(tabulate(X_train[:10], headers='keys', tablefmt='grid' , showindex=True , ))
        
        print( str ( df.isnull().sum() ))

        '''  Specify columns to rescale '''
        columns_to_rescale = [ 'time_diff_seconds', 'fuel_flow_kg_sec', 'origin_longitude', 'origin_latitude', 'origin_elevation', 
            'destination_longitude', 'destination_latitude', 'destination_elevation', 'flight_distance_Nm', 'flight_duration_sec', 
                'fuel_burn_relative_start', 'fuel_burn_relative_end',  'latitude', 'longitude', 'altitude', 'groundspeed', 'track', 'vertical_rate', 'mach', 'TAS', 'CAS']
        
        '''  Apply MinMaxScaler '''
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train[columns_to_rescale] = scaler.fit_transform(X_train[columns_to_rescale])
        
        print ( str ( X_train.shape ))
        print(tabulate(X_train[:10], headers='keys', tablefmt='grid' , showindex=True , ))
        
        ''' convert True False to float '''
        X_train = np.asarray(X_train).astype(np.float32)
        
        print(tabulate(X_train[:10], headers='keys', tablefmt='grid' , showindex=True , ))
       
        ''' declare the model '''
        #tf.random.set_seed ( 42 )
        model = Sequential ( [ Dense( 256 , activation = 'relu' ),
                              Dense( 256 , activation = 'relu' ),
                              Dense( 128 , activation = 'relu' ),
                              Dense(1)])
        
        model.compile(loss = rmse , optimizer = 'adam' , metrics = [rmse])
        model.fit( X_train , y_train , epochs = 100 )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("tensorflow version = " + tf.__version__)
    print("pandas version = " + pd. __version__)
    
    unittest.main()
