'''
Created on 10 oct. 2025

@author: robert

exploring regression with tensor flow
https://www.youtube.com/watch?v=XJFSXH8E6CA 
'''

import pandas as pd
import numpy as np 
# Set the option to display all columns
pd.options.display.max_columns = None

''' warning - use tensor flow 2.12.0 not the latest 2.20.0 that is causing DLL problems '''
import tensorflow as tf

from trajectory.FlightList.FlightListReader import FlightListDatabase
from trajectory.Fuel.FuelReader import FuelDatabase

import logging
logging.basicConfig(level=logging.INFO)

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

print(tf.__version__)

logging.info (' -------------- Train Flight list -------------')

flightList = FlightListDatabase()
assert ( flightList.readTrainFlightList() == True )
assert ( flightList.readRankFlightList() == True )

assert flightList.checkTrainFlightListHeaders() == True
assert flightList.checkRankFligthListHeaders() == True

assert flightList.extendTrainFlightListWithAirportData()
assert flightList.extendTrainFlightDataWithFlightListData() == True

df_concat = flightList.getTrainFlightDataWithFlightListData()

logging.info( str ( df_concat.shape ))
logging.info ( str ( df_concat.isnull().sum() ))
logging.info ( str ( df_concat.dtypes ))
logging.info ( str ( df_concat.sample(10) ))

logging.info (' -------------- aircraft type  -------------')
logging.info( str ( df_concat['aircraft_type'].value_counts()))

logging.info (' -------------- scale the values  -------------')



logging.info (' -------------- fuel -------------')

fuelDatabase = FuelDatabase()
        
assert fuelDatabase.readFuelTrain() == True
assert fuelDatabase.checkFuelTrainHeaders() == True

logging.info( str ( fuelDatabase.getFuelTrainDataframe().sample(10) ))
logging.info ( str ( fuelDatabase.getFuelTrainDataframe().isnull().sum() ))

logging.info ( str ( fuelDatabase.getFuelTrainDataframe().dtypes ))


