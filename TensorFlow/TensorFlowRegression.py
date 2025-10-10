'''
Created on 10 oct. 2025

@author: robert
'''

import pandas as pd
import numpy as np 

''' warning - use tensor flow 2.12.0 not the latest 2.20.0 that is causing DLL problems '''
import tensorflow as tf

from trajectory.FlightList.FlightListReader import FlightListDatabase
from trajectory.Fuel.FuelReader import FuelDatabase


import logging
logging.basicConfig(level=logging.INFO)

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

print(tf.__version__)

print (' -------------- Train Flight list -------------')


flightList = FlightListDatabase()
assert ( flightList.readTrainFlightList() == True )
assert ( flightList.readRankFlightList() == True )

assert flightList.checkTrainFlightListHeaders() == True
assert flightList.checkRankFligthListHeaders() == True

assert flightList.extendTrainFlightListWithAirportData()
assert flightList.extendTrainFlightDataWithFlightListData() == True

assert flightList.extendTrainFlightDataWithFlightListData()

df_concat = flightList.getTrainFlightDataWithFlightListData()
logging.info( str ( df_concat.shape ))
logging.info ( str ( df_concat.isnull().sum() ))

print (' -------------- fuel -------------')

fuelDatabase = FuelDatabase()
        
assert fuelDatabase.readFuelTrain() == True
assert fuelDatabase.checkFuelTrainHeaders() == True

logging.info( str ( fuelDatabase.getFuelTrainDataframe().sample(10) ))
logging.info ( str ( fuelDatabase.getFuelTrainDataframe().isnull().sum() ))
