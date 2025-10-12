'''
Created on 12 oct. 2025

@author: robert


exploring regression with tensor flow
https://www.youtube.com/watch?v=XJFSXH8E6CA 
'''

import pandas as pd
# Set the option to display all columns
pd.options.display.max_columns = None

import numpy as np 
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

from tabulate import tabulate

''' warning - use tensor flow 2.12.0 not the latest 2.20.0 that is causing DLL problems '''
import tensorflow as tf

from trajectory.Fuel.FuelReader import FuelDatabase
from trajectory.Flights import FlightsReader

import logging
import unittest


#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        logging.basicConfig(level=logging.INFO)
        
        
        logging.info (' -------------- Train Fuel database -------------')
        
        fuelDatabase = FuelDatabase()
                
        assert fuelDatabase.readFuelTrain() == True
        assert fuelDatabase.checkFuelTrainHeaders() == True
        df = fuelDatabase.getFuelTrainDataframe()
        
        print("train fuel datafram shape = " + str( df.shape ))
        
        # Pretty print the DataFrame
        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        
        flightsReader = FlightsReader()
        



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("tensorflow version = " + tf.__version__)
    print("pandas version = " + pd. __version__)
    
    unittest.main()
