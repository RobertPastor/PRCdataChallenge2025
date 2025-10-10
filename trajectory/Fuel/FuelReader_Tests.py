'''
Created on 8 oct. 2025

@author: robert
'''

import logging
import unittest
import pandas as pd
import numpy as np

from trajectory.Fuel.FuelReader import FuelDatabase

# Set the option to display all columns
pd.options.display.max_columns = None

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        
        logging.info("---------------- test_main_one  ----------------")

        logging.basicConfig(level=logging.INFO)
        fuelDatabase = FuelDatabase()
        
        assert fuelDatabase.readFuelTrain() == True
        
    def test_main_two(self):
        
        logging.info("---------------- test_main_two  ----------------")

        logging.basicConfig(level=logging.INFO)
        
        logging.info("Read Fuel Train")

        fuelDatabase = FuelDatabase()
        
        assert fuelDatabase.readFuelTrain() == True
        assert fuelDatabase.checkFuelTrainHeaders() == True
        
    def test_main_three(self):
        
        logging.basicConfig(level=logging.INFO)
        logging.info("---------------- test_main_three  ----------------")

        logging.info("Read Fuel Rank")

        fuelDatabase = FuelDatabase()
        
        assert fuelDatabase.readFuelRank() == True
        assert fuelDatabase.checkFuelRankHeaders() == True
        
        df = fuelDatabase.getFuelRankDataframe()
        print ( str ( df.shape ))
        
        print ( str ( df.sample(10) ))
        
        print ( str(  list ( df )) )


        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    logging.basicConfig(level=logging.DEBUG)

    print(pd. __version__)
    
    unittest.main()