'''
Created on 8 oct. 2025

@author: robert
'''

import logging
import unittest
import pandas as pd
import os
from trajectory.Fuel.FuelReader import FuelDatabase

#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        
        print("---------------- test_main_one  ----------------")

        logging.basicConfig(level=logging.INFO)
        fuelDatabase = FuelDatabase()
        
        assert fuelDatabase.readFuelTrain() == True
        
    def test_main_two(self):
        
        print("---------------- test_main_two  ----------------")

        logging.basicConfig(level=logging.INFO)
        
        logging.info("Read Fuel Train")

        fuelDatabase = FuelDatabase()
        
        assert fuelDatabase.readFuelTrain() == True
        assert fuelDatabase.checkFuelTrainHeaders() == True

        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(pd. __version__)
    
    unittest.main()