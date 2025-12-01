'''
Created on 1 déc. 2025

@author: robert
'''

import logging
import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tabulate import tabulate

import logging
from trajectory.Utils.badaAirspeeds import tas2mach, tas2cas, mach2tas, mach2cas


#============================================
class Test_Main(unittest.TestCase):
    
    def test_tas2mach(self):
        logging.basicConfig(level=logging.INFO)

        tasKnots = 350.0 
        altitude_feet = 29000.0
        altitude_feet = 29000.0
        deltaTemperatureCelsius = 0.0
        
        mach = tas2mach( tasKnots = tasKnots, altitude_feet =  altitude_feet ,  deltaTemperatureCelsius = deltaTemperatureCelsius ) 
        logging.info(f"Input TAS {tasKnots} knots - at {altitude_feet} feet -> mach = {mach}")
        
    def test_tas2cas(self):
        logging.basicConfig(level=logging.INFO)

        tasKnots = 350.0 
        altitude_feet = 29000.0
        altitude_feet = 29000.0
        deltaTemperatureCelsius = 0.0
        
        cas = tas2cas ( tasKnots = tasKnots, altitude_feet =  altitude_feet , deltaTemperatureCelsius = deltaTemperatureCelsius  )
        logging.info(f"Input TAS {tasKnots} knots - at {altitude_feet} feet -> CAS = {cas} knots ")
    
    def test_mach2tas(self):
        logging.basicConfig(level=logging.INFO)

        tasKnots = 350.0 
        altitude_feet = 29000.0
        altitude_feet = 29000.0
        deltaTemperatureCelsius = 0.0

        mach = 0.85
        tas = mach2tas ( mach = mach , altitude_feet = altitude_feet, deltaTemperatureCelsius = deltaTemperatureCelsius )
        logging.info(f"Input mach {mach} - at {altitude_feet} feet -> TAS = {tas} knots ")
            
            
    def test_mach2cas(self):
        logging.basicConfig(level=logging.INFO)

        tasKnots = 350.0 
        altitude_feet = 29000.0
        altitude_feet = 29000.0
        deltaTemperatureCelsius = 0.0

        mach = 0.85
        cas = mach2cas ( mach = mach , altitude_feet = altitude_feet, deltaTemperatureCelsius = deltaTemperatureCelsius )
        logging.info(f"Input mach {mach} - at {altitude_feet} feet -> CAS = {cas} knots ")
            
            
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print(pd. __version__)
    unittest.main()


