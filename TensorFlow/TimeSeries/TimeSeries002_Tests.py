'''
Created on 27 déc. 2025

@author: robert
'''
import unittest

import os
import datetime
import logging
import platform


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

from TensorFlow.TimeSeries.TimeSeriesClassFile import FlightTimeSeriesClass

class Test(unittest.TestCase):

    def test_first_test(self):
        print ("--- test one ---")
        extendedFuelTrainDataFileName = "ExtendedFuel_train_2025-12-27-13-58-14.parquet"
        javaTrainRankfilesFolder = "C:/Users/rober/eclipse-2025-09/eclipse-jee-2025-09-R-win32-x86_64/Data-Challenge-2025/documents/"

        aircraft_icao_code = "A320"
        flightTimeSeriesClass = FlightTimeSeriesClass(aircraft_icao_code)
        flightTimeSeriesClass.identifyMostFlownRoutes()
        flightTimeSeriesClass.concat_flights()
            
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    
    logging.basicConfig(level=logging.INFO)
    
    print("python version = " + platform.python_version())
    print("tensorflow version = " + tf.__version__)
    print("pandas version = " + pd. __version__)
    print("numpy version = " + np. __version__)
    
    
    unittest.main()