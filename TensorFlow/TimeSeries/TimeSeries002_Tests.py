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
from tabulate import tabulate

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
        flightTimeSeriesClass.computeMostFlownRoutes()
        flightTimeSeriesClass.listMostFlownRoutesFlightIds()
        flightTimeSeriesClass.getFlightListTakeOff()
        
        flight_ids_list = flightTimeSeriesClass.computeFlightIdsList()
        print ( str ( flight_ids_list ))
        
        first = True
        df_concat_all = None
        count = 0
        for flight_id in flight_ids_list:
            count = count + 1
            if count > 10:
                break
            
            df_flight = flightTimeSeriesClass.computeFlight(flight_id)
            df_fuel   = flightTimeSeriesClass.computeFuel(flight_id)
            
            if first == True:
                first = False
                df_concat_all = flightTimeSeriesClass.concatFlightAndFuel(df_flight , df_fuel)
            else:
                df_concat = flightTimeSeriesClass.concatFlightAndFuel(df_flight , df_fuel)
                df_concat_all = pd.concat ( [df_concat_all , df_concat])
                
            print("-------------- " + str(flight_id) + " --------")
            print ( df_concat_all.shape )
            print("-------------- " + str(flight_id) + " --------")

        print(tabulate(df_concat_all[:10] , headers='keys', tablefmt='grid' , showindex=False , ))
        print(tabulate(df_concat_all[-10:], headers='keys', tablefmt='grid' , showindex=False , ))

        #flightTimeSeriesClass.plotMainFeatures()
        
        #flightTimeSeriesClass.compute_flight_phases()
        #flightTimeSeriesClass.concat_dataframes()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    
    logging.basicConfig(level=logging.INFO)
    
    print("python version = " + platform.python_version())
    print("tensorflow version = " + tf.__version__)
    print("pandas version = " + pd. __version__)
    print("numpy version = " + np. __version__)
    
    
    unittest.main()