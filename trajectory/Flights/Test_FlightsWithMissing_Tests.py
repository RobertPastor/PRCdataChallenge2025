'''
Created on 17 nov. 2025

@author: robert
'''
import logging
import unittest
import pandas as pd
import os
from trajectory.Flights.FlightsReader import FlightsDatabase
from tabulate import tabulate

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.interpolate import UnivariateSpline
from datetime import datetime, timedelta

from trajectory.utils import dropUnusedColumns

listOfErroneousFlightIds = ["prc770822360","prc770885136","prc770887555",
                            "prc770893597","prc772539375","prc776853928","prc777326263","prc784305329"]

def addTimeDiffSeconds(self , df):
    df['time_diff_seconds'] = (df['end'] - df['start']).dt.total_seconds()
    return df

def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta

def plotFlightFeatureVersusTime ( timeSeries, valuesToPlot , columnName , flight_id ):
    pass
    plt.figure(figsize=(8, 5))
    plt.plot(timeSeries, valuesToPlot, label=columnName , color="blue", linewidth=2)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(columnName + " - " + flight_id)
    plt.show()
    
#============================================
class Test_Main(unittest.TestCase):
    
    def test_main_one(self):
        pass
    
        for flight_id in listOfErroneousFlightIds:
            print(" ========================== ")
            print ( flight_id )
            
            flightsDatabase = FlightsDatabase()
            df = flightsDatabase.readOneTrainFileLite(flight_id)
    
            for columnName in ["latitude","longitude",'altitude' ]:
                print ( columnName )
                
                timeSeries = df['timestamp']                
                seriesToPlot = df[columnName]
                
                plotFlightFeatureVersusTime( timeSeries , seriesToPlot , columnName , flight_id)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(pd. __version__)
    
    unittest.main()
