'''
Created on 27 oct. 2025

@author: robert

'''


import matplotlib.pyplot as plt

import pandas as pd
import time
import os
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.ops import inplace_ops
# Set the option to display all columns
pd.options.display.max_columns = None

import numpy as np 
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

from tabulate import tabulate
from trajectory.utils import dropUnusedColumns , oneHotEncoderSklearn , getCurrentDateTimeAsStr

''' warning - use tensor flow 2.12.0 not the latest 2.20.0 that is causing DLL problems '''
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from trajectory.Fuel.FuelReader import FuelDatabase

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope

import logging
import unittest

from pathlib import Path
from tabulate import tabulate

import pandas as pd
import numpy as np
from scipy import stats

#============================================
class Test_Main(unittest.TestCase):

    def test_a_Train(self):
        
        extendedFuelDataFileName = "ExtendedFuel_train_2025-10-25-16-29-19.parquet"
        extendedFuelDataFileName = "ExtendedFuel_train_2025-10-26-10-44-58.parquet"
        extendedFuelDataFileName = "ExtendedFuel_train_2025-10-27-18-08-12.parquet"
        extendedFuelDataFileName = "ExtendedFuel_train_2025-10-31-12-44-23.parquet"
        
        filesFolder = "C:/Users/rober/eclipse-2025-09/eclipse-jee-2025-09-R-win32-x86_64/Data-Challenge-2025/documents"
        
        filePath = os.path.join( filesFolder , extendedFuelDataFileName)
        file = Path(filePath )
        
        directory = Path(filesFolder)
        if directory.is_dir() and file.is_file():
            
            start_time = time.time()
            
            df = pd.read_parquet ( filePath )
            print( df.shape )
            print ( list (df))
            
            for columnName in ["idx","Tail_Height_at_OEW_ft","Wheelbase_ft","Cockpit_to_Main_Gear_ft","Main_Gear_Width_ft"]:
                df.drop( columnName , axis = 1 , inplace = True)
            for columnName in ["Num_Engines","Approach_Speed_knot","Wingspan_ft_without_winglets_sharklets","Length_ft"]:
                df.drop( columnName , axis = 1 , inplace = True)
            for columnName in ["MTOW_kg","MALW_kg","Parking_Area_ft2","flight_date_year","flight_date_month"]:
                df.drop( columnName , axis = 1 , inplace = True)
            for columnName in ["flight_date_day_of_the_year"]:
                df.drop( columnName , axis = 1 , inplace = True)
                
            # 'idx', 'flight_id', 'start', 'end', 'time_diff_seconds', 'fuel_flow_kg_sec', 'aircraft_latitude_deg_at_fuel_start'
            
            
            print( df.shape )

            print(tabulate(df.describe().transpose()[-32:], headers='keys', tablefmt='grid' , showindex=True , ))
            
            


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    
    print("tensorflow version = " + tf.__version__)
    print("pandas version = " + pd. __version__)
    
    unittest.main()