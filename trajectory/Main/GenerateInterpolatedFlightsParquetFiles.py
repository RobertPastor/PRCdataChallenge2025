'''
Created on 19 nov. 2025

@author: robert

'''
import platform
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import logging
import unittest
from tabulate import tabulate

from trajectory.FlightList.FlightListReader import FlightListDatabase
from trajectory.Flights.FlightsReader import FlightsDatabase

from trajectory.FlightList.InterpolateGenerateParquets import generateInterpolatedFlightDataParquetFiles

#============================================
class Test_Main(unittest.TestCase):
    
    def test_main_train(self):
        
        print("---------------- Read Train Rank Final flight list  ----------------")
        
        logging.basicConfig(level=logging.DEBUG)
        train_rank_final = "train"
        #train_rank_final = "rank"
        #train_rank_final = "final"

        logging.info("---------Read Flight List <<" + train_rank_final +">> ------------")
        #generateInterpolatedFlightDataParquetFiles(train_rank_final)
        
    def test_main_rank(self):
        
        print("---------------- Read Train Rank Final flight list  ----------------")
        
        logging.basicConfig(level=logging.DEBUG)
        train_rank_final = "rank"

        logging.info("---------Read Flight List <<" + train_rank_final +">> ------------")
        generateInterpolatedFlightDataParquetFiles(train_rank_final)

    def test_main_final(self):
        
        print("---------------- Read Train Rank Final flight list  ----------------")
        
        logging.basicConfig(level=logging.DEBUG)
        train_rank_final = "final"

        logging.info("---------Read Flight List <<" + train_rank_final +">> ------------")
        generateInterpolatedFlightDataParquetFiles(train_rank_final)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    print("python version = " + platform.python_version())
    print("tensorflow version = " + tf.__version__)
    print("pandas version = " + pd. __version__)
    print("numpy version = " + np. __version__)
    unittest.main()