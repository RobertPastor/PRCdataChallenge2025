'''
Created on 27 déc. 2025

@author: rober
'''
from TensorFlow.TimeSeries.TimeSeriesBaseClassFile import FlightTimeSeriesBaseClass
        
import logging
import unittest
import pandas as pd
from trajectory.FlightList.FlightListReader import FlightListDatabase
from tabulate import tabulate
        
class FlightTimeSeriesClass(FlightTimeSeriesBaseClass):
    
    def __init__(self , aircraft_icao_code ):
        
        self.class_name = self.__class__.__name__
        self.aircraft_icao_code = aircraft_icao_code
        
        logging.basicConfig(level=logging.INFO)
        logging.info(self.class_name + " --- constructor ---")

        super(FlightTimeSeriesClass, self).__init__( aircraft_icao_code )