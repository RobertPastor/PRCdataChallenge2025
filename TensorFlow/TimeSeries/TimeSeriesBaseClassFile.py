'''
Created on 27 déc. 2025

@author: rober
'''


import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

from tabulate import tabulate
import logging

from trajectory.FlightList.FlightListReader import FlightListDatabase
from trajectory.Flights.FlightsReader import FlightsDatabase

class FlightTimeSeriesBaseClass(object):
    '''
    classdocs
    '''
    extendedFinalFuelDataFileName = ""
    
    def __init__(self , aircraft_type_code):
        
        self.class_name = self.__class__.__name__
        logging.basicConfig(level=logging.INFO)
        logging.info(self.class_name + " --- constructor ---")
        
        ''' use train dataset to beneficiate from fuel consumption data '''
        self.aircraft_type_code = aircraft_type_code
        
        train_rank_final = "train"
        flightList = FlightListDatabase(train_rank_final)
        assert flightList.readTrainFlightListLite()
        
        self.trainFlightListDataFrame = flightList.getTrainFlightListDataframe(train_rank_final)
        df = self.trainFlightListDataFrame
        ''' filter on unique aircraft '''
        df = df[df['aircraft_type']==aircraft_type_code]
        print(tabulate(df[:3], headers='keys', tablefmt='grid' , showindex=False , ))
        
    def identifyMostFlownRoutes(self):
        logging.info("---- identify Most Flown Routes  ---")
        df = self.trainFlightListDataFrame

        origin_airport_list = df['origin_icao'].unique().tolist()
        logging.info (self.class_name + " - " + str( origin_airport_list ) )
        
        results = {}
        count = 0
        for origin_airport_code in origin_airport_list:
            df_origin_airports = df[df['origin_icao']==origin_airport_code]
            #print(tabulate(df_origin_airports[:10], headers='keys', tablefmt='grid' , showindex=False , ))
            ''' find top 10 routes with most availables flight data / trajectories '''
            
            for index, row in df_origin_airports.iterrows():
                #print(row)
                #print(f"Index: {index}, origin: {row['origin_icao']}, Age: {row['destination_icao']}")
                origin_icao = row['origin_icao']
                destination_icao = row['destination_icao']
                route = origin_icao + "-" + destination_icao
                #print(route)
                if ( route in results):
                    count = results[route]["count"]
                    results[route] = {"route": route , "count":count+1}
                else:
                    results[route] = {"route": route , "count":1}

        self.top_most_flown_route = ""
        self.top_most_flown_routes_max = 0
        for result in results:
            #print(result + " - " + str( results[result]) )
            if results[result]["count"] > self.top_most_flown_routes_max:
                self.top_most_flown_route = result
                self.top_most_flown_routes_max = results[result]["count"]
            
        logging.info (self.class_name + " - most flown route -> {0} - max = {1}".format(self.top_most_flown_route , self.top_most_flown_routes_max))
        for result in results:
            if ( result == self.top_most_flown_route):
                logging.info (self.class_name + " - " + result + " - " + str( results[result]) )
                origin_airport = str(result).split("-")[0]
                logging.info (self.class_name + " - most flown origin airport = " + origin_airport )
                
    def concat_flights(self):
        logging.info("---- concat flights ---")
        logging.info (self.class_name + " - " + str( self.top_most_flown_route ) )

        df = self.trainFlightListDataFrame
        origin_airport_code = str(self.top_most_flown_route).split("-")[0]

        df = df[df['origin_icao']==origin_airport_code]
        df = df[df['aircraft_type']==self.aircraft_type_code]

        #print(tabulate(df[:3], headers='keys', tablefmt='grid' , showindex=False , ))
        
        flight_ids_list = df['flight_id'].unique().tolist()
        logging.info (self.class_name + " - length of list of flight ids = " + str( len ( flight_ids_list ) ) )
        
        flightsDatabase = FlightsDatabase()

        index = 0
        for flight_id in flight_ids_list:
            index = index + 1
            fileName = flight_id + ".parquet"
            df_flight = flightsDatabase.readOneTrainFileLite(fileName)
            
            #print(tabulate(df_flight[:3], headers='keys', tablefmt='grid' , showindex=False , ))
            print(tabulate(df_flight[:3], headers='keys', tablefmt='grid' , showindex=False , ))
            logging.info (self.class_name + " - index = {0} - train flight id = {1}".format( index , str( flight_id ) ) )

