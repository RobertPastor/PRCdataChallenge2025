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

import pytz

from openap.phase import FlightPhase

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

from tabulate import tabulate
import logging

from trajectory.FlightList.FlightListReader import FlightListDatabase
from trajectory.Flights.FlightsReader import FlightsDatabase
from trajectory.Fuel.FuelReader import FuelDatabase

from trajectory.Utils.utils import keepOnlyColumns

class FlightTimeSeriesBaseClass(object):
    '''
    classdocs
    '''
    extendedFinalFuelDataFileName = ""
    
    def __init__(self , aircraft_type_code):
        
        self.class_name = self.__class__.__name__
        logging.basicConfig(level=logging.INFO)
        logging.info(self.class_name + " --- constructor ---")
        
        self.aircraft_type_code = aircraft_type_code
        
        self.flightsDatabase = FlightsDatabase()
        
        self.fuelTrainDatabase = FuelDatabase(None)
        assert self.fuelTrainDatabase.readFuelTrain() == True
        
        ''' use train dataset to beneficiate from fuel consumption data '''
        train_rank_final = "train"
        
        flightList = FlightListDatabase(train_rank_final)
        assert flightList.readTrainFlightListLite()
        
        self.trainFlightListDataFrame = flightList.getTrainFlightListDataframe(train_rank_final)
        ''' convenience rename '''
        df  = self.trainFlightListDataFrame
        
        ''' filter on one aircraft '''
        df = df[df['aircraft_type'] == aircraft_type_code]
        print(tabulate(df[:3], headers='keys', tablefmt='grid' , showindex=False , ))
        
    def create_plot(self , df ):
        
        print ( list ( df ))
        print ( df.info())
        plot_cols = ['altitude', 'groundspeed', 'vertical_rate','fuel_flow_kg_sec']
        plot_features = df[plot_cols]
        plot_features.index = df['timestamp']
        _ = plot_features.plot(subplots=True)
        
        plt.show()
        
    def computeMostFlownRoutes(self):
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
            
        print ( "--- most flown route - origin and destination airport ---")
        logging.info (self.class_name + " - most flown route -> {0} - max = {1}".format(self.top_most_flown_route , self.top_most_flown_routes_max))
        
        for result in results:
            if ( result == self.top_most_flown_route):
                logging.info (self.class_name + " - " + result + " - " + str( results[result]) )
                self.most_flown_origin_airport = str(result).split("-")[0]
                logging.info (self.class_name + " - most flown origin airport = " + self.most_flown_origin_airport )
                self.most_flown_destination_airport = str(result).split("-")[1]
                logging.info (self.class_name + " - most flown destination airport = " + self.most_flown_destination_airport )
        
    def getFlightListTakeOff(self):
        
        logging.info("---- most flown route flight Ids ---")
        df = self.trainFlightListDataFrame
        
        ''' filter on list of flight ids '''
        # Filter rows where the 'City' column matches any value in the list
        df = df[df['origin_icao']      == self.most_flown_origin_airport]
        df = df[df['destination_icao'] == self.most_flown_destination_airport]
        df = df[df['aircraft_type']    == self.aircraft_type_code]
        
        filtered_df = df[df['flight_id'].isin(self.flight_ids_list)]
        print(tabulate(filtered_df[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        print("dataframe shape = {0}".format(filtered_df.shape))

    def listMostFlownRoutesFlightIds(self):
        
        logging.info("---- most flown route flight Ids ---")
        logging.info (self.class_name + " - " + str( self.top_most_flown_route ) )

        df = self.trainFlightListDataFrame
        origin_airport_code = str(self.top_most_flown_route).split("-")[0]
        destination_airport_code = str(self.top_most_flown_route).split("-")[1]

        df = df[df['origin_icao']       == origin_airport_code]
        df = df[df['destination_icao']  == destination_airport_code]
        df = df[df['aircraft_type']     == self.aircraft_type_code]

        #print(tabulate(df[:3], headers='keys', tablefmt='grid' , showindex=False , ))
        
        self.flight_ids_list = df['flight_id'].unique().tolist()
        logging.info (self.class_name + " - length of list of flight ids = " + str( len ( self.flight_ids_list ) ) )
        logging.info (self.class_name + " - list of flight ids = " + str(  ( self.flight_ids_list ) ) )
        
        
    def computeFlight(self):
        
        logging.info("---- compute flight  ---")
        
        df_flightList = self.trainFlightListDataFrame
        origin_airport_code      = str(self.top_most_flown_route).split("-")[0]
        destination_airport_code = str(self.top_most_flown_route).split("-")[1]

        df_flightList = df_flightList[df_flightList['origin_icao']      == origin_airport_code]
        df_flightList = df_flightList[df_flightList['destination_icao'] == destination_airport_code]
        df_flightList = df_flightList[df_flightList['aircraft_type']    == self.aircraft_type_code]

        #print(tabulate(df[:3], headers='keys', tablefmt='grid' , showindex=False , ))
        self.flight_ids_list = df_flightList['flight_id'].unique().tolist()
        logging.info (self.class_name + " - length of list of flight ids = " + str( len ( self.flight_ids_list ) ) )
        
        flight_id = self.flight_ids_list[0]
        print ("first flight id = {0}".format(flight_id))
        
        ''' convert flight list takeoff in datetime '''
        df_flightList['takeOff_datetime'] = pd.to_datetime( df_flightList['takeoff'] )
        
        ''' Get the first value in the 'takeOff_datetime' column '''
        takeOffDateTime = df_flightList['takeOff_datetime'].iloc[0]

        ''' filter on one flight id '''
        df_flightList = df_flightList[df_flightList['flight_id'] == flight_id]
        print(''' should list only one an only one flight id ''')
        print(tabulate(df_flightList[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        
        ''' goal is to add takeoff column to the merge of flight and fuel for the same flight id '''
        print ( "---- one flight dataframe for one flight id ---")
        df_flight = self.flightsDatabase.readOneTrainFileLite(flight_id)
        
        df_flight['takeoff'] = takeOffDateTime
        print(tabulate(df_flight[:10], headers='keys', tablefmt='grid' , showindex=False , ))

        df_flight['time_difference'] = df_flight['timestamp'] - df_flight['takeoff']
        
        ''' relative reference for a flight '''
        df_flight['time_difference_seconds'] = df_flight['time_difference'].dt.total_seconds()
        ''' rename column '''
        df_flight.rename(columns={'time_difference_seconds': 'timestamp_deltaSeconds'}, inplace=True)
        
        list_of_columns_to_keep = ['flight_id','takeoff','altitude','groundspeed','vertical_rate','timestamp_deltaSeconds']
        df_flight = keepOnlyColumns (df_flight , list_of_columns_to_keep)
        
        df_flight.rename(columns={'timestamp_deltaSeconds': 'timestamp'}, inplace=True)

        print(tabulate(df_flight[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        print(tabulate(df_flight[-10:], headers='keys', tablefmt='grid' , showindex=False , ))
        
        self.df_flight = df_flight
        return flight_id
        
    def computeFuel(self, flight_id):

        print(''' ---- filter fuel on flight id  --- ''')
        ''' in order for the fuel start and end to exist as new rows in the flight dataframe '''     
        #assert self.fuelTrainDatabase.readFuelTrain()
        df_fuel = self.fuelTrainDatabase.getFuelTrainDataframe()
        #print ( str ( list ( df_fuel )))
        
        #list_of_columns_to_keep = ['flight_id','takeoff','fuel_burn_start','fuel_burn_end','fuel_kg','time_diff_seconds','fuel_flow_kg_sec','flight_distance_Nm','flight_duration_sec','fuel_burn_relative_start','fuel_burn_relative_end']
        list_of_columns_to_keep = ['flight_id','takeoff','fuel_flow_kg_sec','fuel_burn_relative_start','fuel_burn_relative_end']
        df_fuel = keepOnlyColumns (df_fuel , list_of_columns_to_keep)
        df_fuel = df_fuel[df_fuel['flight_id'] == flight_id]
        
        # Convert index to DatetimeIndex
        df_fuel.index = pd.to_datetime(df_fuel.takeoff)
        df_fuel['takeoff'] = df_fuel['takeoff'].dt.tz_localize(None)
        #df_fuel.index = df_fuel.index.tz_convert(None)
        # Convert to another UTC
        #df_fuel = df_fuel.tz_convert(None)
        
        print(''' rename column to timestamp ''')
        df_fuel.rename(columns={'fuel_burn_relative_end': 'timestamp'}, inplace=True)
        # Convert to another UTC
        print(tabulate(df_fuel[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        print(tabulate(df_fuel[-10:], headers='keys', tablefmt='grid' , showindex=False , ))
        
        self.df_fuel = df_fuel
        
    def concatFlightAndFuel(self):

        print ( ''' -------- concat flight and fuel dataframes --------------- ''')
        df_concat  = pd.concat ( [self.df_flight , self.df_fuel])
        
        # Trier par la colonne 'Âge'
        df_concat = df_concat.sort_values(by="timestamp")
        print(tabulate(df_concat[:10] , headers='keys', tablefmt='grid' , showindex=False , ))
        print(tabulate(df_concat[-10:], headers='keys', tablefmt='grid' , showindex=False , ))
        
        self.df_concat = df_concat

    def concat_dataframes(self):
        
        logging.info("---- concat flights ---")
        logging.info (self.class_name + " - " + str( self.top_most_flown_route ) )

        df_flightList = self.trainFlightListDataFrame
        origin_airport_code      = str(self.top_most_flown_route).split("-")[0]
        destination_airport_code = str(self.top_most_flown_route).split("-")[1]

        df_flightList = df_flightList[df_flightList['origin_icao']      == origin_airport_code]
        df_flightList = df_flightList[df_flightList['destination_icao'] == destination_airport_code]
        df_flightList = df_flightList[df_flightList['aircraft_type']    == self.aircraft_type_code]

        #print(tabulate(df[:3], headers='keys', tablefmt='grid' , showindex=False , ))
        flight_ids_list = df_flightList['flight_id'].unique().tolist()
        logging.info (self.class_name + " - length of list of flight ids = " + str( len ( flight_ids_list ) ) )
                
        ''' convert flight list takeoff in datetime '''
        df_flightList['takeOff_datetime'] = pd.to_datetime( df_flightList['takeoff'] )
        print(tabulate(df_flightList[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        
        ''' goal is to add takeoff column to the merge of flight and fuel for the same flight id '''
        flightsDatabase = FlightsDatabase()

        index = 0
        first = True
        max_timestamp = 0.0
        max_flights = 10
        for flight_id in flight_ids_list:
            print(" --------------------------- " + flight_id + " --------------------")
            index = index + 1
            if index > max_flights:
                break
            fileName = flight_id + ".parquet"
            
            ''' dataframe of one flight '''
            df_flight = flightsDatabase.readOneTrainFileLite(fileName)
            logging.info (self.class_name + " - index = {0} - train flight id = {1}".format( index , str( flight_id ) ) )
            
            ''' reference is the takeoff of the flight from the flight list '''
            
            # Convert 'timestamp' column to datetime
            #df['timestamp'] = pd.to_datetime(df['timestamp'])
            df_flight['time_difference'] = df_flight['timestamp'] - df_flight['timestamp'].min()
            
            df_flight['time_difference_seconds'] = df_flight['time_difference'].dt.total_seconds()
            
            df_flight = df_flight.drop(['timestamp', 'time_difference'], axis=1 )
            df_flight = df_flight.rename(columns={'time_difference_seconds':'timestamp'}).copy()
            
            # Sort by 'timestamp' column
            df_flight = df_flight.sort_values(by='timestamp')
            
            if first == True:
                df_concat = df_flight
                max_timestamp = df_flight['timestamp'].max()
                first = False
                
            else:
                df_flight['timestamp'] = df_flight['timestamp'] + max_timestamp
                df_concat = pd.concat([df_concat,df_flight])
                max_timestamp = df_flight['timestamp'].max()
            
            print ("---- concat dataframe length = {0}".format( df_concat.shape[0]))
            
        ''' clean '''
        df_concat.drop(df_concat[df_concat['groundspeed'] > 600.0].index, inplace=True)
        df_concat.drop(df_concat[df_concat['vertical_rate'] < -3000.0].index, inplace=True)
        
        self.df_concat = df_concat
        
    def plotMainFeatures(self):
    
        ''' plot '''
        self.create_plot( self.df_concat )
        
    def compute_flight_phases(self):
        logging.info("---- compute flight phases ---")
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
            print(" --------------------------- " + flight_id + " --------------------")
            index = index + 1
            fileName = flight_id + ".parquet"
            df = flightsDatabase.readOneTrainFileLite(fileName)
            logging.info (self.class_name + " - index = {0} - train flight id = {1}".format( index , str( flight_id ) ) )
            # Count total NaNs in the DataFrame
            total_nans = df.isna().sum().sum()
            #print("\nTotal NaNs in DataFrame: {0} - nb rows = {1}".format ( total_nans , df.shape[0]) )
            
            #print(df_flight.isna().sum())
            
            # Keep only 'timestamp','altitude', 'groundspeed','vertical_rate' columns
            df = df[['timestamp','altitude', 'groundspeed','vertical_rate']]
            #print (list ( df ))
            
            # Rename columns
            df = df.rename(columns={'timestamp':'ts','altitude': 'alt', 'groundspeed': 'spd','vertical_rate':'roc'}).copy()
            #print (list ( df ))
            
            # Convert 'timestamp' column to datetime
            df['ts'] = pd.to_datetime(df['ts'])
            df['time_difference'] = df['ts'] - df['ts'].min()

            df['time_difference_seconds'] = df['time_difference'].dt.total_seconds()
            
            df = df.drop(['ts', 'time_difference'], axis=1 )
            df = df.rename(columns={'time_difference_seconds':'ts'}).copy()

            # Sort by 'timestamp' column
            df = df.sort_values(by='ts')
            #print(tabulate(df[:3], headers='keys', tablefmt='grid' , showindex=False , ))

            #print ( df.dtypes )
            
            ts = df["ts"].values
            ts = ts - ts[0]
            alt = df["alt"].values
            spd = df["spd"].values
            roc = df["roc"].values

            ts_ = np.arange(0, ts[-1], 1)
            alt_ = np.interp(ts_, ts, alt)
            spd_ = np.interp(ts_, ts, spd)
            roc_ = np.interp(ts_, ts, roc)

            fp = FlightPhase()
            fp.set_trajectory(ts_, alt_, spd_, roc_)
            labels = fp.phaselabel()
            #print ( labels )
            #print ( set(labels))
            print (list(dict.fromkeys(labels)))

            #print(tabulate(df_flight[:3], headers='keys', tablefmt='grid' , showindex=False , ))
            #print(tabulate(df_flight[:3], headers='keys', tablefmt='grid' , showindex=False , ))

