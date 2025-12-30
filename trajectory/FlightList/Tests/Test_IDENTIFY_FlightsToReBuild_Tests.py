'''
Created on 21 nov. 2025

@author: robert
'''

import platform
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import logging
from tabulate import tabulate

from trajectory.FlightList.FlightListReader import FlightListDatabase
from trajectory.Flights.FlightsReader import FlightsDatabase
import matplotlib.pyplot as plt

class FlightsToReBuild(object):
    
    flightIdsWithNaNList = []
    flightIdsList = []
    columnNameList = ['latitude', 'longitude','altitude','groundspeed','track','vertical_rate', 'mach', 'TAS', 'CAS']
    
    def __init__(self , train_rank_final  ):
        self.flightIdsList = []
        self.train_rank_final = train_rank_final
        assert self.train_rank_final == "train" or self.train_rank_final == "rank" or self.train_rank_final == "final"
        
        logging.info("---------Read Flight List <<" + train_rank_final +">> ------------")
        self.flightIdsWithNaNList = []
        self.flightsData = FlightsDatabase()
        
    def readFlightListDatabase(self):

        self.flightListDatabase = FlightListDatabase(self.train_rank_final)
        if self.flightListDatabase.readTrainRankFinalFlightListLite(self.train_rank_final):
            logging.info("train rank final flight list read correctly")
            
            self.flightListDataframe = self.flightListDatabase.getTrainRankFinalFlightListDataframe(self.train_rank_final)
            print(tabulate(self.flightListDataframe[:10], headers='keys', tablefmt='grid' , showindex=False , ))

    def getFlightListIds(self):
        flightIdsList = []
        for index, row in self.flightListDataframe.iterrows():
            flightIdsList.append(row['flight_id'])
        return flightIdsList

    def loopThroughFlightDataList(self):
        count = 0
        ''' read the flight_id '''
        for index, row in self.flightListDataframe.iterrows():
            
            print(f"----- Index: {index} , Name: { row['flight_id'] } ----- ")
            flight_id = row['flight_id']
            print(flight_id)
            yield  flight_id                               

    def getFlightDataframe(self , flight_id):
        df_flightsDataframe  =  self.flightsData.readOneFlightFileLite(self.train_rank_final , flight_id )
        if ( df_flightsDataframe.empty == False ):
            return df_flightsDataframe
        return None

    def detectNaNInOneFlightDataFile( self , flight_id ):
        df_flightsDataframe  =  self.flightsData.readOneFlightFileLite(train_rank_final , flight_id )
        if ( df_flightsDataframe.empty == False ):
            
            rowCount = df_flightsDataframe.shape[0]
            ''' threshold to zero it is looking after any flight not usable for interpolation '''
            threshold = int ( 0 ) 
            #columnNameList = ['latitude', 'longitude','altitude','groundspeed','track','vertical_rate', 'mach', 'TAS', 'CAS']
            print(''' ============ count nan =================''')
            for column in self.columnNameList:
                nan_count_col1 = df_flightsDataframe[column].isna().sum()
                #print(f"NaN values in {column}: {nan_count_col1}")
                
            if ( df_flightsDataframe["latitude"].isna().sum() > threshold ) or \
                ( df_flightsDataframe["longitude"].isna().sum() > threshold ) or \
                ( df_flightsDataframe["altitude"].isna().sum() > threshold ) or \
                ( df_flightsDataframe["mach"].isna().sum() > threshold ): 
                self.flightIdsWithNaNList.append(flight_id)
                print("-"*90)
                return flight_id
        return None
    
    def getFlightIdsWithNaNAsList(self):
        return self.flightIdsWithNaNList

    def loopThroughFlightsWithNan(self):
        for flight_id in self.flightIdsList:
            pass
        
    def plotFlightFeatureVersusTime ( self, timeSeries, valuesToPlot , featureName , flight_id):
        
        plt.figure(figsize=(10, 6))
        plt.plot(timeSeries, valuesToPlot, label=featureName , marker='o', color="blue", linewidth=2)
        plt.legend()
        plt.xlabel("Timestamp")
        plt.ylabel(featureName)
        
        plt.title(featureName + "_vs timestamp" + "_for_" + flight_id)
        plt.grid(True)
        plt.show()
        # Format the x-axis to show readable dates
        #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        #plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Adjust interval as needed
        #plt.gcf().autofmt_xdate()  # Rotate date labels for better readability
        
    def plotFeature(self, x , y , feature , flight_id ):
        y_limit = max(y)
        plt.plot(x, label='timestamps')
        plt.plot(y, label='validation_loss')
        plt.title(feature)
        plt.ylim([0,y_limit])
        plt.xlabel('timestamp')
        plt.ylabel(feature)
        plt.legend()
        #plt.grid(True)
        
    def savePlotFile(self , feature , plt):
        # Save the plotFlightFeatureVersusTime to a file
        plotFileName = feature + '_' + flight_id + '.png'
        filesFolder = os.path.dirname(__file__)
        plotFilePath = os.path.join(filesFolder , plotFileName)
        plt.savefig(plotFilePath)  # Save as PNG
        # Close the plotFlightFeatureVersusTime to free memory
        plt.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    print("python version = " + platform.python_version())
    print("tensorflow version = " + tf.__version__)
    print("pandas version = " + pd. __version__)
    print("numpy version = " + np. __version__)
        
    logging.basicConfig(level=logging.DEBUG)
    train_rank_final = "rank"
    
    flightsToReBuild = FlightsToReBuild( train_rank_final )
    flightsToReBuild.readFlightListDatabase()
    
    flightIdsWithNan = []
    
    for flight_id in flightsToReBuild.getFlightListIds():
        #logging.info(flight_id)
        oneflightIdWithNan = flightsToReBuild.detectNaNInOneFlightDataFile(flight_id)
        if oneflightIdWithNan:
            flightIdsWithNan.append(oneflightIdWithNan)
            logging.info(oneflightIdWithNan)

        print("-"*90)

    print("-"*90)
    for flight_id in flightIdsWithNan:
        print(flight_id)
        
        #df = flightsToReBuild.getFlightDataframe(flight_id)
        #for feature in flightsToReBuild.columnNameList:
        #    print ( train_rank_final + " ; " + flight_id + "  ; " + feature )
        #    x = df['timestamp']
        #    y = df[feature]
        #    y_limit = df[feature].max()
            #flightsToReBuild.plotFlightFeatureVersusTime(x, y , feature, flight_id)
        
    print("-"*90)

        