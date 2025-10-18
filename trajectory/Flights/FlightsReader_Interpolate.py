'''
Created on 17 oct. 2025

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

def addTimeDiffSeconds(self , df):
    df['time_diff_seconds'] = (df['end'] - df['start']).dt.total_seconds()
    return df

def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta


def plot ( timeSeries, valuesToPlot , columnName ):
    pass
    plt.figure(figsize=(8, 5))
    plt.plot(timeSeries, valuesToPlot, label=columnName , color="blue", linewidth=2)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(columnName)
    plt.show()
    
#============================================
class Test_Main(unittest.TestCase):
    
    def test_main_smooth(self):
        
        fileName = "prc806725776.parquet"
        flightsDatabase = FlightsDatabase()
        df_flight = flightsDatabase.readOneRankFile(fileName)
        
        print( df_flight.shape )
        print( list ( df_flight ) )
        
        # Example time and data
        min_time_value = df_flight['timestamp'].min()
        max_time_value = df_flight['timestamp'].max()
        
        df_flight['start'] = df_flight['timestamp'].min()
        df_flight['end'] = df_flight['timestamp'].max()
        print("flight shape = " , df_flight.shape)
        
        #time_intervals = pd.interval_range(start=min_time_value, end=max_time_value , freq=3 , )
        dt_time_intervals = [ dt for dt in datetime_range(min_time_value, max_time_value, timedelta(minutes=1))]
        df_time_intervals = pd.DataFrame( dt_time_intervals , columns=['timestamp'])
        print ("time intervals shape = " ,df_time_intervals.shape)
        #print(tabulate(df_time_intervals[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        
        df_flight['time_diff_seconds'] = (df_flight['timestamp'] - df_flight['start']).dt.total_seconds()
        #print(tabulate(df_flight[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        
        #df_merge = pd.merge ( df_time_intervals , df_flight,  on='timestamp', how='left')
        ''' outer means keep rows from both dataframes '''
        df = pd.merge_ordered ( df_flight , df_time_intervals , on='timestamp' , how='outer')
        print("merge shape = " ,df.shape)

        df['start'] = df['timestamp'].min()
        df['end'] = df['timestamp'].max()
        df['time_diff_seconds'] = (df['timestamp'] - df['start']).dt.total_seconds()

        #print(tabulate(df_merge[:100], headers='keys', tablefmt='grid' , showindex=False , ))
        
        for columnName in ['flight_id', 'aircraft_type_code','source']:
            ''' first non Nan value ni column '''
            df[columnName] = df.loc[df[columnName].first_valid_index(), columnName]
            
        # Interpolate the DataFrame
        interpolatedColumnList = ['latitude', 'longitude','altitude','groundspeed','track','vertical_rate', 'mach', 'TAS', 'CAS']
        df[interpolatedColumnList] = df[interpolatedColumnList].interpolate(method='linear',axis=0, Direction='both')
        
        print("count of nulls in vertical rate = " , str ( df['vertical_rate'].isnull().count() ))
        if df.shape[0] == df['vertical_rate'].isnull().count():
            print("--> the column vertical rate contains only nulls !!! ")
            ''' altitude are given in feet from PRC web site -> altitude: altitude [ft] '''
            #df.apply(lambda row: print ( str(df['timestamp'].iloc[row.name]) , str(row.name) , str(row.name-1) ) 
            #         if ( row.name-1 > 0 and row.name +1< len(df) ) else None , axis=1)
            ''' vertical rate -> feet per minutes '''
            df['vertical_rate'] = df.apply(lambda row: (df['altitude'].iloc[row.name] -  df['altitude'].iloc[row.name-1]) / (abs( df['time_diff_seconds'].iloc[row.name] - df['time_diff_seconds'].iloc[row.name-1]) / 60.0 ) 
                                           if ( (row.name > 0 )and (row.name < len(df))  and (df['time_diff_seconds'].iloc[row.name] - df['time_diff_seconds'].iloc[row.name-1]) > 0.0 ) else 0.0 , axis=1)
            ''' Drop rows where any value is outside the threshold'''
            
        print("show dataframe after extending empty vertical rates")
        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        print(tabulate(df[-10:], headers='keys', tablefmt='grid' , showindex=False , ))
        
        verticalRateMean = df['vertical_rate'].mean()
        verticalRateStd = df['vertical_rate'].std()
        maxVerticalRateFeetMinutes = 2000.0
        ''' suppress vertical rates outside 3 standard deviation '''
        df['vertical_rate'] = df['vertical_rate'].mask( ( df['vertical_rate'] < verticalRateMean - 3*verticalRateStd ) | ( df['vertical_rate'] > verticalRateMean + 3*maxVerticalRateFeetMinutes  ) )
        print ( df.isnull().sum() )
        #print("shape before dropping outliers on vertical rate = " , str(df.shape))
        #df = df[~((df['vertical_rate'] < verticalRateMean - maxVerticalRateFeetMinutes) | (df['vertical_rate'] > verticalRateMean + maxVerticalRateFeetMinutes)).any(axis=1)]
        #print("shape after dropping outliers on vertical rate = " , str(df.shape))
        
        

        df = df.fillna(0.0)
        print(tabulate(df.describe().transpose(), headers='keys', tablefmt='grid' , showindex=True ,))
        
        
        timeSeries = df['time_diff_seconds']
        for columnName in ['latitude', 'longitude','altitude','groundspeed','track','vertical_rate', 'mach', 'TAS', 'CAS']:
            seriesToPlot = df[columnName]
            plot( timeSeries , seriesToPlot , columnName)

        ''' drop added columns '''
        df = dropUnusedColumns( df , ['start','end','time_diff_seconds'] ) 
        print ( list ( df ))
        
        #Mais la donnée « aircraft track angle » sera plus facile à interpréter pour le modèle si vous convertissez les colonnes de direction et de vitesse de l’avion 
        #en un vecteur vitesse 
        ''' convert track angles from degrees to radians '''
        track_radians = df.pop('track')* np.pi / 180
        
        groundSpeedMax = df['groundspeed'].max()
        groundSpeed = df.pop('groundspeed')
        
        # Calculate the aircraft speed x and y components.
        df['groundspeed_x'] = groundSpeed * np.cos(track_radians)
        df['groundspeed_y'] = groundSpeed * np.sin(track_radians)
        
        # Calculate the max aircraft speed x and y components.
        df['max groundspeed_x'] = groundSpeedMax * np.cos(track_radians)
        df['max groundspeed_y'] = groundSpeedMax * np.sin(track_radians)
        
        plt.hist2d(df['groundspeed_x'], df['groundspeed_y'], bins=(50, 50), vmax=400)
        plt.colorbar()
        plt.xlabel('Aircraft ground speed X [m/s]')
        plt.ylabel('Aircraft ground speed Y [m/s]')
        ax = plt.gca()
        ax.axis('tight')
        plt.show()
        
        # Apply smoothing spline
        #smoothing_factor = 3  # Adjust this to control the smoothness
        #splineAltitudes = UnivariateSpline(time, altitudes, s=smoothing_factor)
        #splineVerticalRates = UnivariateSpline(time, verticalRates, s=smoothing_factor)
        # Generate smoothed data
        #smoothedAltitudes = splineAltitudes(time)
        #smoothedVerticalRates = splineVerticalRates(time)
        
        #df['mach'] =  smoothed_data
        #print(tabulate(smoothed_data[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        
        # Plot original and smoothed data
       
        
        
        #Key Parameters
        #s (smoothing factor): Controls the trade-off between closeness to the data and smoothness of the curve.
        # A smaller value fits the data more closely, while a larger value smooths more aggressively.
        #time: Ensure your time intervals are evenly spaced for better results. If not, consider resampling or interpolating first.
        #Alternative: Savitzky-Golay Filter
        #For simpler smoothing, you can use scipy.signal.savgol_filter:

        #Python
        
        #Copier le code
        #from scipy.signal import savgol_filter
        
        #smoothed_data = savgol_filter(data, window_length=11, polyorder=2)
        #Both methods are effective, but the choice depends on your specific data and smoothing needs.
        



    
    def test_main_one(self):
        logging.basicConfig(level=logging.INFO)

        print("---------------- plot  ----------------")
        
        fileName = "prc806725776.parquet"
        flightsDatabase = FlightsDatabase()
        df = flightsDatabase.readOneRankFile(fileName)
        
        print( df.shape )
        print( list ( df ) )
        
        ''' check if there are null values '''
        
        print( str ( df.isnull().sum() ))
        
        #print ( df.describe().transpose())
        print(tabulate(df.describe().transpose()[:10], headers='keys', tablefmt='grid' , showindex=True , ))

        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        
        dates = df['timestamp']
        altitudes = df['altitude']
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(dates, altitudes, marker='o', linestyle='-', color='b', label='Altitude')
        
        # Format the x-axis to show readable dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Adjust interval as needed
        plt.gcf().autofmt_xdate()  # Rotate date labels for better readability
        
        # Add labels and title
        plt.xlabel('Timestamp')
        plt.ylabel('Altitude (m)')
        plt.title('Altitude vs. Timestamp')
        plt.legend()
        plt.grid(True)
        
        # Show the plot
        plt.show()
        


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(pd. __version__)
    
    unittest.main()