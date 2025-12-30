'''
Created on 9 oct. 2025

@author: robert
'''

import logging
import os
from pathlib import Path
import pandas as pd
from tabulate import tabulate
from datetime import datetime, timedelta
from trajectory.Utils.utils import dropUnusedColumns

initialHeaders = ['timestamp', 'flight_id','typecode','latitude', 'longitude', 'altitude', 'groundspeed', 'track', 'vertical_rate', 'mach', 'TAS', 'CAS', 'source']
''' type_code renamed as aircraft_type_code '''
expectedHeaders = ['timestamp', 'flight_id', 'aircraft_type_code', 'latitude', 'longitude', 'altitude', 'groundspeed', 'track', 'vertical_rate', 'mach', 'TAS', 'CAS', 'source']

def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta
        
class FlightsDatabase(object):
    
    className = ''
    
    def __init__(self):
        self.className = self.__class__.__name__
        
        #self.filesFolder = "C:\\Users\\rober\\git\\PRCdataChallenge2025\\Data-Download-OpenSkyNetwork\\competition-train-data"
        self.filesFolder = os.path.dirname(__file__)
        self.filesFolderTrain = os.path.join( self.filesFolder , ".." , ".." , "Data-Download-OpenSkyNetwork" , "competition-train-data")
        self.filesFolderRank = os.path.join( self.filesFolder , ".." , ".." , "Data-Download-OpenSkyNetwork" , "competition-rank-data")
        self.filesFolderFinal = os.path.join( self.filesFolder , ".." , ".." , "Data-Download-OpenSkyNetwork" , "competition-final-data")
        
        self.filesFolderTrainInterpolated = os.path.join( self.filesFolder , ".." , ".." , "Data-Download-OpenSkyNetwork" , "competition-train-data-interpolated")
        self.filesFolderRankInterpolated = os.path.join( self.filesFolder , ".." , ".." , "Data-Download-OpenSkyNetwork" , "competition-rank-data-interpolated")
        self.filesFolderFinalInterpolated = os.path.join( self.filesFolder , ".." , ".." , "Data-Download-OpenSkyNetwork" , "competition-final-data-interpolated")
        
        self.filesFolderRankComputed = os.path.join( self.filesFolder , ".." , ".." , "Data-Download-OpenSkyNetwork" , "competition-rank-data-computed")
        self.filesFolderTrainComputed = os.path.join( self.filesFolder , ".." , ".." , "Data-Download-OpenSkyNetwork" , "competition-train-data-computed")
        self.filesFolderFinalComputed = os.path.join( self.filesFolder , ".." , ".." , "Data-Download-OpenSkyNetwork" , "competition-final-data-computed")

        assert Path(self.filesFolderTrain).is_dir() == True
        assert Path(self.filesFolderRank).is_dir() == True
        assert Path(self.filesFolderFinal).is_dir() == True
        
        assert Path(self.filesFolderTrainInterpolated).is_dir() == True
        assert Path(self.filesFolderRankInterpolated).is_dir() == True
        assert Path(self.filesFolderFinalInterpolated).is_dir() == True
        
    ''' assumption : there is at least one row with a non null value in the typecode column '''
    def getFirstNonNullValueInColumn(self , df, columnName):
        # Get the first valid non-null value from a specific column
        assert columnName in initialHeaders
        first_valid_index = df[columnName].first_valid_index()
        first_valid_value = df.loc[first_valid_index, columnName]

        print(f"First valid non-null value in column : {columnName} : {first_valid_value}")
        return first_valid_value
            
    def getTrainFlightsFolderPathStr(self):
        return self.filesFolderTrain
    
    def getRankFlightsFolderPathStr(self):
        return self.filesFolderRank
    
    def getTrainRankFinalFlightsComputedFolderPathStr(self, train_rank_final):
        if train_rank_final == "train":
            return self.filesFolderTrainComputed
        elif train_rank_final == "rank":
            return self.filesFolderRankComputed
        else:
            return self.filesFolderFinalComputed
        
    def getTrainRankFinalFlightsFolderPathStr(self, train_rank_final):
        if train_rank_final == "train":
            return self.filesFolderTrain
        elif train_rank_final == "rank":
            return self.filesFolderRank
        else:
            return self.filesFolderFinal
        
    def getFlightsInterpolatedFolderPathStr(self , train_rank_final):
        if train_rank_final == "train":
            return self.filesFolderTrainInterpolated
        elif train_rank_final == "rank":
            return self.filesFolderRankInterpolated
        else:
            return self.filesFolderFinalInterpolated

    def checkFlightsTrainHeaders(self):
        return (set(self.FlightsTrainDataframe) == set(expectedHeaders))
    
    def checkFlightsRankHeaders(self):
        return (set(self.FlightsRankDataframe) == set(expectedHeaders))
    
    def renameColumns(self, df):
        return df.rename(columns= {'typecode':'aircraft_type_code'})
    
    def getFlightId(self):
        return self.flightId
    
    def getMinTimeStamp(self , df_flight):
        return df_flight['timestamp'].min()
    
    def getMaxTimeStamp(self , df_flight):
        return df_flight['timestamp'].max()
    
    def interpolateTimeSeries(self , df_flight ):
        
        print( df_flight.shape )
        print( list ( df_flight ) )
        
        # Example time and data
        min_time_value = df_flight['timestamp'].min()
        max_time_value = df_flight['timestamp'].max()
        
        df_flight['start'] = df_flight['timestamp'].min()
        df_flight['end']   = df_flight['timestamp'].max()
        #print("flight shape = " , df_flight.shape)
        
        #time_intervals = pd.interval_range(start=min_time_value, end=max_time_value , freq=3 , )
        dt_time_intervals = [ dt for dt in datetime_range(min_time_value, max_time_value, timedelta(minutes=1))]
        df_time_intervals = pd.DataFrame( dt_time_intervals , columns=['timestamp'])
        #print ("time intervals shape = " ,df_time_intervals.shape)
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
            
        #print("show dataframe after extending empty vertical rates")
        #print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        #print(tabulate(df[-10:], headers='keys', tablefmt='grid' , showindex=False , ))
        
        verticalRateMean = df['vertical_rate'].mean()
        verticalRateStd = df['vertical_rate'].std()
        maxVerticalRateFeetMinutes = 2000.0
        ''' suppress vertical rates outside 3 standard deviation '''
        df['vertical_rate'] = df['vertical_rate'].mask( ( df['vertical_rate'] < verticalRateMean - (3*verticalRateStd)) | ( df['vertical_rate'] > (verticalRateMean + 3*verticalRateStd ) ) )
        #print ( df.isnull().sum() )
        #print("shape before dropping outliers on vertical rate = " , str(df.shape))
        #df = df[~((df['vertical_rate'] < verticalRateMean - maxVerticalRateFeetMinutes) | (df['vertical_rate'] > verticalRateMean + maxVerticalRateFeetMinutes)).any(axis=1)]
        #print("shape after dropping outliers on vertical rate = " , str(df.shape))
        
        df = df.fillna(0.0)
        #print(tabulate(df.describe().transpose(), headers='keys', tablefmt='grid' , showindex=True ,))
        
        ''' drop added columns '''
        df = dropUnusedColumns( df , ['start','end','time_diff_seconds'] ) 
        #print ( list ( df ))
        
        return df
        
    
    def readOneRankFile(self , fileName):
        
        if str(fileName).endswith("parquet") == False:
            fileName = fileName + ".parquet"
        
        #logging.info(self.className + ": file name = " + fileName)
        filePath = os.path.join( self.filesFolderRank , fileName)
        file = Path(filePath)
        
        assert file.is_file() == True
        
        self.FlightsRankDataframe = pd.read_parquet(filePath)
        assert list(self.FlightsRankDataframe) == initialHeaders
        
        self.flightId = self.FlightsRankDataframe['flight_id'].unique()
        
        ''' column typecode renamed as aircraft type code '''
        self.FlightsRankDataframe = self.renameColumns(self.FlightsRankDataframe)
        
        ''' convert datetime to UTC '''
        self.FlightsRankDataframe['timestamp'] = pd.to_datetime(self.FlightsRankDataframe['timestamp'], utc=True)
        
        assert self.checkFlightsRankHeaders()
        
        ''' correct erroneous values in source column '''
        self.FlightsRankDataframe['source'] = self.FlightsRankDataframe['source'].apply(lambda x: str('unknown_source') if not isinstance(x, str) else str(x)  )
        
        ''' one hot encode the source column '''
        ''' do not hot encode on a per file basis as some file may have only one value in the source '''
        ''' hot encoding is done after all 13.000 thousands Flight Data files are concated '''
        #self.FlightsTrainDataframe  = self.oneHotEncodeSource(self.FlightsTrainDataframe, "source")
        
        self.FlightsRankDataframe = self.interpolateTimeSeries( self.FlightsRankDataframe  )
        
        return self.FlightsRankDataframe
    
    ''' extend timestamp series to one minute interval '''
   
    ''' read flight parquet file and return a dataframe , either a train or a rank flight dataframe '''
    def readOneFlightFileLite(self , train_rank_final , flightfileName ):
        if str(flightfileName).endswith("parquet") == False:
            flightfileName = flightfileName + ".parquet"
        folderPathStr = ""
        if train_rank_final == "train":
            folderPathStr = self.filesFolderTrain 
        elif train_rank_final == "rank":
            folderPathStr = self.filesFolderRank 
        else:
            folderPathStr = self.filesFolderFinal 

        logging.info(self.className + ": file name = " + flightfileName)
        logging.info(self.className + ": file path = " + folderPathStr)
        flightFilePath = os.path.join( folderPathStr , flightfileName)
        print (flightFilePath)
        file = Path(flightFilePath)
        assert file.is_file() == True
        
        self.FlightsDataframe = pd.read_parquet(flightFilePath)
        return self.FlightsDataframe
    
    def readOneRankFileLite(self, fileName ):
        
        if str(fileName).endswith("parquet") == False:
            fileName = fileName + ".parquet"
        
        #logging.info(self.className + ": file name = " + fileName)
        filePath = os.path.join( self.filesFolderRank , fileName)
        file = Path(filePath)
        
        assert file.is_file() == True
        
        self.FlightsRankDataframe = pd.read_parquet(filePath)
        return self.FlightsRankDataframe
    
    def readOneTrainFileLite(self, fileName ):
        
        if str(fileName).endswith("parquet") == False:
            fileName = fileName + ".parquet"
        
        #logging.info(self.className + ": file name = " + fileName)
        filePath = os.path.join( self.filesFolderTrain , fileName)
        file = Path(filePath)
        
        assert file.is_file() == True
        
        self.FlightsTrainDataframe = pd.read_parquet(filePath)
        return self.FlightsTrainDataframe

    def readOneTrainFile(self, fileName):
        
        if str(fileName).endswith("parquet") == False:
            fileName = fileName + ".parquet"
        
        #logging.info(self.className + ": file name = " + fileName)
        filePath = os.path.join( self.filesFolderTrain , fileName)
        file = Path(filePath)
        
        assert file.is_file() == True
        
        self.FlightsTrainDataframe = pd.read_parquet(filePath)
        assert list(self.FlightsTrainDataframe) == initialHeaders
        
        self.flightId = self.FlightsTrainDataframe['flight_id'].unique()

        ''' column typecode renamed as aircraft type code '''
        self.FlightsTrainDataframe = self.renameColumns(self.FlightsTrainDataframe)
        
        ''' convert datetime to UTC '''
        self.FlightsTrainDataframe['timestamp'] = pd.to_datetime(self.FlightsTrainDataframe['timestamp'], utc=True)
        
        assert self.checkFlightsTrainHeaders()
        
        ''' correct erroneous values in source column '''
        self.FlightsTrainDataframe['source'] = self.FlightsTrainDataframe['source'].apply(lambda x: str('unknown_source') if not isinstance(x, str) else str(x)  )
        
        ''' one hot encode the source column '''
        ''' do not hot encode on a per file basis as some file may have only one value in the source '''
        ''' hot encoding is done after all 13.000 thousands Flight Data files are concated '''
        #self.FlightsTrainDataframe  = self.oneHotEncodeSource(self.FlightsTrainDataframe, "source")
        
        self.FlightsTrainDataframe = self.interpolateTimeSeries( self.FlightsTrainDataframe  )
        
        return self.FlightsTrainDataframe
        
    def readSomeTrainFiles(self, testMode = False):
        file_count = 0
        if testMode == True:
            file_count = 0
            
        directory = Path(self.filesFolderTrain)
        if directory.is_dir():
            
            for fileName in os.listdir(directory):
                #logging.info(self.className + ": file name = " + fileName)
                
                self.filePath = os.path.join(self.filesFolderTrain , fileName)
                
                file = Path(self.filePath)
                if file.is_file() and fileName.endswith("parquet"):
                    
                    if (testMode == True) and (file_count < 10):
                        
                        self.FlightsTrainDataframe = pd.read_parquet(self.filePath)
                        self.FlightsTrainDataframe = self.renameColumns(self.FlightsTrainDataframe)
                                                
                        print(tabulate(self.FlightsTrainDataframe[:10], headers='keys', tablefmt='grid' , showindex=False , ))

                        #logging.info( str ( self.FlightsTrainDataframe.head()))
                        #logging.info( str ( self.FlightsTrainDataframe.shape ) )
                        
                        #logging.info ( str(  list ( self.FlightsTrainDataframe )) )
                        file_count = file_count + 1
                    
                    else:
                        self.FlightsTrainDataframe = pd.read_parquet(self.filePath)
                        self.FlightsTrainDataframe = self.renameColumns(self.FlightsTrainDataframe)

                        logging.info( str ( self.FlightsTrainDataframe.head()))
                        logging.info( str ( self.FlightsTrainDataframe.shape ) )
                        
                        logging.info ( str(  list ( self.FlightsTrainDataframe )) )
                    return True

        else:
            return False
                    
    def readRankFileForPlot(self, fileName):
        
        if str(fileName).endswith("parquet") == False:
            fileName = fileName + ".parquet"
        
        #logging.info(self.className + ": file name = " + fileName)
        filePath = os.path.join( self.filesFolderTrain , fileName)
        file = Path(filePath)
        
        assert file.is_file() == True
        
        self.FlightsTrainDataframe = pd.read_parquet(filePath)
        