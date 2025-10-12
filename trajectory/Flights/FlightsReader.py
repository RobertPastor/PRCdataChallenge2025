'''
Created on 9 oct. 2025

@author: robert
'''

import logging
import os
from pathlib import Path
import pandas as pd
from tabulate import tabulate


''' type_code renamed as aircraft_type_code '''
expectedHeaders = ['timestamp', 'flight_id', 'aircraft_type_code', 'latitude', 'longitude', 'altitude', 'groundspeed', 'track', 'vertical_rate', 'mach', 'TAS', 'CAS', 'source']


class FlightsDatabase(object):
    
    className = ''
    
    def __init__(self):
        self.className = self.__class__.__name__
        
        #self.filesFolder = "C:\\Users\\rober\\git\\PRCdataChallenge2025\\Data-Download-OpenSkyNetwork\\competition-train-data"
        self.filesFolder = os.path.dirname(__file__)
        self.filesFolderTrain = os.path.join( self.filesFolder , ".." , ".." , "Data-Download-OpenSkyNetwork" , "competition-train-data")
        self.filesFolderRank = os.path.join( self.filesFolder , ".." , ".." , "Data-Download-OpenSkyNetwork" , "competition-rank-data")
        
        assert Path(self.filesFolderTrain).is_dir() == True
        assert Path(self.filesFolderRank).is_dir() == True
        #logging.info(self.filesFolder)
        
    def checkFlightsTrainHeaders(self):
        return (set(self.FlightsTrainDataframe) == set(expectedHeaders))
    
    def renameColumns(self, df):
        return df.rename(columns= {'typecode':'aircraft_type_code'})
    
    def oneHotEncodeSource(self , df , columnName):
        ''' source columns contains only two values
        INFO:root:source
        adsb     25453
        acars        8 '''
        
        # Get one hot encoding of columns B
        one_hot = pd.get_dummies(df[columnName])
        # Drop column B as it is now encoded
        df = df.drop(columnName ,axis = 1)
        # Join the encoded df
        return df.join(one_hot)
    
    def readOneTrainFile(self, fileName):
        
        if str(fileName).endswith("parquet") == False:
            fileName = fileName + ".parquet"
        
        #logging.info(self.className + ": file name = " + fileName)
        filePath = os.path.join( self.filesFolderTrain , fileName)
        file = Path(filePath)
        
        assert file.is_file() == True
        
        self.FlightsTrainDataframe = pd.read_parquet(filePath)
        self.FlightsTrainDataframe = self.renameColumns(self.FlightsTrainDataframe)
        
        ''' convert datetime to UTC '''
        self.FlightsTrainDataframe['timestamp'] = pd.to_datetime(self.FlightsTrainDataframe['timestamp'], utc=True)
        
        assert self.checkFlightsTrainHeaders()
        
        ''' rename typecode into aircraft type code '''
        self.FlightsTrainDataframe  = self.renameColumns(self.FlightsTrainDataframe )
        
        ''' one hot encode the source column '''
        ''' do not hot encode on a per file basis as some file may have only one value in the source '''
        #self.FlightsTrainDataframe  = self.oneHotEncodeSource(self.FlightsTrainDataframe, "source")
        
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


                        logging.info( str ( self.FlightsTrainDataframe.head()))
                        logging.info( str ( self.FlightsTrainDataframe.shape ) )
                        
                        logging.info ( str(  list ( self.FlightsTrainDataframe )) )
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
                    