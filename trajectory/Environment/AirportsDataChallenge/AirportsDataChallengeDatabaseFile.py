'''
Created on 7 oct. 2025

@author: rober
'''
import os
import logging
import pandas as pd

from pathlib import Path
from trajectory.Guidance.WayPointFile import Airport

expectedHeaders = ['icao'  , 'longitude' ,'latitude' , 'elevation']

class AirportsDataChallengeDatabase(object):
    className = ''
    dataChallengeAirportsDict  = {}
    dataframe = None
    
    def __init__(self):
        
        self.className = self.__class__.__name__
        self.fileName = "apt.parquet"
        
        self.airportsFilesFolder = os.path.dirname(__file__)
        logging.info ( self.className + ': file folder= {0}'.format(self.airportsFilesFolder) )
        
        self.filePath = os.path.join(self.airportsFilesFolder , self.fileName)
        logging.info ( self.className + ': file path= {0}'.format(self.filePath) )
        
        self.airportsDataframe = None
        self.dataChallengeAirportsDict = {}
        
    def checkHeaders(self):
        return (set(self.airportsDataframe) == set(expectedHeaders))
        
    def getAirPort(self , ICAOcode = ""):
        if ICAOcode in self.dataChallengeAirportsDict:
            airport = self.dataChallengeAirportsDict[ICAOcode]
            assert  isinstance( airport , Airport )
            return airport
        else:
            return None
        
    def read(self):
        
        directory = Path(self.airportsFilesFolder)
        logging.info(directory)
        
        self.dataChallengeAirportsDict = {}
        
        file = Path(self.filePath)
        
        if directory.is_dir() and file.is_file():
            
            logging.info (self.className + "it is a directory - {0}".format(self.airportsFilesFolder))
            logging.info (self.className + "it is a file - {0}".format(self.filePath))
            
            df = pd.read_parquet ( self.filePath )
            logging.info ( self.className + ": shape = " + str(df.shape ) )
            logging.info ( self.className + ": list of headers = " +  str(  list ( df)) )
            
            #logging.info ( df.head(10) )
            
            self.airportsDataframe = df.dropna()
            #logging.info ( self.airportsDataframe.head(10) )
            
            for index, row in self.airportsDataframe.iterrows():
                #logging.info("index = " + str(index))
                #print(row['icao'], row['longitude'] , row['latitude'], row['latitude'] , )
                
                self.dataChallengeAirportsDict[row['icao']] = Airport (Name                          = row['icao'],
                                                                   LatitudeDegrees                   = float( row['latitude'] ) ,
                                                                   LongitudeDegrees                  = float( row['longitude'] ) ,
                                                                   fieldElevationAboveSeaLevelMeters = float( row['elevation']) ,
                                                                   ICAOcode                          = row['icao'] ,
                                                                   Country                           = "unknown")
            
            return True
            
        else:
            self.airportsDataframe = None
            logging.error("Path = {0} is not a directory".format( directory ))
            return False
        
    def getAirportsDataframe(self):
        return self.airportsDataframe