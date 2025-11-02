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
finalHeaders = ['airport_icao'  , 'airport_longitude_deg' ,'airport_latitude_deg' , 'airport_elevation_ft']

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
        return (set(self.airportsDataframe) == set(finalHeaders))
    
    def isAirportInDatabase(self , ICAOcode = ""):
        return ICAOcode in self.dataChallengeAirportsDict
        
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
            
            self.airportsDataframe = pd.read_parquet ( self.filePath )
            logging.info ( self.className + ": shape = " + str(self.airportsDataframe.shape ) )
            logging.info ( self.className + ": list of headers = " +  str(  list ( self.airportsDataframe)) )
                        
            #self.airportsDataframe = df.dropna()
            print ( list ( self.airportsDataframe ))
            ''' rename columns to add a unit such as degrees '''
            self.airportsDataframe = self.airportsDataframe.rename(columns=
                                        {'icao' : 'airport_icao', 
                                         'latitude': 'airport_latitude_deg', 
                                        'longitude': 'airport_longitude_deg' ,
                                        'elevation': 'airport_elevation_ft' })
            #logging.info ( self.airportsDataframe.head(10) )
            print ("airports dataframe columns = " + str ( list ( self.airportsDataframe ) ) )
            print (  str( self.airportsDataframe.shape ))
            
            for index, row in self.airportsDataframe.iterrows():
                #logging.info("index = " + str(index))
                #print(row['airport_icao'], row['airport_longitude_deg'] , row['airport_latitude_deg'], row['airport_elevation_ft'] , )
                airport_icao_code = str(row['airport_icao']).strip()
                
                self.dataChallengeAirportsDict[airport_icao_code] = Airport (Name = airport_icao_code,
                                                                   LatitudeDegrees                   = float( row['airport_latitude_deg'] ) ,
                                                                   LongitudeDegrees                  = float( row['airport_longitude_deg'] ) ,
                                                                   fieldElevationAboveSeaLevelMeters = float( row['airport_elevation_ft']) ,
                                                                   ICAOcode                          = row['airport_icao'] ,
                                                                   Country                           = "unknown")
            
            return True
            
        else:
            self.airportsDataframe = None
            logging.error("Path = {0} is not a directory".format( directory ))
            return False
        
    def getAirportsDataframe(self):
        return self.airportsDataframe