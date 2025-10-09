'''
Created on 9 oct. 2025

@author: robert
'''

import logging
import os
from pathlib import Path
import pandas as pd

from datetime import date

expectedHeaders = ['timestamp', 'flight_id', 'typecode', 'latitude', 'longitude', 'altitude', 'groundspeed', 'track', 'vertical_rate', 'mach', 'TAS', 'CAS', 'source']


class FlightsDatabase(object):
    
    className = ''
    
    def __init__(self):
        self.className = self.__class__.__name__
        
        self.filesFolder = "C:\\Users\\rober\\git\\PRCdataChallenge2025\\Data-Download-OpenSkyNetwork\\competition-train-data"
        
        logging.info(self.filesFolder)
        
        directory = Path(self.filesFolder)
        logging.info(directory)
        
        assert directory.is_dir()
        
    def checkFlightsTrainHeaders(self):
        return (set(self.FlightsTrainDataframe) == set(expectedHeaders))
        
    def readSomeFiles(self, testMode = False):
        file_count = 0
        if testMode == True:
            file_count = 0
            
        directory = Path(self.filesFolder)
        if directory.is_dir():
            
            for fileName in os.listdir(directory):
                logging.info(self.className + ": file name = " + fileName)
                
                self.filePath = os.path.join(self.filesFolder , fileName)
                
                file = Path(self.filePath)
                if file.is_file() and fileName.endswith("parquet"):
                    
                    if (testMode == True) and (file_count < 10):
                        
                        self.FlightsTrainDataframe = pd.read_parquet(self.filePath)
                        logging.info( str ( self.FlightsTrainDataframe.head()))
                        logging.info( str ( self.FlightsTrainDataframe.shape ) )
                        
                        logging.info ( str(  list ( self.FlightsTrainDataframe )) )
                        file_count = file_count + 1
                    
                    else:
                        self.FlightsTrainDataframe = pd.read_parquet(self.filePath)
                        logging.info( str ( self.FlightsTrainDataframe.head()))
                        logging.info( str ( self.FlightsTrainDataframe.shape ) )
                        
                        logging.info ( str(  list ( self.FlightsTrainDataframe )) )
                    return True

    
        else:
            return False
                    