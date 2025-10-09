'''
Created on 9 oct. 2025

@author: robert
'''

import logging
import os
from pathlib import Path
import pandas as pd
from calendar import Calendar, monthrange
from datetime import date

expectedHeaders = ['flight_id', 'timestamp', 'latitude', 'longitude', 'altitude', 'groundspeed', 'track', 'vertical_rate', 'icao24', 'u_component_of_wind', 'v_component_of_wind', 'temperature', 'specific_humidity']

def date_iter( year, month):
    for i in range(1, monthrange(year, month)[1]  + 1):
        yield date(year, month, i)

class FlightsDatabase(object):
    
    className = ''
    
    def __init__(self):
        self.className = self.__class__.__name__
        
        self.fileNameFlightsPattern = "YYYY-MM-DD.parquet"
        logging.info(self.fileNameFlightsPattern)
        
        self.filesFolder = "C:\\Users\\rober\\git\\PRCdataChallenge2025\\Data-Download-OpenSkyNetwork\\competition-train-data"
        
        logging.info(self.filesFolder)
        
        directory = Path(self.filesFolder)
        logging.info(directory)
        
        assert directory.is_dir()
        
    def checkFlightsTrainHeaders(self):
        return (set(self.FlightsTrainDataframe) == set(expectedHeaders))

        
    def readSomeFiles(self, testMode = False):
        
        first = True
        df_final = None
        yearInt = 2022
        
        #testMode = True
        if ( testMode == True ):
            theDate = date(yearInt, 1, 1)
            fileName = str(theDate) + "." + "parquet"
            logging.info(self.className + ": file name = " + fileName)
            
            self.filePath = os.path.join(self.filesFolder , fileName)
            
            file = Path(self.filePath)
            if file.is_file():
                
                self.FlightsTrainDataframe = pd.read_parquet(file)
                logging.info( str ( self.FlightsTrainDataframe.head()))
                logging.info( str ( self.FlightsTrainDataframe.shape ) )
                
                logging.info ( str(  list ( self.FlightsTrainDataframe )) )
                return True

    
        else:
            for iMonth in range(1,13):
                for d in date_iter( yearInt, iMonth):
                    print(str( d ))
                    fileName = str(d) + "." + "parquet"
                    logging.info(self.className + ": file name = " + fileName)
                    
        return False
                    