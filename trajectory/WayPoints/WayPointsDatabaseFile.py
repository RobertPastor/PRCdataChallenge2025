'''
Created on 24 déc. 2022

@author: robert
'''

import os
import logging
import pandas as pd

from trajectory.Guidance.WayPointFile import WayPoint
from trajectory.Guidance.Utils import convertDegreeMinuteSecondToDecimal

fieldNames = ['WayPoint', 'Country' , 'Type', 'Latitude', 'Longitude' , 'Name']

class WayPointsDatabase(object):
    WayPointsDict = {}
    ColumnNames = []
    className = ''
    
    def __init__(self):
        self.className = self.__class__.__name__
        
        logging.info(self.className + ": ----- WayPointsDatabase init -----")
        
        self.FileName = 'WayPoints.xlsx'  
        self.FilesFolder = os.path.dirname(__file__)

        logging.info ( self.className + ': file folder= {0}'.format(self.FilesFolder) )
        self.FilePath = os.path.abspath(self.FilesFolder + os.path.sep + self.FileName)
        logging.info ( self.className + ': file path= {0}'.format(self.FilePath) )

        self.WayPointsDict = {}
        self.ColumnNames = {}
        
        self.sheetName = "WayPoints"

    def exists(self):
        logging.info(self.className + ": path exists = {0}".format(os.path.exists(self.FilesFolder)))
        logging.info(self.className + ": file exists = {0}".format(os.path.isfile(self.FilePath)))
        return os.path.exists(self.FilesFolder) and os.path.isfile(self.FilePath)
    
    def computeContinent(self , wayPointName ):
        wayPointAsDict = {}
        if wayPointName in self.WayPointsDict:
            
            wayPointAsDict = self.WayPointsDict[wayPointName]
            
            Continent = 'Unknown-continent'
            if (wayPointAsDict['Latitude'] >= 20. and wayPointAsDict['Longitude'] >= -170. and wayPointAsDict['Longitude'] < -50.):
                Continent = 'North America'
            if (wayPointAsDict['Latitude'] >= 35. and wayPointAsDict['Longitude'] >= -50. and wayPointAsDict['Longitude'] < 50.):
                Continent = 'Europe'
            if (wayPointAsDict['Latitude'] >= 5. and wayPointAsDict['Longitude'] >= 50. and wayPointAsDict['Longitude'] < 90.):
                Continent = 'India'
                
            logging.info(Continent)
            return Continent
        else:
            return "Unknown-continent"
    
    def getWayPoint(self, wayPointName ):
        logging.info( wayPointName )
        wayPointAsDict = {}
        if wayPointName in self.WayPointsDict:
            logging.info(self.className + ": wayPoint with name = {0} found in database".format( wayPointName ))
            wayPointAsDict = self.WayPointsDict[wayPointName]
            logging.info(wayPointAsDict)
            return WayPoint(Name = wayPointName , 
                            LatitudeDegrees = wayPointAsDict['Latitude'],
                            LongitudeDegrees = wayPointAsDict['Longitude'] ,
                            AltitudeMeanSeaLevelMeters = 0.0)
        else:
            logging.info( "WayPoint = {0} not available in the WayPoints database ".format(wayPointName) )
            return None
    
    def read(self):
        assert len(self.FilePath)>0
        
        if self.exists():
            df_source = pd.DataFrame(pd.read_excel(self.FilePath, sheet_name=self.sheetName , engine="openpyxl"))
            
            for index, row in df_source.iterrows():
                #logging.info('Index is: {}'.format(index))
                #logging.info('ID is: {} - WayPoint is: {} - Latitude = {} - Longitude = {}'.format(index, row['WayPoint'], row['Latitude'], row['Longitude']))
                
                WayPointName = str(row['WayPoint']).strip().upper()
                if not(WayPointName in self.WayPointsDict.keys()):
                    
                    strLatitude = str(row['Latitude']).strip()
                    strLongitude = str(row['Longitude']).strip()
                    
                    wayPointDict = {}
                    wayPointDict["WayPoint"] = WayPointName
                    
                    if '°' in strLatitude:
                        strLatitude = str(strLatitude).replace('°','-')
        
                        strLatitude = str(strLatitude).strip().replace("'", '-').replace(' ','').replace('"','')
                        wayPointDict["Latitude"] = convertDegreeMinuteSecondToDecimal(strLatitude)
                        
                    if '°' in strLongitude:
                        strLongitude = str(strLongitude).replace('°','-')
        
                        strLongitude = str(strLongitude).strip().replace("'", '-').replace(' ','').replace('"','')
                        wayPointDict["Longitude"] = convertDegreeMinuteSecondToDecimal(strLongitude)
    
                    self.WayPointsDict[WayPointName] = wayPointDict
                    
                    ''' create a way point '''    
                else:
                    print ("duplicates found in Way Points database - way Point= {0}".format(WayPointName))
                    return False
            
            return True
        else:
            return False
                    
