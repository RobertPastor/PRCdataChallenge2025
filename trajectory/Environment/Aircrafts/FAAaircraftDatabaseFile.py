'''
Created on 20 oct. 2024

@author: robert

'''

import os
import pandas as pd
from pathlib import Path


class FaaAircraftDatabase(object):
    

    dataframe = None
    
    def __init__(self):
        
        filesFolder = os.path.dirname(__file__)
        self.aircraftFileName = "FAA-Aircraft-Char-DB-AC-150-5300-13B-App-2023-09-07.xlsx"
        self.aircraftFilePath = os.path.join(filesFolder , self.aircraftFileName)
        
        self.directoryPath = Path(filesFolder)
        self.filePath = Path ( self.aircraftFilePath )
        
        if self.directoryPath.is_dir() and self.filePath.is_file():
            
            print ( "it is a directory - {0}".format(self.directoryPath))
            filePath = os.path.join(self.directoryPath, self.aircraftFileName)
            
            print ( filePath )
            
    def exists(self):
        return self.directoryPath.is_dir() and self.filePath.is_file()
            
    def read(self):
        pass
        if self.directoryPath.is_dir() and self.filePath.is_file():

            self.df_aircrafts = pd.read_excel( self.aircraftFilePath )
            print ( self.df_aircrafts.shape )
            print ( list ( self.df_aircrafts ) )
            #print ( self.df_aircrafts.head() )
            
            return True
        return False
    
    def getListOfExtendedCharacteristics(self):
        return ["MTOW_lb" , "MALW_lb" , "Num_Engines", "Approach_Speed_knot" , "Wingspan_ft_without_winglets_sharklets" , "Length_ft", "Parking_Area_ft2"]
    
    def createCaracteristicsDataframe(self , ICAOcode = ""):
        assert self.isICAOcodeExisting(ICAOcode) == True
        for aircraft_type in self.df_aircrafts['ICAO_Code']:
            if ( str(aircraft_type) == ICAOcode ):
                aircraftRecord  = self.df_aircrafts.loc[self.df_aircrafts['ICAO_Code'] == ICAOcode]

                data = { "aircraft_type" : ICAOcode ,
                        "MTOW_lb" : aircraftRecord.iloc[0]["MTOW_lb"],
                        "MALW_lb" : aircraftRecord.iloc[0]["MALW_lb"],
                        "Num_Engines" : aircraftRecord.iloc[0]["Num_Engines"],
                        "Approach_Speed_knot" : aircraftRecord.iloc[0]["Approach_Speed_knot"],
                        "Wingspan_ft_without_winglets_sharklets" : aircraftRecord.iloc[0]["Wingspan_ft_without_winglets_sharklets"],
                        "Length_ft" : aircraftRecord.iloc[0]["Length_ft"],
                        "Parking_Area_ft2" : aircraftRecord.iloc[0]["Parking_Area_ft2"],
                        }
                return pd.DataFrame(data , index=range(1))
        return None
    
    def isICAOcodeExisting(self, ICAOcode = ""):
        for aircraft_type in self.df_aircrafts['ICAO_Code']:
            if ( str(aircraft_type) == ICAOcode ):
                return True
        return False
    
    def getGenericCaracteristic(self , ICAOcode = "" , ColumnName = ""):
        
        for aircraft_type in self.df_aircrafts['ICAO_Code']:
            if ( str(aircraft_type) == ICAOcode ):
                
                record  = self.df_aircrafts.loc[self.df_aircrafts['ICAO_Code'] == ICAOcode]
                #print ( df_MTOW_lb['MTOW_lb'] )
                recordValue = record.iloc[0][ColumnName] 
                #print ( mass )
                return recordValue
        return 0.0 
        

    def getMTOW_lb(self, ICAOcode = ""):
        for aircraft_type in self.df_aircrafts['ICAO_Code']:
            if ( str(aircraft_type) == ICAOcode ):
                
                df_MTOW_lb  = self.df_aircrafts.loc[self.df_aircrafts['ICAO_Code'] == ICAOcode]
                #print ( df_MTOW_lb['MTOW_lb'] )
                mass = df_MTOW_lb.iloc[0]['MTOW_lb'] 
                #print ( mass )
                return mass
        return 0.0 
    
    def getMALW_lb(self, ICAOcode = ""):
        for aircraft_type in self.df_aircrafts['ICAO_Code']:
            if ( str(aircraft_type) == ICAOcode ):
                
                df_MALW_lb  = self.df_aircrafts.loc[self.df_aircrafts['ICAO_Code'] == ICAOcode]
                #print ( df_MALW_lb['MALW_lb'] )
                mass = df_MALW_lb.iloc[0]['MALW_lb']
                #print ( mass )
                return mass
        return 0.0 
               
    def getPhysicalClassEngine(self, ICAOcode = ""):
        for aircraft_type in self.df_aircrafts['ICAO_Code']:
            if ( str(aircraft_type) == ICAOcode ):
                
                engineClass  = self.df_aircrafts.loc[self.df_aircrafts['ICAO_Code'] == ICAOcode]
                #print ( df_MALW_lb['MALW_lb'] )
                physicalEngineClass = engineClass.iloc[0]['Physical_Class_Engine']
                #print ( mass )
                return physicalEngineClass
        return "Jet"

    def getNumberOfEngines(self, ICAOcode = ""):
        for aircraft_type in self.df_aircrafts['ICAO_Code']:
            if ( str(aircraft_type) == ICAOcode ):
                
                numberOfEngines  = self.df_aircrafts.loc[self.df_aircrafts['ICAO_Code'] == ICAOcode]
                #print ( df_MALW_lb['MALW_lb'] )
                nbEngines = numberOfEngines.iloc[0]['Num_Engines']
                #print ( mass )
                return nbEngines
        return 2.0
    
    def getApproachSpeedKnots(self, ICAOcode = ""):
        for aircraft_type in self.df_aircrafts['ICAO_Code']:
            if ( str(aircraft_type) == ICAOcode ):
                
                approachSpeedKnots  = self.df_aircrafts.loc[self.df_aircrafts['ICAO_Code'] == ICAOcode]
                #print ( df_MALW_lb['MALW_lb'] )
                approachSpeedKnots = approachSpeedKnots.iloc[0]['Approach_Speed_knot']
                #print ( mass )
                return approachSpeedKnots
        return 0.0
 
    def getWingSpanFt(self, ICAOcode = ""):
        for aircraft_type in self.df_aircrafts['ICAO_Code']:
            if ( str(aircraft_type) == ICAOcode ):
                
                WingSpanFt  = self.df_aircrafts.loc[self.df_aircrafts['ICAO_Code'] == ICAOcode]
                #print ( df_MALW_lb['MALW_lb'] )
                WingSpanFt = WingSpanFt.iloc[0]['Wingspan_ft_without_winglets_sharklets']
                #print ( mass )
                return WingSpanFt
        return 0.0
 
    def getLengthFt(self, ICAOcode = ""):
        for aircraft_type in self.df_aircrafts['ICAO_Code']:
            if ( str(aircraft_type) == ICAOcode ):
                
                LengthFt  = self.df_aircrafts.loc[self.df_aircrafts['ICAO_Code'] == ICAOcode]
                #print ( df_MALW_lb['MALW_lb'] )
                LengthFt = LengthFt.iloc[0]['Length_ft']
                #print ( mass )
                return LengthFt
        return 0.0
    
    def getParkingAreaM2 (self, ICAOcode =""):
        # Parking_Area_ft2
        for aircraft_type in self.df_aircrafts['ICAO_Code']:
            if ( str(aircraft_type) == ICAOcode ):
                
                ParkingAreaM2  = self.df_aircrafts.loc[self.df_aircrafts['ICAO_Code'] == ICAOcode]
                #print ( df_MALW_lb['MALW_lb'] )
                ParkingAreaM2 = ParkingAreaM2.iloc[0]['Parking_Area_ft2']
                #print ( mass )
                return ParkingAreaM2
        return 0.0