'''
Created on 12 nov. 2024

@author: robert
'''

#import sys
#sys.path.append("C:/Users/rober/git/openap/") #replace PATH with the path to Foo

from openap import prop
import json

from trajectory.Openap.AircraftConfigurationFile import OpenapAircraftConfiguration

from trajectory.Environment.Earth.EarthFile import Earth
from trajectory.Environment.Atmosphere.AtmosphereFile import Atmosphere
from trajectory.Environment.Constants import Meter2NauticalMiles

import logging
# create logger
logger = logging.getLogger()

class OpenapAircraft(OpenapAircraftConfiguration):
    
    aircraftICAOcode = ""
    openapAircraft = None

    def __init__( self, aircraftICAOcode , earth , atmosphere , initialMassKilograms ):
        
        logger.setLevel(logging.INFO)
        self.className = self.__class__.__name__
        assert isinstance (earth, Earth)
        assert isinstance (atmosphere, Atmosphere)
        super().__init__(aircraftICAOcode , earth , atmosphere)

        self.aircraftICAOcode = aircraftICAOcode
        self.openapAircraft = prop.aircraft( ac=str(aircraftICAOcode).lower(), use_synonym=True) 
        
        if (initialMassKilograms is None):
            self.setInitialMassKilograms( self.getReferenceMassKilograms() )
        else:
            self.setInitialMassKilograms(initialMassKilograms)
        logging.info ( self.className  + " --- " + self.getAircraftName() )
        
    def getAircraftName(self):
        return self.openapAircraft['aircraft']
    
    def getAircraft(self):
        return self.openapAircraft
    
    def __str__(self):
        return json.dumps( self.openapAircraft )
    
    def generateStateVectorHistoryFile( self ):
        filePrefix = "Vertical-Profile-" + str(self.aircraftICAOcode).upper()
        self.createStateVectorHistoryFile(filePrefix)
    
    def createStateVectorOutputSheet(self, workbook, abortedFlight, aircraftICAOcode, AdepICAOcode, AdesICAOcode):
        assert ( type(abortedFlight) == bool )
        filePrefix = ""
        if abortedFlight:
            filePrefix = "Aborted"
        filePrefix += "-" + aircraftICAOcode + "-" + AdepICAOcode + "-" + AdesICAOcode
        self.createStateVectorHistorySheet(workbook)
        
    def createPRCdataChallengeFlightDataframe(self , finalRoute , abortedFlight , aircraftICAOcode , 
                                              flight_id , takeOffInstant):
        return self.createAircraftPRCdataChallengeFlightDataframe(finalRoute , abortedFlight , aircraftICAOcode , flight_id , takeOffInstant)


