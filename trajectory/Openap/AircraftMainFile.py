'''
Created on 12 nov. 2024

@author: robert
'''

#import sys
#sys.path.append("C:/Users/rober/git/openap/") #replace PATH with the path to Foo

from openap import prop
import json

from trajectory.Openap.AircraftConfigurationFile import OpenapAircraftConfiguration
import time

from trajectory.Environment.Earth import Earth
from trajectory.Environment.Atmosphere import Atmosphere
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

    
if __name__ == '__main__':
    
    earth = Earth()
    atmosphere = Atmosphere()
    
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    print("-"*80)
    
    departureRunwayAltitudeMSLmeters = 300.0
    
    available_acs = prop.available_aircraft(use_synonym=True)

    for actype in available_acs:
        start = time.time()
        #print(actype)
        
        #if ( str( actype ).lower() not in ['a320'] ):
        #    continue
        
        if ( str( actype ).lower() in ['a359','a388','b38m','b744','b748','b752','b763','b773','b77w','b788','b789','c550'] \
             or str( actype ).lower() in ['e145','glf6','a124','a306','a310','at72','at75','at76','b733','b735','b762','b77l'] \
             or str ( actype ).lower() in ['c25a','c525','c56x','crj2','crj9','e290','glf5','gl5t','gl6t','tj45','md11','pc24','su95','lj45','bx3m'] ):
            
            continue
        
        elapsedTimeSeconds = 0.0
        deltaTimeSeconds = 1.0
        totalDistanceFlownMeters = 0.0
        
        aircraft = prop.aircraft(ac=actype, use_synonym=True)
        
        ac = OpenapAircraft( actype , earth , atmosphere , initialMassKilograms = None)
        ac.setDepartureRunwayMSLmeters( departureRunwayAltitudeMSLmeters )
        ac.setArrivalRunwayMSLmeters( departureRunwayAltitudeMSLmeters )
    
        initialMassKilograms = ac.getReferenceMassKilograms()
        print("reference mass = {0} kilograms".format( initialMassKilograms ))
        logging.info( ac )
    
        ac.setCruiseLevelFeet()
        altitudeMSLmeters = departureRunwayAltitudeMSLmeters
        
        try:
            while ( ac.isApproach() == False ):
                totalDistanceFlownMeters , altitudeMSLmeters = ac.fly(elapsedTimeSeconds       = elapsedTimeSeconds , 
                                                                      deltaTimeSeconds         = deltaTimeSeconds ,
                                                                      totalDistanceFlownMeters = totalDistanceFlownMeters , 
                                                                      altitudeMSLmeters        = altitudeMSLmeters )
                
                elapsedTimeSeconds = elapsedTimeSeconds + deltaTimeSeconds 
                
                logger.info( " - distance flown = {0:.2f} meters - distance flown = {1:.2f} Nautical miles ".format( totalDistanceFlownMeters , totalDistanceFlownMeters * Meter2NauticalMiles ))
        except Exception as e:
            print ( "main - exception = {0}".format( e ) )
            
        print ( "duration {0:.2f} seconds".format( time.time() - start ) )
        
        ac.generateStateVectorHistoryFile()
