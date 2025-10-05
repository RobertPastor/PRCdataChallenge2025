'''
Created on 26 nov. 2024

@author: robert

'''

import logging 
logger = logging.getLogger(__name__)

from trajectory.aerocalc.airspeed import tas2cas, tas2mach, default_temp_units

from trajectory.Environment.Constants import  Meter2Feet , MeterSecond2Knots
from trajectory.Environment.Constants import  Meter2NauticalMiles
from trajectory.Openap.AircraftStateVectorFile import OpenapAircraftStateVector


class OpenapAircraftFlightPhases(OpenapAircraftStateVector):
    ''' openap wrap flight phases '''
    aircraftConfigurationList = ['take-off', 
                                 'initial-climb', 
                                 'climb',    
                                 'cruise', 
                                 'descent',
                                 'approach', 
                                 'landing']
    aircraftCurrentConfiguration = ''
    totalDistanceFlownMeters = 0.0
 
    def __init__(self , aircraftICAOcode ):
        
        self.className = self.__class__.__name__
        self.aircraftICAOcode = aircraftICAOcode
        super().__init__(aircraftICAOcode)

        self.aircraftCurrentConfiguration = self.aircraftConfigurationList[0]
        
        logger.info ( self.className  + ' ===================================================' )
        self.flightPathAngleDegrees = 0.0
        logger.info ( self.className + ' default configuration= ' + self.aircraftCurrentConfiguration )
        logger.info ( self.className + ' ===================================================' )
        
        self.totalDistanceFlownMeters = 0.0
        
    def setTotalDistanceFlownMeters(self, totalDistanceFlownMeters):
        self.totalDistanceFlownMeters = totalDistanceFlownMeters
        
    def getTotalDistanceFlownMeters(self):
        return self.totalDistanceFlownMeters 
        
    def getCurrentConfiguration(self):
        return self.aircraftCurrentConfiguration
    
    def setTakeOffConfiguration(self, elapsedTimeSeconds):
        ''' take off starts at the end of the ground-run when speed > 1.2 * Take-off stall speed '''
        ''' high lifting devices are used '''
        newConfiguration = 'take-off'
        if self.aircraftCurrentConfiguration != newConfiguration:
            self.showConfigurationChange(newConfiguration, elapsedTimeSeconds)
            self.aircraftCurrentConfiguration = newConfiguration
        
    def setInitialClimbConfiguration(self, elapsedTimeSeconds):
        ''' high lifting devices are used - gear are retracted '''
        newConfiguration = 'initial-climb'
        if self.aircraftCurrentConfiguration != newConfiguration:
            self.showConfigurationChange(newConfiguration, elapsedTimeSeconds)
            self.aircraftCurrentConfiguration = newConfiguration
            
    def setClimbConfiguration(self , elapsedTimeSeconds):
        ''' almost clean configuration - gear is retracted '''
        newConfiguration = 'climb'
        if self.aircraftCurrentConfiguration != newConfiguration:
            self.showConfigurationChange(newConfiguration, elapsedTimeSeconds)
            self.aircraftCurrentConfiguration = newConfiguration
            
    def setCruiseConfiguration(self, elapsedTimeSeconds):
        newConfiguration = 'cruise'
        if self.aircraftCurrentConfiguration != newConfiguration:
            self.showConfigurationChange(newConfiguration, elapsedTimeSeconds)
            self.aircraftCurrentConfiguration = newConfiguration
 
    def setDescentConfiguration(self , elapsedTimeSeconds):
        newConfiguration = 'descent'
        if self.aircraftCurrentConfiguration != newConfiguration:
            self.showConfigurationChange(newConfiguration, elapsedTimeSeconds)
            self.aircraftCurrentConfiguration = newConfiguration
            
    def setFinalApproachConfiguration(self, elapsedTimeSeconds):
        newConfiguration = 'approach'
        if self.aircraftCurrentConfiguration != newConfiguration:
            self.showConfigurationChange(newConfiguration, elapsedTimeSeconds)
            self.aircraftCurrentConfiguration = newConfiguration
            
    def setLandingConfiguration(self, elapsedTimeSeconds):
        newConfiguration = 'landing'
        if self.aircraftCurrentConfiguration != newConfiguration:
            self.showConfigurationChange(newConfiguration, elapsedTimeSeconds)
            self.aircraftCurrentConfiguration = newConfiguration
    
    def isTakeOff(self):
        return (self.aircraftCurrentConfiguration=='take-off')
    
    def isInitialClimb(self):
        return (self.aircraftCurrentConfiguration=='initial-climb')

    def isClimb(self):
        return (self.aircraftCurrentConfiguration=='climb')
            
    def isCruise(self):
        return (self.aircraftCurrentConfiguration=='cruise')
    
    def isDescent(self):
        return (self.aircraftCurrentConfiguration=='descent')
    
    def isApproach(self):
        return (self.aircraftCurrentConfiguration=='approach')
    
    def isLanding(self):
        return (self.aircraftCurrentConfiguration=='landing')
            
    def showConfigurationChange(self, newConfiguration, elapsedTimeSeconds):
        assert isinstance(newConfiguration, str)
        altitudeMeanSeaLevelMeters = self.getAltitudeMSLmeters()
        currentDistanceFlownMeters = self.getTotalDistanceFlownMeters()
        tas = self.getCurrentTrueAirSpeedMetersSecond()
        #cas = self.atmosphere.tas2cas(tas = tas, altitude = altitudeMeanSeaLevelMeters,alt_units='m', speed_units='m/s',)
        cas = tas2cas(tas = tas, altitude = altitudeMeanSeaLevelMeters, temp='std', speed_units='m/s', alt_units='m')
        #mach = self.atmosphere.tas2mach(tas = tas, altitude = altitudeMeanSeaLevelMeters, alt_units='m', speed_units='m/s')
        mach = tas2mach(tas = tas , temp='std', altitude = altitudeMeanSeaLevelMeters, temp_units = default_temp_units, alt_units = 'm' , speed_units='m/s')
        logger.info ( self.className + ' ====================================' )
        logger.info ( self.className + ' elapsed time = {0:.0f} seconds ->  {1:.2f} hours '.format(elapsedTimeSeconds , elapsedTimeSeconds/3600.0))
        logger.info ( self.className + ' entering {0} configuration - distance flown {1:.2f} meters - distance flown {2:.2f} Nm'.format(newConfiguration, currentDistanceFlownMeters, currentDistanceFlownMeters*Meter2NauticalMiles) )
        logger.info ( self.className + ' alt= {0:.2f} meters - alt= {1:.2f} feet'.format(altitudeMeanSeaLevelMeters, altitudeMeanSeaLevelMeters * Meter2Feet ) ) 
        logger.info ( self.className + ' TAS= {0:.2f} m/s - TAS= {1:.2f} knots - CAS= {2:.2f} m/s - CAS= {3:.2f} knots - Mach= {4:.2f}'.format(tas, (tas*MeterSecond2Knots), cas, (cas*MeterSecond2Knots), mach) )
        logger.info ( self.className + ' ====================================' )


