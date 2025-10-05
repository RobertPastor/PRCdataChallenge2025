'''
Created on 15 nov. 2024

@author: robert

'''


import math
from trajectory.Environment.Constants import Knots2MetersSeconds,  MeterSecond2Knots
from trajectory.Environment.Constants import Meter2Feet
from trajectory.aerocalc.airspeed import mach_alt2cas


#sys.path.append("C:/Users/rober/git/openap/") #replace PATH with the path to Foo

from trajectory.Openap.AircraftEngineFile import OpenapAircraftEngine
from trajectory.aerocalc.airspeed import tas2mach , default_temp_units , mach2tas
import logging
# create logger
logger = logging.getLogger()

def interpolate(x , xArray, yArray):
    #x1 must be greater to x0
    x0 , x1 = xArray
    y0 , y1 = yArray
    
    y = ( ( (y1 - y0) / (x1 - x0) ) * ( x - x0 ) ) + y0
    
    return y

class OpenapAircraftSpeeds(OpenapAircraftEngine):

    maximumSpeedVmoKnots = 0.0
    initialTASknots = 0.0
    currentTASknots = 0.0
    maximumSpeedMmoMach = 0.0
    targetCruiseMach = 0.0

    def __init__(self , aircraftICAOcode):
        
        logger.setLevel(logging.INFO)
        self.className = self.__class__.__name__
        super().__init__(aircraftICAOcode)
        
        self.maximumSpeedVmoKnots = self.aircraft['vmo']
        #logger.info( self.className + " - max operational speed = {0} knots".format ( self.maximumSpeedVmoKnots ))
        
        self.maximumSpeedMmoMach = self.aircraft['mmo']
        #logger.info( self.className + " - max operational speed = {0} mach".format ( self.maximumSpeedMmoMach ))

        self.initialTASknots = 0.0
        self.currentTASknots = self.initialTASknots
        
        self.takeOffCASspeedsMeterSecondsDict = self.wrap.takeoff_speed() 
        #logger.info( self.className + " - Take Off CAS speeds = (m/s)" + json.dumps ( self.takeOffCASspeedsMeterSecondsDict ) )
        
        self.takeOffAccelerationMetersSecondsSquareDict = self.wrap.takeoff_acceleration()
        #logger.info( self.className + " - Take Off mean acceleration = {0} meters per seconds square".format( json.dumps ( self.takeOffAccelerationMetersSecondsSquareDict ) ) )

        self.landingDecelerationMetersSecondsSquareDict = self.wrap.landing_acceleration()
        #logger.info( self.className + " - Landing deceleration = {0} meters per seconds square".format( json.dumps ( self.landingDecelerationMetersSecondsSquareDict ) ) )

        self.initialDescentCASset = False
        self.initialDescentCASknots = 0.0
        self.initialDescentAltitudeFeet = 0.0
        
        self.initialClimbCASset = False
        self.initialClimbCASknots = 0.0
        self.initialClimbAltitudeFeet = 0.0
        
        self.initialApproachCASset = False
        self.initialApproachCASknots = 0.0
        self.initialApproachAltitudeFeet = 0.0
        
        self.initialCruiseTASset = False
        self.initialCruiseTASknots = 0.0
        self.initialCruiseAltitudeFeet = 0.0
        
    def getMaximumSpeedMmoMach(self):
        return self.maximumSpeedMmoMach
    
    def setTargetCruiseMach(self, targetCruiseMach ):
        logging.info( self.className + " - target cruise mach = {0:.2f}".format(targetCruiseMach))
        self.targetCruiseMach = targetCruiseMach
        
    def getTargetCruiseMach(self):
        return self.targetCruiseMach
        
    def getDefaultTakeOffCASknots(self):
        ''' @TODO correct for difference to reference mass '''
        return self.takeOffCASspeedsMeterSecondsDict['default'] * MeterSecond2Knots
    
    def getDefaultTakeOffAccelerationMetersSecondsSquare(self):
        ''' @TODO correct for difference to reference mass '''
        return self.takeOffAccelerationMetersSecondsSquareDict['default']
    
    def getMinimumTakeOffAccelerationMetersSecondsSquare(self):
        return self.takeOffAccelerationMetersSecondsSquareDict['minimum']
    
    def getDefaultLandingDecelerationMetersSecondsSquare(self):
        return self.landingDecelerationMetersSecondsSquareDict['default']
        
    def getCurrentTrueAirSpeedMetersSecond(self):
        return self.currentTASknots * Knots2MetersSeconds
        
    def getCurrentTASspeedKnots(self):
        #logger.info ( self.className + " --- current TAS = {0:.2f} knots".format( self.currentTASknots ))
        return self.currentTASknots
    
    def getCurrentTASmetersSeconds(self):
        return self.currentTASknots * Knots2MetersSeconds
    
    def setCurrentTASmetersSeconds(self, TASmetersSeconds , aircraftAltitudeMSLfeet ):
        if TASmetersSeconds < 0.0:
            #logger.info ( self.className + " - Error - TAS is negative " )
            raise ValueError( self.className + " - Error - TAS is negative ")
        if tas2mach(tas        = TASmetersSeconds ,
                    temp       = 'std',
                    altitude   = aircraftAltitudeMSLfeet,
                    temp_units = default_temp_units,
                    alt_units  = 'ft',
                    speed_units= 'm/s' ) > self.targetCruiseMach:
            self.currentTASknots = mach2tas ( mach     = self.targetCruiseMach,
                                              temp     = 'std',
                                              altitude = aircraftAltitudeMSLfeet,
                                              temp_units = default_temp_units,
                                              alt_units  = 'ft',
                                              speed_units = 'kt')
        else:
            self.currentTASknots = TASmetersSeconds * MeterSecond2Knots
        
        
    def getFinalApproachCASknots( self ):
        return self.wrap.finalapp_vcas()['default'] 
    
    def computeLandingStallSpeedCasKnots(self):
        return self.wrap.landing_speed()['default'] 
    
    def computeClimbCASknots(self , altitudeMSLfeet , CASknots ):
        
        if self.initialClimbCASset == False:
            self.initialClimbCASset = True
            self.initialClimbCASknots = CASknots
            self.initialClimbAltitudeFeet = altitudeMSLfeet
        
        self.climbCASknots = CASknots
        ''' cross over altitude when constant CAS climb starts '''
        if ( altitudeMSLfeet < self.wrap.climb_cross_alt_concas() ['default'] * 1000.0 * Meter2Feet ):
            ''' below cross over altitude when constant CAS climb starts '''
            #logging.info( self.className + " - aircraft altitude MSL {0:.2f} feet".format ( altitudeMSLfeet ))
            
            self.constantClimbCASknots = self.wrap.climb_const_vcas()['default']
            ''' xp must be in increasing order '''
            if ( self.initialClimbCASknots < self.constantClimbCASknots):
                self.climbCASknots = interpolate ( x = altitudeMSLfeet , 
                                            xArray = [ self.initialClimbAltitudeFeet , self.wrap.climb_cross_alt_concas() ['default'] * 1000.0 * Meter2Feet ],
                                            yArray = [ self.initialClimbCASknots , self.constantClimbCASknots ] )
            else:
                self.climbCASknots = self.initialClimbCASknots
                self.initialClimbCASknots = self.climbCASknots
                
        elif ( altitudeMSLfeet < self.wrap.climb_cross_alt_conmach() ['default'] * 1000.0 * Meter2Feet ):
            ''' cross over altitude from constant CAS to use constant climb mach '''
            #logging.info( self.className + " - aircraft altitude MSL {0:.2f} feet".format ( altitudeMSLfeet ))

            self.constantClimbMach = self.wrap.climb_const_mach()['default']
            
            self.constantClimbCASknots = mach_alt2cas( mach = self.constantClimbMach , 
                                                         altitude = altitudeMSLfeet , 
                                                         alt_units = 'ft',
                                                         speed_units = 'kt')
            self.climbCASknots = interpolate ( x = altitudeMSLfeet , 
                                            xArray= [ self.wrap.climb_cross_alt_concas() ['default'] * 1000.0 * Meter2Feet , self.wrap.climb_cross_alt_conmach() ['default'] * 1000.0 * Meter2Feet ],
                                            yArray = [ self.initialClimbCASknots , self.constantClimbCASknots ,   ] )
            self.lastClimbCASknots = self.climbCASknots
            
        else:
            #logging.info( self.className + " - aircraft altitude = {0:.2f} feet".format( altitudeMSLfeet ))
            self.constantClimbMach = self.wrap.climb_const_mach()['default']
            
            self.targetCruiseCASknots = mach_alt2cas( mach        = self.getTargetCruiseMach() , 
                                                      altitude    = self.getCruiseLevelFeet()  , 
                                                      alt_units   = 'ft',
                                                      speed_units = 'kt')
            #logging.info( self.className + " target cruise CAS {0:.2f} mach - {1:.2f} knots".format ( self.getTargetCruiseMach() , self.targetCruiseCASknots ))
            self.climbCASknots = interpolate ( x = altitudeMSLfeet ,
                                             xArray = [ self.wrap.climb_cross_alt_conmach() ['default'] * 1000.0 * Meter2Feet , self.getCruiseLevelFeet() ] ,
                                             yArray = [ self.lastClimbCASknots , self.targetCruiseCASknots ])
            #logging.info( self.className + " - cruise CAS = {0:.2f} knots ".format( self.climbCASknots  ))
            #logging.info( "---------")
            
        #logger.info( self.className + " - climb CAS speed = {0} kt".format (  self.climbCASknots ) )
        return self.climbCASknots
        
    def computeDescentCASknots(self , altitudeMSLfeet , CASknots ):
        
        if self.initialDescentCASset == False:
            self.initialDescentCASset = True
            self.initialDescentCASknots = CASknots
            self.initialDescentAltitudeFeet = altitudeMSLfeet
        
        ''' distance from runway = 10 Nautical miles '''
        #glideSlopeLenthFeet = 10.0 * NauticalMiles2Meter * Meter2Feet  # 10.0 Nautical miles
        ''' 3 degrees glide slope '''
        #glideSlopeHeightFeet = math.tan( math.radians( 3.0 )) * glideSlopeLenthFeet
        
        self.constantCASdescentKnots = 0.0
        
        if ( altitudeMSLfeet > self.wrap.descent_cross_alt_conmach() ['default'] * 1000.0 * Meter2Feet ):
            ''' above altitude to transition from constant Mach to constant CAS '''
            constantMachDescent = self.wrap.descent_const_mach()['default']
            self.constantCASDescentKnots = mach_alt2cas( mach = constantMachDescent , 
                                                         altitude = altitudeMSLfeet , 
                                                         alt_units = 'ft',
                                                         speed_units = 'kt')
            
            if ( CASknots > self.constantCASDescentKnots ):
                ''' xp must be in increasing order '''
                self.constantCASdescentKnots = interpolate ( x = altitudeMSLfeet , 
                                                      xArray = [ self.wrap.descent_cross_alt_conmach() ['default'] * 1000.0 * Meter2Feet , self.initialDescentAltitudeFeet],
                                                      yArray = [ self.constantCASDescentKnots , self.initialDescentCASknots ])
            else:
                self.constantCASdescentKnots = self.initialDescentCASknots
            
            
        elif ( altitudeMSLfeet <= self.wrap.descent_cross_alt_conmach() ['default'] * 1000.0 * Meter2Feet ) and \
            ( altitudeMSLfeet > self.wrap.descent_cross_alt_concas() ['default'] * 1000.0 * Meter2Feet ):
            
            ''' below altitude to transition from constant MACH to constant CAS '''
            
            constantMachDescent = self.wrap.descent_const_mach()['default']
            self.constantMachdescentKnots = mach_alt2cas( mach = constantMachDescent , 
                                                         altitude = altitudeMSLfeet , 
                                                         alt_units = 'ft',
                                                         speed_units = 'kt')
            
            self.constantCASdescentKnots = self.wrap.descent_const_vcas()['default']
            ''' xp must be in increasing order '''            
            self.constantCASdescentKnots = interpolate ( x = altitudeMSLfeet , 
                                                  xArray = [ self.wrap.descent_cross_alt_concas() ['default'] * 1000.0 * Meter2Feet , self.wrap.descent_cross_alt_conmach() ['default'] * 1000.0 * Meter2Feet ],
                                                  yArray = [ self.constantCASdescentKnots , self.initialDescentCASknots ])
             
            
        else:
            ''' low altitude to transition from constant CAS to final approach CAS '''
            altitudeConstantCASfeet = self.wrap.descent_cross_alt_concas() ['default'] * 1000.0 * Meter2Feet
            ''' altitude of final approach '''
            #altitudeFinalApproachStartFeet = altitudeConstantCASfeet - glideSlopeHeightFeet
            altitudeFinalApproachStartFeet = self.getTargetApproachWayPoint().getAltitudeMeanSeaLevelMeters() * Meter2Feet
            ''' transition speed before entering final approach configuration '''
            descentConstantCAS = self.wrap.descent_const_vcas()['default']
            finalApproachCAS = self.wrap.finalapp_vcas()['default']
            
            ''' xp must be in increasing order '''
            self.constantCASdescentKnots = interpolate ( x = altitudeMSLfeet , 
                                                       xArray = [ altitudeFinalApproachStartFeet , altitudeConstantCASfeet  ] , 
                                                       yArray = [ finalApproachCAS , descentConstantCAS ])
            
        #logger.info( self.className + " - descent CAS speed = {0:.2f} kt".format (  self.constantCASdescentKnots ) )
        self.altitudeFinalDescentMSLfeet = altitudeMSLfeet
        #logger.info( self.className + " - altitude final descent = {0:.2f} feet".format ( self.altitudeFinalDescentMSLfeet ))
        return self.constantCASdescentKnots
    
    def computeCruiseTASknots (self):
        if self.initialCruiseTASset == False:
            self.initialClimbCASset = True
            
        #logging.info( self.className + " - target cruise = {0:.2f} mach".format( self.getTargetCruiseMach() ))
        self.targetCruiseCASknots = mach2tas( mach        = self.getTargetCruiseMach() ,
                                              temp        = 'std',
                                                altitude    = self.getCruiseLevelFeet()  , 
                                                temp_units  = default_temp_units,
                                                alt_units   = 'ft',
                                                speed_units = 'kt')
        return self.targetCruiseCASknots
        
    
    def computeApproachCASknots(self , altitudeMSLfeet , currentCASknots , arrivalRunwayAltitudeMSLfeet ):
        ''' approach is flying the last turn followed by the descent glide slope to the arrival runway '''
        if self.initialApproachCASset == False:
            self.initialApproachCASset = True
            self.initialApproachCASknots = currentCASknots
            self.initialApproachAltitudeFeet = altitudeMSLfeet
        
        #logger.info( self.className + " - altitude final descent = {0:.2f} feet".format ( self.altitudeFinalDescentMSLfeet ))
        #logger.info#( self.className + " - altitude arrival runway = {0:.2f} feet".format ( arrivalRunwayAltitudeMSLfeet ))
        assert ( self.altitudeFinalDescentMSLfeet >= arrivalRunwayAltitudeMSLfeet )
        
        #self.approachCASknots = self.wrap.finalapp_vcas()['default']
        #logger.info( self.className + " - approach CAS = {0:.2f} knots".format( self.approachCASknots ))
        
        #logging.info( self.className + " - last descent CAS = {0:.2f} knots".format( self.constantCASdescentKnots ))
        
        self.landingCASknots = self.wrap.landing_speed()['default']
        #logger.info( self.className + " - landing CAS = {0:.2f} knots".format( self.landingCASknots ))
        
        ''' interpolate from current altitude to runway altitude '''
        ''' interpolate from current CAS to landing CAS '''
        ''' xp must be in increasing order '''
        self.approachCASknots = interpolate ( x = altitudeMSLfeet , 
                                            xArray = [ arrivalRunwayAltitudeMSLfeet , self.altitudeFinalDescentMSLfeet  ] , 
                                            yArray = [ self.landingCASknots , self.constantCASdescentKnots ])
        
        #logging.info( self.className + ' - approach CAS = {0:.2f} knots'.format( self.approachCASknots ))
        return self.approachCASknots
    
    
        
    def isCruiseSpeedReached(self):
        return False