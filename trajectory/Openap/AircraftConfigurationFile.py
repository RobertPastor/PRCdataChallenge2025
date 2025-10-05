'''
Created on 14 nov. 2024

@author: robert
'''

#sys.path.append("C:/Users/rober/git/openap/") #replace PATH with the path to Foo

from trajectory.Environment.Constants import Meter2Feet , MeterSecond2Knots, RollingFrictionCoefficient
from trajectory.Environment.Constants import Meter2NauticalMiles, FeetMinutes2MetersSeconds
from trajectory.Environment.Constants import MeterSeconds2FeetMinutes 
from trajectory.Environment.Constants import Knots2MetersSeconds , ConstantTaxiSpeedCasKnots

import logging 
import math
from trajectory.Guidance.ConstraintsFile import feet2Meters

logger = logging.getLogger(__name__)
from trajectory.Openap.AircaftSpeedsFile import OpenapAircraftSpeeds
from trajectory.aerocalc.airspeed import tas2cas, cas2tas

class OpenapAircraftConfiguration(OpenapAircraftSpeeds):
    
    rollingFrictionCoefficient = RollingFrictionCoefficient        # rolling friction coefficient (mur)
    
    def __init__(self , aircraftICAOcode , earth , atmosphere ):
        
        logger.setLevel(logging.INFO)
        self.className = self.__class__.__name__
        super().__init__(aircraftICAOcode)
        
        self.earth = earth
        self.atmosphere = atmosphere
        
        self.distanceFlownMeters = 0.0
        self.ceilingMeters = self.aircraft['ceiling']
        
    def computeLiftCoeff(self, aircraftMassKilograms, altitudeMSLmeters, TrueAirSpeedMetersSecond, latitudeDegrees):
        '''
        lift coeff = ( 2 * aircraft-mass * gravity ) / ( rho * TAS * TAS * WingSurface ) 
        '''
        if self.isTakeOff():
            liftCoeff = 0.1
        else:
            gravityCenterMetersPerSquaredSeconds = self.earth.gravityWelmec( heightMSLmeters = altitudeMSLmeters , latitudeDegrees = latitudeDegrees)
            airDensity = self.atmosphere.getAirDensityKilogramsPerCubicMeters(altitudeMSLmeters)
            liftCoeff = 2 * aircraftMassKilograms * gravityCenterMetersPerSquaredSeconds
            if TrueAirSpeedMetersSecond > 0.0:
                wingAreaSurfaceSquareMeters = self.getWingAreaSurfaceSquareMeters()
                liftCoeff = liftCoeff / ( airDensity * TrueAirSpeedMetersSecond * TrueAirSpeedMetersSecond * wingAreaSurfaceSquareMeters)
            else:
                liftCoeff = 0.0
        return liftCoeff

        
    def computeLiftNewtons(self, aircraftMassKilograms, altitudeMeanSeaLevelMeters, trueAirSpeedMetersSecond, latitudeDegrees):
        '''  Qinf = 0.5 * rho * (aircraftSpeed ** 2) '''
        '''  Lift = Qinf * aircraft.WingPlanformAreaSquareMeters * CL '''
        ''' @todo '''
        #@TODO logger.info 'to be fixed... aircraft mass ... depends upon flight path range in Kilometers + reserve'
        liftCoeff = self.computeLiftCoeff(  aircraftMassKilograms, altitudeMeanSeaLevelMeters, trueAirSpeedMetersSecond, latitudeDegrees)
        liftNewtons = 0.5 * self.atmosphere.getAirDensityKilogramsPerCubicMeters(altitudeMeanSeaLevelMeters)
        liftNewtons = liftNewtons * trueAirSpeedMetersSecond * trueAirSpeedMetersSecond
        liftNewtons = liftNewtons * self.getWingAreaSurfaceSquareMeters() * liftCoeff
        #logger.info ( self.className + ' - lift = {0:.2f} Newtons'.format( liftNewtons ) )

        return liftNewtons
    
    def computeApproachFlightPathAngleDegrees(self , currentPosition ):
        ''' current position versus touch down position '''
        #logging.info( self.className + " - current position = {0} ".format( str ( currentPosition ) ) )
        arrivalRunwayTouchDownPoint = self.getArrivalRunwayTouchDownPoint()
        
        distanceToArrivalRunwayTouchDownPointMeters =  currentPosition.getDistanceMetersTo( arrivalRunwayTouchDownPoint )
        #logging.info( self.className + " - distance to touch down = {0:.2f} meters".format( distanceToArrivalRunwayTouchDownPointMeters ))
        
        altitudeDifferenceMeters = currentPosition.getAltitudeMeanSeaLevelMeters() - arrivalRunwayTouchDownPoint.getAltitudeMeanSeaLevelMeters()
        #logging.info( self.className + " - altitude difference to touch down = {0:.2f} meters".format( altitudeDifferenceMeters ))
        
        ''' coté opposé sur coté adjacent '''
        tangenteAlphaRadians = math.atan ( altitudeDifferenceMeters / distanceToArrivalRunwayTouchDownPointMeters )
        tangenteAlphaDegrees = math.degrees( tangenteAlphaRadians )
        #logging.info( self.className + " - Approach flight path angle = {0:.2f} degrees ".format( tangenteAlphaDegrees ) )
        return tangenteAlphaDegrees
        
        
    def computeFlightPathAngleDegrees(self , rateOfClimbMetersPerSeconds = 0.0, trueAirspeedMetersPerSeconds = 0.0):
        if ( self.aircraftCurrentConfiguration == 'departure-ground-run' ):
            return 0.0
        else:
            if trueAirspeedMetersPerSeconds < 3e-3:
                return 0.0
            try:
                flightPathAngleDegrees = min ( math.degrees ( math.asin( rateOfClimbMetersPerSeconds / trueAirspeedMetersPerSeconds)) , 4.0 )
            except Exception as e:
                print( self.className + " - exception = {0}".format( e ))
                return 0.0
            #logger.info ( self.className + ' - flight path angle = {0:.2f} degrees'.format( flightPathAngleDegrees) )
            return flightPathAngleDegrees
        
    
    def setAltitudeMSLfeet(self , altitudeMSLfeet ):
        self.altitudeMSLmeters = altitudeMSLfeet * feet2Meters
        
    def setAltitudeMSLmeters(self , altitudeMSLmeters):
        if ( altitudeMSLmeters * Meter2Feet > self.cruiseLevelFeet ):
            self.altitudeMSLmeters = self.cruiseLevelFeet * Meter2Feet
            
        elif ( altitudeMSLmeters < self.getArrivalRunwayMSLmeters() ):
            self.altitudeMSLmeters = self.altitudeMSLmeters
            
        else:
            self.altitudeMSLmeters = altitudeMSLmeters
        
    def getAltitudeMSLmeters(self):
        return self.altitudeMSLmeters
    
    def getCurrentAltitudeSeaLevelMeters(self):
        return self.altitudeMSLmeters
    
    def getAircraftAltitudeMSLmeters(self):
        return self.altitudeMSLmeters
    
    #def setCruiseLevelFeet(self ):
    #    logger.info ( self.className + json.dumps ( self.wrap.cruise_alt() ) )
    #    self.cruiseLevelFeet = self.wrap.cruise_alt()['default'] * 1000.0 * Meter2Feet
        
    def setCruiseLevelFeet(self , cruiseRequestedFlightLevelFeet ):
        self.cruiseLevelFeet = cruiseRequestedFlightLevelFeet
        
    def getCruiseLevelFeet(self):
        return self.cruiseLevelFeet
    
    def getMaxCruiseAltitudeFeet(self):
        #self.maxCruiseAltitudeFeet = self.wrap.cruise_max_alt()['default'] * 1000.0 * Meter2Feet
        self.maxCruiseAltitudeFeet = self.ceilingMeters * Meter2Feet
        return self.maxCruiseAltitudeFeet
    
    def setDepartureRunwayMSLmeters(self, departureRunwayMSLmeters ):
        self.departureRunwayMSLmeters = departureRunwayMSLmeters
        
    def getDepartureRunwayMSLmeters(self):
        return self.departureRunwayMSLmeters
    
    def setArrivalRunwayMSLmeters(self , arrivalRunwayMSLmeters ):
        self.arrivalRunwayMSLmeters = arrivalRunwayMSLmeters
        
    def getArrivalRunwayMSLmeters(self):
        return self.arrivalRunwayMSLmeters
    
    def computeRateOfClimbFeetMinutes(self , rateOfClimbFeetMinutes , altitudeMSLfeet ):
        if ( altitudeMSLfeet < ( self.getCruiseLevelFeet() - 2500.0 )):
            return rateOfClimbFeetMinutes
        else:
            if (altitudeMSLfeet < ( self.cruiseLevelFeet - 100.0 ) ):
                ''' 5 minutes to close the gap '''
                rateOfClimbFeetMinutes = abs ( self.getCruiseLevelFeet() - altitudeMSLfeet ) / 5.0
            else:
                rateOfClimbFeetMinutes = 100.0
        return rateOfClimbFeetMinutes
    
    def fly(self , elapsedTimeSeconds, 
            deltaTimeSeconds, 
            totalDistanceFlownMeters = 0.0,
            altitudeMSLmeters        = 0.0,
            distanceStillToFlyMeters = 0.0,
            currentPosition          = None,
            distanceToLastFixMeters  = None):
        '''
        main aircraft entry point : computes for a delta time 
        1) the ground distance flown (hence needs a ground speed)
        Needs = ground speed => obtained from TAS and Wind speed
        2) the delta increase - decrease in altitude
        
        '''
        endOfSimulation = False
        
        self.setTotalDistanceFlownMeters(totalDistanceFlownMeters)
        #logger.info (self.className + " ---> aircraft fly")
        
        altitudeMSLfeet = altitudeMSLmeters * Meter2Feet
        self.setAltitudeMSLfeet(altitudeMSLfeet)
        
        latitudeDegrees = currentPosition.getLatitudeDegrees()
        #latitudeRadians = math.radians(latitudeDegrees)
        
        gravityCenterMetersPerSquaredSeconds = self.earth.gravityWelmec( heightMSLmeters = altitudeMSLmeters, latitudeDegrees = latitudeDegrees )
        #logger.info( self.className + " --- gravity at a given altitude = {0:.2f} m/s2 at altitude = {1:.2f} meters".format( gravityCenterMetersPerSquaredSeconds , altitudeMSLmeters ))

        tasKnots = self.getCurrentTASspeedKnots()
        aircraftMassKilograms = self.getCurrentMassKilograms()

        if ( elapsedTimeSeconds < deltaTimeSeconds ):
            pass
            ''' display this only once'''
            #logger.info( self.className + " -----------------------------")
            #logger.info( self.className + " ----- start flying ----------")
            #logger.info( self.className + " -----------------------------")
            #self.initStateVector(elapsedTimeSeconds , self.getCurrentConfiguration() , 0.0 , 0.0 , altitudeMSLmeters , aircraftMassKilograms)
        else:
            pass
            #logger.info( self.className + " ---------------------------------------------------")
            #logger.info( self.className + " ----- {0} - elapsed time {1:.0f} seconds----".format ( self.aircraftCurrentConfiguration , elapsedTimeSeconds ) )
            #logger.info( self.className + " ---------------------------------------------------")
            
        
        if self.isTakeOff():
            
            #logger.info(self.className + " --> TakeOff phase")
            
            rateOfClimbFeetMinutes = 0.0
            rateOfClimbMetersSeconds = rateOfClimbFeetMinutes * FeetMinutes2MetersSeconds
            
            thrustNewtons = self.computeThrustNewtons( tasKnots , altitudeMSLfeet , rateOfClimbFeetMinutes)
            dragNewtons = self.computeDragNewtons ( aircraftMassKilograms , tasKnots , altitudeMSLfeet  )

            ''' as openap wrap has not a correct drag computation during ground run / takeoff '''
            ''' use the mean acceleration '''
            
            aircraftAccelerationMetersSecondSquare = self.getDefaultTakeOffAccelerationMetersSecondsSquare()
            #logger.info( self.className + " - aircraft acceleration = {0:.2f} meters per seconds square".format( aircraftAccelerationMetersSecondSquare ))

            trueAirSpeedMetersSecond = self.getCurrentTASmetersSeconds()
            trueAirSpeedMetersSecond = trueAirSpeedMetersSecond + ( aircraftAccelerationMetersSecondSquare * deltaTimeSeconds )
            
            #logger.info( self.className + " - TAS = {0:.2f} meters per second - TAS = {1:.2f} knots ".format( trueAirSpeedMetersSecond , trueAirSpeedMetersSecond * MeterSecond2Knots))
            self.setCurrentTASmetersSeconds(trueAirSpeedMetersSecond , altitudeMSLfeet)
            
            ''' distance flown '''
            flightPathAngleDegrees = self.computeFlightPathAngleDegrees( rateOfClimbMetersSeconds , trueAirSpeedMetersSecond )
            deltaDistanceFlownMeters = trueAirSpeedMetersSecond * math.cos(math.radians(flightPathAngleDegrees)) * deltaTimeSeconds
            totalDistanceFlownMeters = totalDistanceFlownMeters + deltaDistanceFlownMeters
            self.setTotalDistanceFlownMeters(totalDistanceFlownMeters)
            #logger.info( self.className + " - distance flown = {0:.2f} meters - distance flown = {1:.2f} Nautical miles ".format( totalDistanceFlownMeters , totalDistanceFlownMeters * Meter2NauticalMiles ))
    
            ''' mass loss due to fuel flow '''
            fuelFlowKilogramsSeconds = self.computeFuelFlowKilogramsSeconds(trueAirSpeedMetersSecond * MeterSecond2Knots , altitudeMSLfeet , aircraftMassKilograms , rateOfClimbFeetMinutes)
            aircraftMassKilograms = aircraftMassKilograms - ( fuelFlowKilogramsSeconds * deltaTimeSeconds )
            self.setAircraftMassKilograms(aircraftMassKilograms)
            
            casKnots = tas2cas( tas = trueAirSpeedMetersSecond, altitude = altitudeMSLfeet , temp = 'std' , speed_units = 'm/s', alt_units = 'ft') * MeterSecond2Knots
            #logger.info( self.className + " - CAS = {0:.2f} meters per second - CAS = {1:.2f} knots ".format( casKnots * Knots2MetersSeconds , casKnots ) )
    
            if ( casKnots >= self.getDefaultTakeOffCASknots() ):
                logger.info ( self.className + ' - CAS= {0:.2f} knots >= takeoff Stall Speed= {1:.2f} knots'.format(casKnots, self.getDefaultTakeOffCASknots()) )
                self.setInitialClimbConfiguration(elapsedTimeSeconds + deltaTimeSeconds)
            
        
        elif self.isInitialClimb():
            ''' aircraft is airborne and landing gear is retracted '''
            
            rateOfClimbMetersSeconds = self.getInitialClimbVerticalRateMeterSeconds()
            rateOfClimbFeetMinutes = rateOfClimbMetersSeconds * MeterSeconds2FeetMinutes
            
            thrustNewtons = self.computeThrustNewtons( tasKnots , altitudeMSLfeet , rateOfClimbFeetMinutes)
            dragNewtons = self.computeDragNewtons ( aircraftMassKilograms , tasKnots , altitudeMSLfeet  )
            
            trueAirSpeedMetersSecond = self.getCurrentTASmetersSeconds()
            #liftNewtons = self.computeLiftNewtons( aircraftMassKilograms = aircraftMassKilograms, altitudeMeanSeaLevelMeters =  altitudeMSLmeters, trueAirSpeedMetersSecond = trueAirSpeedMetersSecond , latitudeDegrees=latitudeDegrees)
            
            ''' compute new True Air Speed '''
            aircraftAccelerationMetersSecondSquare = ((thrustNewtons - dragNewtons) / aircraftMassKilograms) - ((gravityCenterMetersPerSquaredSeconds * rateOfClimbMetersSeconds )/ trueAirSpeedMetersSecond ) 
            trueAirSpeedMetersSecond = trueAirSpeedMetersSecond + aircraftAccelerationMetersSecondSquare * deltaTimeSeconds
            
            #logger.info( self.className + " - TAS = {0:.2f} meters per second - TAS = {1:.2f} knots ".format( trueAirSpeedMetersSecond , trueAirSpeedMetersSecond * MeterSecond2Knots))
            self.setCurrentTASmetersSeconds(trueAirSpeedMetersSecond , altitudeMSLfeet)
             
            ''' distance flown '''
            flightPathAngleDegrees = self.computeFlightPathAngleDegrees( rateOfClimbMetersSeconds , trueAirSpeedMetersSecond )
            deltaDistanceFlownMeters = trueAirSpeedMetersSecond * math.cos(math.radians(flightPathAngleDegrees)) * deltaTimeSeconds
            totalDistanceFlownMeters = totalDistanceFlownMeters + deltaDistanceFlownMeters
            
            self.setTotalDistanceFlownMeters(totalDistanceFlownMeters)
            #logger.info( self.className + " - distance flown = {0:.2f} meters - distance flown = {1:.2f} Nautical miles ".format( totalDistanceFlownMeters , totalDistanceFlownMeters * Meter2NauticalMiles ))

            ''' mass loss due to fuel flow '''
            fuelFlowKilogramsSeconds = self.computeFuelFlowKilogramsSeconds(TASknots         = trueAirSpeedMetersSecond * MeterSecond2Knots , 
                                                                     aircraftAltitudeMSLfeet = altitudeMSLfeet , 
                                                                     aircraftMassKilograms   = aircraftMassKilograms , 
                                                                     verticalRateFeetMinutes  = rateOfClimbFeetMinutes,
                                                                     accelerationMetersSecondsSquare = aircraftAccelerationMetersSecondSquare)
            aircraftMassKilograms = aircraftMassKilograms - ( fuelFlowKilogramsSeconds * deltaTimeSeconds )
            self.setAircraftMassKilograms(aircraftMassKilograms)
                
            ''' transition to  climb as soon as height above ground is 35 feet above ground '''
            deltaAltitudeMeters = rateOfClimbMetersSeconds * deltaTimeSeconds
            altitudeMSLmeters = altitudeMSLmeters + deltaAltitudeMeters
            #logger.info( self.className + " - departure runway MSL altitude = {0:.2f} meters - aircraft altitude MSL = {1:.2f} meters ".format( self.getDepartureRunwayMSLmeters() , altitudeMSLmeters ))
            
            AboveGroundMeters = 35.0 * feet2Meters
            ''' From the application of takeoff power, through rotation and to an altitude of 35 feet above runway elevation.  '''
            if ( altitudeMSLmeters > self.getDepartureRunwayMSLmeters() + AboveGroundMeters):
                logging.info( self.className + " -> 35 feet above ground" )
                self.setClimbConfiguration(elapsedTimeSeconds + deltaTimeSeconds)


        elif self.isClimb():
            ''' We consider the climbing section up to 1500 ft to be the initial climb. '''
            ''' The climb segment starts when the aircraft reaches clean configuration ''' 
            ''' and lasts until the moment when it reaches cruise altitude '''
            
            rateOfClimbMetersSeconds = self.getClimbVerticalRateMeterSeconds( altitudeMSLfeet )
            rateOfClimbFeetMinutes = rateOfClimbMetersSeconds * MeterSeconds2FeetMinutes

            rateOfClimbFeetMinutes = self.computeRateOfClimbFeetMinutes ( rateOfClimbFeetMinutes , altitudeMSLfeet )
            rateOfClimbMetersSeconds = rateOfClimbFeetMinutes * FeetMinutes2MetersSeconds
            
            #rateOfClimbMetersSeconds = self.computeROCD(deltaTimeSeconds, thrustNewtons, dragNewtons, trueAirSpeedMetersSecond, aircraftMassKilograms, gravityCenterMetersPerSquaredSeconds)
            
            thrustNewtons = self.computeThrustNewtons( tasKnots , altitudeMSLfeet , rateOfClimbFeetMinutes)
            dragNewtons = self.computeDragNewtons ( aircraftMassKilograms , tasKnots , altitudeMSLfeet , rateOfClimbFeetMinutes )
            
            trueAirSpeedMetersSecond = self.getCurrentTASmetersSeconds()
            #liftNewtons = self.computeLiftNewtons( aircraftMassKilograms = aircraftMassKilograms, altitudeMeanSeaLevelMeters =  altitudeMSLmeters, trueAirSpeedMetersSecond = trueAirSpeedMetersSecond , latitudeDegrees=latitudeDegrees)
            
            ''' compute new True Air Speed '''
            ''' dVTAS/dt = ( T - D ) / m - ( ( g0 * dh /dt ) / VTas ) '''
            aircraftAccelerationMetersSecondSquare = ((thrustNewtons - dragNewtons) / aircraftMassKilograms) - ((gravityCenterMetersPerSquaredSeconds * rateOfClimbMetersSeconds )/ trueAirSpeedMetersSecond ) 
            #trueAirSpeedMetersSecond = trueAirSpeedMetersSecond + aircraftAccelerationMetersSecondSquare * deltaTimeSeconds
            
            climbCASknots = self.computeClimbCASknots( altitudeMSLfeet = altitudeMSLfeet ,
                                                       CASknots        = tas2cas ( 
                                                                            tas         = self.getCurrentTASspeedKnots() ,
                                                                            altitude    = altitudeMSLfeet ,
                                                                            temp        = 'std' ,
                                                                            speed_units = 'kt' , 
                                                                            alt_units   = 'ft' , 
                                                                            temp_units  = 'C' ) )
            
            trueAirSpeedKnots = cas2tas( cas         = climbCASknots ,
                                         altitude    = altitudeMSLfeet ,
                                         temp        = 'std' ,
                                         speed_units = 'kt' ,
                                         alt_units   = 'ft' ,
                                         temp_units  = 'C' )
            
            trueAirSpeedMetersSecond = trueAirSpeedKnots * Knots2MetersSeconds
            #logger.info( self.className + " - TAS = {0:.2f} meters per second - TAS = {1:.2f} knots ".format( trueAirSpeedMetersSecond , trueAirSpeedMetersSecond * MeterSecond2Knots))
            self.setCurrentTASmetersSeconds(trueAirSpeedMetersSecond , altitudeMSLfeet)
            ''' verify that speeds limits have been met '''
            trueAirSpeedMetersSecond = self.getCurrentTASmetersSeconds()

             
            ''' distance flown '''
            flightPathAngleDegrees = self.computeFlightPathAngleDegrees( rateOfClimbMetersSeconds , trueAirSpeedMetersSecond )
            deltaDistanceFlownMeters = trueAirSpeedMetersSecond * math.cos(math.radians(flightPathAngleDegrees)) * deltaTimeSeconds
            totalDistanceFlownMeters = totalDistanceFlownMeters + deltaDistanceFlownMeters
            self.setTotalDistanceFlownMeters(totalDistanceFlownMeters)
            #logger.info( self.className + " - distance flown = {0:.2f} meters - distance flown = {1:.2f} Nautical miles ".format( totalDistanceFlownMeters , totalDistanceFlownMeters * Meter2NauticalMiles ))

            ''' mass loss due to fuel flow '''
            fuelFlowKilogramsSeconds = self.computeFuelFlowKilogramsSeconds(TASknots                 =trueAirSpeedMetersSecond * MeterSecond2Knots , 
                                                                     aircraftAltitudeMSLfeet         =altitudeMSLfeet , 
                                                                     aircraftMassKilograms           =aircraftMassKilograms , 
                                                                     verticalRateFeetMinutes         =rateOfClimbFeetMinutes,
                                                                     accelerationMetersSecondsSquare =aircraftAccelerationMetersSecondSquare)
            aircraftMassKilograms = aircraftMassKilograms - ( fuelFlowKilogramsSeconds * deltaTimeSeconds )
            self.setAircraftMassKilograms(aircraftMassKilograms)

            ''' transition to initial climb as soon as height above ground is 1000 feet / 300 meters '''
            deltaAltitudeMeters = rateOfClimbMetersSeconds * deltaTimeSeconds
            altitudeMSLmeters = altitudeMSLmeters + deltaAltitudeMeters
            
            #logging.info( self.className + " - distance still to fly = {0:.2f} meters".format( distanceStillToFlyMeters ))
            #logging.info( self.className + " - distance still to fly = {0:.2f} Nm".format( distanceStillToFlyMeters * Meter2NauticalMiles ))
            
            #logger.info( self.className + " - cruise level feet = {0:.2f} feet ".format ( self.getCruiseLevelFeet() ) )
            if ( ( altitudeMSLmeters * Meter2Feet ) > ( self.getCruiseLevelFeet() - 10.0 ) ) and  \
                     ( ( altitudeMSLmeters * Meter2Feet )  < ( self.getCruiseLevelFeet() + 100.0 ) ):
                self.setCruiseConfiguration( elapsedTimeSeconds + deltaTimeSeconds )

        elif self.isCruise():
            '''The final approach starts from 1000 ft toward the end of the descent '''
            
            rateOfClimbFeetMinutes = 0.0
            rateOfClimbMetersSeconds = 0.0
            
            thrustNewtons = self.computeThrustNewtons( tasKnots , altitudeMSLfeet , rateOfClimbFeetMinutes)
            dragNewtons = self.computeDragNewtons ( aircraftMassKilograms , tasKnots , altitudeMSLfeet , rateOfClimbFeetMinutes )
            ''' @TODO temporary apply cruise speed '''
            cruiseTASknots = self.computeCruiseTASknots()
            trueAirSpeedMetersSecond = cruiseTASknots * Knots2MetersSeconds
            #liftNewtons = self.computeLiftNewtons( aircraftMassKilograms = aircraftMassKilograms, altitudeMeanSeaLevelMeters =  altitudeMSLmeters, trueAirSpeedMetersSecond = trueAirSpeedMetersSecond , latitudeDegrees=latitudeDegrees)
            
            ''' compute new True Air Speed '''
            ''' dVTAS/dt = ( T - D ) / m - ( ( g0 * dh /dt ) / VTas ) '''
            aircraftAccelerationMetersSecondSquare = ((thrustNewtons - dragNewtons) / aircraftMassKilograms) - ((gravityCenterMetersPerSquaredSeconds * rateOfClimbMetersSeconds )/ trueAirSpeedMetersSecond ) 
            #if ( aircraftAccelerationMetersSecondSquare > 0.0 ):
            #    trueAirSpeedMetersSecond = trueAirSpeedMetersSecond + aircraftAccelerationMetersSecondSquare * deltaTimeSeconds
                #logging.info( self.className + " - TAS = {0:.2f} m/s".format( trueAirSpeedMetersSecond ))
            
            #logger.info( self.className + " - TAS = {0:.2f} meters per second - TAS = {1:.2f} knots ".format( trueAirSpeedMetersSecond , trueAirSpeedMetersSecond * MeterSecond2Knots))
            self.setCurrentTASmetersSeconds(trueAirSpeedMetersSecond , altitudeMSLfeet)
             
            ''' distance flown '''
            flightPathAngleDegrees = self.computeFlightPathAngleDegrees( rateOfClimbMetersSeconds , trueAirSpeedMetersSecond )
            deltaDistanceFlownMeters = trueAirSpeedMetersSecond * math.cos(math.radians(flightPathAngleDegrees)) * deltaTimeSeconds
            totalDistanceFlownMeters = totalDistanceFlownMeters + deltaDistanceFlownMeters
            self.setTotalDistanceFlownMeters(totalDistanceFlownMeters)
            
            #logger.info( self.className + " - distance flown = {0:.2f} meters - distance flown = {1:.2f} Nautical miles ".format( totalDistanceFlownMeters , totalDistanceFlownMeters * Meter2NauticalMiles ))

            ''' mass loss due to fuel flow '''
            fuelFlowKilogramsSeconds = self.computeFuelFlowKilogramsSeconds(TASknots         = trueAirSpeedMetersSecond * MeterSecond2Knots , 
                                                                     aircraftAltitudeMSLfeet = altitudeMSLfeet , 
                                                                     aircraftMassKilograms   = aircraftMassKilograms , 
                                                                     verticalRateFeetMinutes  = rateOfClimbFeetMinutes,
                                                                     accelerationMetersSecondsSquare=aircraftAccelerationMetersSecondSquare)
            aircraftMassKilograms = aircraftMassKilograms - ( fuelFlowKilogramsSeconds * deltaTimeSeconds )
            self.setAircraftMassKilograms(aircraftMassKilograms)
            
            rateOfDescentMetersSeconds = self.getDescentVerticalRateMeterSeconds( altitudeMSLfeet )
            rateOfDescentFeetMinutes   = rateOfDescentMetersSeconds * MeterSeconds2FeetMinutes

            #===================================================================
            #logging.info( self.className + " - rate of descent feet per minutes = {0:.2f}".format( rateOfDescentFeetMinutes ) )
            #logging.info( self.className + " - distance still to fly = {0:.2f} meters".format( distanceStillToFlyMeters ))
            #logging.info( self.className + " - distance still to fly = {0:.2f} Nm".format( distanceStillToFlyMeters * Meter2NauticalMiles ))
            
            descentRangeMeters = self.getDescentRangeMeters()
            #===================================================================0:
            
            ''' altitude does not change , or increases slightly due to lighter aircraft '''
            if ( distanceStillToFlyMeters <=  descentRangeMeters ):
                self.setDescentConfiguration( elapsedTimeSeconds )
                
            
        elif self.isDescent():
            
            trueAirSpeedMetersSecond = self.getCurrentTASmetersSeconds()

            ''' compute descent rate from vertival and lateral distance '''
            rateOfDescentMetersSeconds = self.getDescentVerticalRateMeterSeconds( altitudeMSLfeet )
            rateOfDescentFeetMinutes   = rateOfDescentMetersSeconds * MeterSeconds2FeetMinutes
            #logging.info( self.className + " - rate of descent from wrap database = {0:.2f} feet per minutes ".format( rateOfDescentFeetMinutes ))
            
            deltaDistanceMeters = trueAirSpeedMetersSecond * deltaTimeSeconds
            rateOfDescentMetersSeconds = self.computeDescentVerticalRateMeterSeconds ( deltaTimeSeconds , currentPosition , distanceToLastFixMeters , deltaDistanceMeters )
            rateOfDescentFeetMinutes   = rateOfDescentMetersSeconds * MeterSeconds2FeetMinutes
            #logging.info( self.className + " - rate of descent computed = {0:.2f} feet per minutes ".format( rateOfDescentFeetMinutes ))

            thrustNewtons = self.computeThrustNewtons( tasKnots , altitudeMSLfeet , rateOfDescentFeetMinutes)
            dragNewtons = self.computeDragNewtons ( aircraftMassKilograms , tasKnots , altitudeMSLfeet , rateOfDescentFeetMinutes )

            #liftNewtons = self.computeLiftNewtons( aircraftMassKilograms      = aircraftMassKilograms, 
            #                                       altitudeMeanSeaLevelMeters = altitudeMSLmeters, 
            #                                       trueAirSpeedMetersSecond   = trueAirSpeedMetersSecond , 
            #                                       latitudeDegrees            = latitudeDegrees)
            
            ''' compute new True Air Speed '''
            ''' dVTAS/dt = ( T - D ) / m - ( ( g0 * dh /dt ) / VTas ) '''
            aircraftAccelerationMetersSecondSquare = ((thrustNewtons - dragNewtons) / aircraftMassKilograms) - ((gravityCenterMetersPerSquaredSeconds * rateOfDescentMetersSeconds )/ trueAirSpeedMetersSecond ) 
            #trueAirSpeedMetersSecond = trueAirSpeedMetersSecond + aircraftAccelerationMetersSecondSquare * deltaTimeSeconds
            
            descentCASknots = self.computeDescentCASknots( altitudeMSLfeet = altitudeMSLfeet ,
                                                           CASknots        = tas2cas ( 
                                                            tas            = self.getCurrentTASspeedKnots() ,
                                                            altitude       = altitudeMSLfeet ,
                                                            temp        = 'std' ,
                                                            speed_units = 'kt' , 
                                                            alt_units   = 'ft' , 
                                                            temp_units  = 'C' ) )
            trueAirSpeedKnots = cas2tas( cas         = descentCASknots ,
                                         altitude    = altitudeMSLfeet ,
                                         temp        = 'std' ,
                                         speed_units = 'kt' ,
                                         alt_units   = 'ft' ,
                                         temp_units  = 'C' )
            
            trueAirSpeedMetersSecond = trueAirSpeedKnots * Knots2MetersSeconds
            #logger.info( self.className + " - TAS = {0:.2f} meters per second - TAS = {1:.2f} knots ".format( trueAirSpeedMetersSecond , trueAirSpeedMetersSecond * MeterSecond2Knots))
            self.setCurrentTASmetersSeconds(trueAirSpeedMetersSecond , altitudeMSLfeet)
             
            ''' flight path angle  '''
            flightPathAngleDegrees = self.computeFlightPathAngleDegrees( rateOfDescentMetersSeconds , trueAirSpeedMetersSecond )
            
            ''' distance flown '''
            deltaDistanceFlownMeters = trueAirSpeedMetersSecond * math.cos(math.radians(flightPathAngleDegrees)) * deltaTimeSeconds
            totalDistanceFlownMeters = totalDistanceFlownMeters + deltaDistanceFlownMeters
            self.setTotalDistanceFlownMeters(totalDistanceFlownMeters)
            #logger.info( self.className + " - distance flown = {0:.2f} meters - distance flown = {1:.2f} Nautical miles ".format( totalDistanceFlownMeters , totalDistanceFlownMeters * Meter2NauticalMiles ))

            ''' mass loss due to fuel flow '''
            fuelFlowKilogramsSeconds = self.computeFuelFlowDescentKilogramsSeconds(descentIdleThrustNewtons = thrustNewtons , 
                                                                                   aircraftAltitudeMSLfeet = altitudeMSLfeet )
                                                                   
            aircraftMassKilograms = aircraftMassKilograms - ( fuelFlowKilogramsSeconds * deltaTimeSeconds )
            self.setAircraftMassKilograms(aircraftMassKilograms)
            
            ''' transition to final approach as soon as altitude is below last turn followed by the descent glide slope to the arrival runway  '''
            deltaAltitudeMeters = rateOfDescentMetersSeconds * deltaTimeSeconds
            altitudeMSLmeters = altitudeMSLmeters + deltaAltitudeMeters
            
            if ( descentCASknots < self.getFinalApproachCASknots() ):
                pass
                #self.setFinalApproachConfiguration( elapsedTimeSeconds )
            #logging.info ( self.className + " - target approach waypoint = {0} meters".format( str( self.getTargetApproachWayPoint() ) ))
            
            if ( altitudeMSLmeters < self.getTargetApproachWayPoint().getAltitudeMeanSeaLevelMeters() ):
                ''' target approach is the top of the last turn before the descent glide slope to the runway '''
                logging.info( self.className + ' - current aircraft altitude = {0:.2f} meters'.format ( altitudeMSLmeters ))
                logging.info( self.className + ' - approach last fix altitude = {0:.2f} meters'.format ( self.getTargetApproachWayPoint().getAltitudeMeanSeaLevelMeters() ))
                self.setFinalApproachConfiguration( elapsedTimeSeconds )
                
        
        elif self.isApproach(): 
            
            #logging.info( self.className + " - aircraft altitude MSL = {0:.2f} meters".format( altitudeMSLmeters ))
            #logging.info( self.className + " - total distant flown = {0:.2f} meters - {1:.2f} Nm ".format ( totalDistanceFlownMeters , totalDistanceFlownMeters * Meter2NauticalMiles))
            
            ''' approach starts on top of the last turn and descent glide slope '''
            rateOfDescentMetersSeconds = self.getFinalApproachVerticalRateMeterSeconds( altitudeMSLfeet )
            rateOfDescentFeetMinutes   = rateOfDescentMetersSeconds * MeterSeconds2FeetMinutes
            
            thrustNewtons = self.computeThrustNewtons( tasKnots , altitudeMSLfeet , rateOfDescentFeetMinutes)
            dragNewtons = self.computeDragNewtons ( aircraftMassKilograms , tasKnots , altitudeMSLfeet , rateOfDescentFeetMinutes )

            trueAirSpeedMetersSecond = self.getCurrentTASmetersSeconds( )
            currentCASknots = tas2cas(tas      = trueAirSpeedMetersSecond, 
                                      altitude = altitudeMSLfeet ,
                                      temp     = 'std', 
                                      speed_units = 'm/s', 
                                      alt_units   = 'ft', 
                                      temp_units  = 'C')
            #liftNewtons = self.computeLiftNewtons( aircraftMassKilograms      = aircraftMassKilograms, 
            #                                       altitudeMeanSeaLevelMeters = altitudeMSLmeters, 
            #                                       trueAirSpeedMetersSecond   = trueAirSpeedMetersSecond , 
            #                                       latitudeDegrees            = latitudeDegrees)
            
            aircraftAccelerationMetersSecondSquare = ((thrustNewtons - dragNewtons) / aircraftMassKilograms) - ((gravityCenterMetersPerSquaredSeconds * rateOfDescentMetersSeconds )/ trueAirSpeedMetersSecond ) 

            approachCASknots = self.computeApproachCASknots( altitudeMSLfeet = altitudeMSLfeet,
                                                             currentCASknots = currentCASknots,
                                                             arrivalRunwayAltitudeMSLfeet = self.arrivalRunwayMSLmeters * Meter2Feet)
            trueAirSpeedKnots = cas2tas( cas         = approachCASknots ,
                                         altitude    = altitudeMSLfeet ,
                                         temp        = 'std' ,
                                         speed_units = 'kt' ,
                                         alt_units   = 'ft' ,
                                         temp_units  = 'C' )
            
            trueAirSpeedMetersSecond = trueAirSpeedKnots * Knots2MetersSeconds
            #logger.info( self.className + " - TAS = {0:.2f} m/s - TAS = {1:.2f} kt ".format( trueAirSpeedMetersSecond , trueAirSpeedMetersSecond * MeterSecond2Knots))
            self.setCurrentTASmetersSeconds(trueAirSpeedMetersSecond , altitudeMSLfeet)
             
            ''' flight path angle  '''
            #flightPathAngleDegrees = self.computeFlightPathAngleDegrees( rateOfDescentMetersSeconds , trueAirSpeedMetersSecond )
            flightPathAngleDegrees = self.computeApproachFlightPathAngleDegrees( currentPosition )
            #logging.info ( self.className + " - approach descent flight path angle = {0:.2f} degrees". format ( flightPathAngleDegrees ))
            
            ''' distance flown '''
            deltaDistanceFlownMeters = trueAirSpeedMetersSecond * math.cos(math.radians(flightPathAngleDegrees)) * deltaTimeSeconds
            totalDistanceFlownMeters = totalDistanceFlownMeters + deltaDistanceFlownMeters
            self.setTotalDistanceFlownMeters(totalDistanceFlownMeters)
            #logger.info( self.className + " - distance flown = {0:.2f} meters - distance flown = {1:.2f} Nautical miles ".format( totalDistanceFlownMeters , totalDistanceFlownMeters * Meter2NauticalMiles ))

            ''' mass loss due to fuel flow '''
            fuelFlowKilogramsSeconds = self.computeFuelFlowKilogramsSeconds(TASknots          = trueAirSpeedMetersSecond * MeterSecond2Knots , 
                                                                     aircraftAltitudeMSLfeet  = altitudeMSLfeet , 
                                                                     aircraftMassKilograms    = aircraftMassKilograms , 
                                                                     verticalRateFeetMinutes  = rateOfDescentMetersSeconds,
                                                                     accelerationMetersSecondsSquare = aircraftAccelerationMetersSecondSquare)
            aircraftMassKilograms = aircraftMassKilograms - ( fuelFlowKilogramsSeconds * deltaTimeSeconds )
            self.setAircraftMassKilograms(aircraftMassKilograms)
            
            ''' transition to landing as soon as landing stall speed is reached '''
            #deltaAltitudeMeters = rateOfDescentMetersSeconds * deltaTimeSeconds
            
            deltaAltitudeMeters = deltaDistanceFlownMeters * math.tan( math.radians ( flightPathAngleDegrees ) )
            #logging.info( self.className + " - delta altitude = {0:.2f} meters".format(deltaAltitudeMeters))
            
            altitudeMSLmeters = altitudeMSLmeters - deltaAltitudeMeters
            #logging.info( self.className + " - current altitude = {0:.2f} meters - {1:.2f} feet".format( altitudeMSLmeters , altitudeMSLmeters * Meter2Feet))
            
            ''' compute stall speed to change configuration to landing '''
            LandingStallSpeedCASKnots = self.computeLandingStallSpeedCasKnots()
            
            ''' move to Landing as soon as Stall CAS reached '''
            if ( ( tas2cas(tas = trueAirSpeedMetersSecond , altitude = altitudeMSLmeters, temp = 'std',
                     speed_units = 'm/s', alt_units = 'm') * MeterSecond2Knots ) <= LandingStallSpeedCASKnots):
                ''' as soon as speed decrease to landing configuration => change aircraft configuration '''
                #logger.debug ( self.className +' distance to approach fix= {0:.2f} meters - delta altitude to approach fix= {1:.2f} meters'.format(distanceToTargetApproachFixMeters, deltaAltitudeToTargetApproachFixMeters) )
                self.setLandingConfiguration(elapsedTimeSeconds + deltaTimeSeconds )
                
            ''' move to landing as soon as arrival runway altitude is reached '''
            if ( altitudeMSLmeters <= self.arrivalRunwayMSLmeters ):
                logging.info( self.className + " - aircraft flying at the runway altitude ")
                self.setLandingConfiguration(elapsedTimeSeconds + deltaTimeSeconds )
            
        elif self.isLanding():
            
            #logging.info( self.className + " - total distant flown = {0:.2f} meters - {1:.2f} Nm ".format ( totalDistanceFlownMeters , totalDistanceFlownMeters * Meter2NauticalMiles))
            #logging.info( self.className + " - aircraft altitude MSL = {0:.2f} meters".format( altitudeMSLmeters ))
            
            flightPathAngleDegrees = 0.0
            #logging.info( self.className + " - touch down CAS = {0:.2f} knots".format( self.landingCASknots ))
            
            #logging.info( self.className + " - current altitude = {0:.2f} meters - {1:.2f} feet".format( altitudeMSLfeet * feet2Meters , altitudeMSLfeet))
            
            trueAirSpeedMetersSecond = self.getCurrentTASmetersSeconds( )
            currentCASknots = tas2cas(tas      = trueAirSpeedMetersSecond, 
                                      altitude = altitudeMSLfeet ,
                                      temp     = 'std', 
                                      speed_units = 'm/s', 
                                      alt_units   = 'ft', 
                                      temp_units  = 'C')
            #logging.info( self.className + " - current CAS = {0:.2f} knots".format( currentCASknots ))
            
            thrustNewtons = self.computeThrustNewtons( currentCASknots , altitudeMSLfeet , 0.0 )
            dragNewtons = self.computeDragNewtons ( aircraftMassKilograms , currentCASknots , altitudeMSLfeet  )
            
            aircraftDecelerationMetersSecondSquare = self.getDefaultLandingDecelerationMetersSecondsSquare()
            #logging.info( self.className + " - aircraft default landing deceleration = {0:.2f} meters per seconds square".format( aircraftDecelerationMetersSecondSquare ))

            trueAirSpeedMetersSecond = trueAirSpeedMetersSecond + ( aircraftDecelerationMetersSecondSquare * deltaTimeSeconds )
            self.setCurrentTASmetersSeconds(trueAirSpeedMetersSecond , altitudeMSLfeet)
            
            ''' distance flown '''
            deltaDistanceFlownMeters = trueAirSpeedMetersSecond * deltaTimeSeconds
            totalDistanceFlownMeters = totalDistanceFlownMeters + deltaDistanceFlownMeters
            self.setTotalDistanceFlownMeters(totalDistanceFlownMeters)
            #logging.info( self.className + " - distance flown = {0:.2f} meters - distance flown = {1:.2f} Nm ".format( totalDistanceFlownMeters , totalDistanceFlownMeters * Meter2NauticalMiles ))

            ''' mass loss due to fuel flow '''
            fuelFlowKilogramsSeconds = self.computeFuelFlowKilogramsSeconds(TASknots          = trueAirSpeedMetersSecond * MeterSecond2Knots , 
                                                                     aircraftAltitudeMSLfeet  = altitudeMSLfeet , 
                                                                     aircraftMassKilograms    = aircraftMassKilograms , 
                                                                     verticalRateFeetMinutes  = 0.0,
                                                                     accelerationMetersSecondsSquare = aircraftDecelerationMetersSecondSquare)
            #logging.info( self.className + " - fuel flow = {0:.2f} kg/s".format( fuelFlowKilogramsSeconds ))
            aircraftMassKilograms = aircraftMassKilograms - ( fuelFlowKilogramsSeconds * deltaTimeSeconds )
            self.setAircraftMassKilograms(aircraftMassKilograms)
            
            if currentCASknots < ConstantTaxiSpeedCasKnots:
                logging.info( self.className + " - taxi speed reached - end of simulation")
                logging.info ( self.className + " - elapsed time = {0:.0f} seconds ->  {1:.2f} hours ".format(elapsedTimeSeconds , elapsedTimeSeconds/3600.0))

                endOfSimulation = True

        
        self.setAltitudeMSLmeters(altitudeMSLmeters)
        elapsedTimeSeconds = elapsedTimeSeconds + deltaTimeSeconds 
        #logging.info(self.className + " - update state vector")
        self.updateAircraftStateVector( elapsedTimeSeconds       = elapsedTimeSeconds , 
                                        flightPhase              = self.getCurrentConfiguration() , 
                                        flightPathAngleDegrees   = flightPathAngleDegrees , 
                                        trueAirSpeedMeterSecond  = trueAirSpeedMetersSecond , 
                                        altitudeMSLmeters        = altitudeMSLmeters ,
                                        totalDistanceFlownMeters = totalDistanceFlownMeters , 
                                        distanceStillToFlyMeters = 0.0 , 
                                        aircraftMassKilograms    = aircraftMassKilograms , 
                                        thrustNewtons            = thrustNewtons , 
                                        dragNewtons              = dragNewtons)
        #logging.info( self.className + " - state vector updated")
        return endOfSimulation , deltaDistanceFlownMeters , altitudeMSLmeters
        