# -*- coding: UTF-8 -*-
'''
Created on 31 December 2014

@author: PASTOR Robert

        Written By:
                Robert PASTOR 
                @Email: < robert [--DOT--] pastor0691 (--AT--) gmail [--DOT--] com >

        @http://trajectoire-predict.monsite-orange.fr/ 
        @copyright: Copyright 2015 Robert PASTOR 

        This program is free software; you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation; either version 3 of the License, or
        (at your option) any later version.
 
        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.
 
        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.

Manage the ground run phase
'''
import math
import logging
from trajectory.aerocalc.airspeed import tas2cas

from trajectory.Environment.Runways.RunWayFile import RunWay
from trajectory.Guidance.WayPointFile import Airport, WayPoint

from trajectory.Guidance.GraphFile import Graph
from trajectory.Openap.AircraftMainFile import OpenapAircraft

from trajectory.Environment.Constants import  MeterPerSecond2Knots 
from trajectory.Environment.Utils import logElapsedRealTime

class GroundRunLeg(Graph):
    '''
    ground run inputs are:
    1) airport field elevation above sea level (meters)
    2) runway true heading (in degrees)
    
    departure Ground Run :
    1) initial speed is 0.0 meters / second
    
    arrival Ground Run:
    1) final speed = taxi speed
    '''
    aircraft = None
    airport = None
    runway = None
    elapsedTimeSeconds = 0.0
    totalLegDistanceMeters = 0.0
    
    def __init__(self,
                 runway,
                 aircraft,
                 airport):
            
        ''' base class init '''
        Graph.__init__(self)
        self.className = self.__class__.__name__
        # @TODO - unify RunWay from Envifonment and AirlineRunway
        assert (isinstance(runway, RunWay) and not(runway is None))
        
        self.runway = runway
        logging.debug ( self.className + ': ground run - run-way true heading= ' + str(self.runway.getTrueHeadingDegrees()) + ' degrees' )
        
        assert (isinstance(aircraft, OpenapAircraft) and not(aircraft is None))
        self.aircraft = aircraft
        
        assert (isinstance(airport, Airport)  and not(airport is None))
        self.airport = airport
    
    def computeTouchDownWayPoint(self):
        ''' get landing length in meters '''
        #landingLengthMeters = self.aircraft.getLandingLengthMeters()
        touchDownZoneLengthMeters = self.runway.getLengthMeters() - ( self.runway.getLengthMeters() / 3.0 )

        ''' run-way orientation '''
        runwayTrueHeadingDegrees = self.runway.getTrueHeadingDegrees()
        
        ''' run-way end point '''
        strRunWayEndPointName = self.runway.getName()  + '-' + self.airport.getName() 
        runWayEndPoint = WayPoint (Name              = strRunWayEndPointName, 
                                    LatitudeDegrees  = self.runway.getLatitudeDegrees(),
                                    LongitudeDegrees = self.runway.getLongitudeDegrees(),
                                    AltitudeMeanSeaLevelMeters =self.airport.getFieldElevationAboveSeaLevelMeters())
        
        strTouchDownWayPointName = self.runway.getName() + '-touchDown-' + self.airport.getName() 
        touchDownWayPoint = runWayEndPoint.getWayPointAtDistanceBearing(Name = strTouchDownWayPointName, 
                                                                        DistanceMeters = touchDownZoneLengthMeters, 
                                                                        BearingDegrees = runwayTrueHeadingDegrees)
        touchDownWayPoint.setAltitudeMeanSeaLevelMeters(self.airport.getFieldElevationAboveSeaLevelMeters())
        return touchDownWayPoint
        
    def buildArrivalGroundRun(self,
                              deltaTimeSeconds,
                              elapsedTimeSeconds,
                              initialWayPoint):
        
        #assert isinstance(initialWayPoint, WayPoint)
        ''' 
        speed decreases from 1.2 V Stall to taxi speed
        (according to the airport elevation stall speed changes with air density)
        '''
        ''' delta time in seconds '''
        elapsedTimeSeconds = elapsedTimeSeconds
        
        ''' get landing length in meters '''
        landingLengthMeters = self.aircraft.getLandingLengthMeters()
        ''' run-way orientation '''
        runwayTrueHeadingDegrees = self.runway.getTrueHeadingDegrees()
        runwayTrueHeadingDegrees = math.fmod(runwayTrueHeadingDegrees + 180.0, 360.0)

        ''' run-way end point '''
        strRunWayEndPointName = self.runway.getName() + '-touchDown-'+  self.airport.getName()
        ''' rename the initial way point '''
        initialWayPoint.setName(strRunWayEndPointName)

        ''' graph index '''
        index = 0
        
        distanceStillToFlyMeters = landingLengthMeters
        ''' loop until => end of simulation ( aircraft' speed reduced to the taxi speed = 15 knots) '''
        endOfSimulation = False
        while (endOfSimulation == False) :
            ''' initialisation '''
            if index == 0:
                intermediateWayPoint = initialWayPoint
                
            ''' fly => decrease the true air speed '''
            endOfSimulation, deltaDistanceMeters , altitudeMeters = self.aircraft.fly(  elapsedTimeSeconds        = elapsedTimeSeconds,
                                                                                        deltaTimeSeconds          = deltaTimeSeconds , 
                                                                                        totalDistanceFlownMeters  = self.aircraft.getTotalDistanceFlownMeters(),
                                                                                        altitudeMSLmeters         = self.aircraft.getCurrentAltitudeSeaLevelMeters(),
                                                                                        distanceStillToFlyMeters  = distanceStillToFlyMeters,
                                                                                        currentPosition           = intermediateWayPoint,
                                                                                        distanceToLastFixMeters   = 0.0)
            distanceStillToFlyMeters = distanceStillToFlyMeters - deltaDistanceMeters
            #trueAirSpeedMetersSecond = self.aircraft.getCurrentTrueAirSpeedMetersSecond()
            #logging.debug 'true air speed= ' + str(trueAirSpeedMetersSecond) + ' meters/second'
            
            ''' name of the next point '''
            Name = ''
            if index == 0:
                Name = 'groundRun-{0}'.format( self.runway.getName() )
            #bearingDegrees = math.fmod ( runwayTrueHeadingDegrees + 180.0 , 360.0 )
            bearingDegrees = runwayTrueHeadingDegrees
            newIntermediateWayPoint = intermediateWayPoint.getWayPointAtDistanceBearing(Name = Name, 
                                                                                  DistanceMeters = deltaDistanceMeters, 
                                                                                  BearingDegrees = bearingDegrees)
            ''' during the ground run - altitude = airport field elevation '''
            newIntermediateWayPoint.setAltitudeMeanSeaLevelMeters(self.airport.getFieldElevationAboveSeaLevelMeters())
            
            ''' update route way-point '''
            elapsedTimeSeconds += deltaTimeSeconds
            newIntermediateWayPoint.setElapsedTimeSeconds(elapsedTimeSeconds)
            self.elapsedTimeSeconds = elapsedTimeSeconds
            
            ''' insert in the route '''
            self.addVertex(newIntermediateWayPoint)
            
            ''' copy the intermediate way-point '''
            intermediateWayPoint = newIntermediateWayPoint 
            ''' increment the index '''
            index += 1
  
        #logging.debug ('============ end of arrival ground run ======================')
        logElapsedRealTime ( self.className , elapsedTimeSeconds)
        strRunWayEndPointName = self.runway.getName() + '-' + self.airport.getName() 
        #logging.debug( '{0}: current distance flown = {1:.2f} meters = {2:.2f} Nm'.format ( self.className, self.aircraft.getCurrentDistanceFlownMeters(), self.aircraft.getCurrentDistanceFlownMeters() * Meter2NauticalMiles) )
        intermediateWayPoint.setName(Name = strRunWayEndPointName)
        
        
    def buildDepartureGroundRun(self, 
                                deltaTimeSeconds,
                                elapsedTimeSeconds,
                                distanceStillToFlyMeters,
                                distanceToLastFixMeters):
        
        #logging.info( self.className + " : build departure ground run")
        ''' elapsedTimeSeconds in seconds '''
        # -> @TODO to be suppressed -> elapsedTimeSeconds = elapsedTimeSeconds

        ''' run-way end point '''
        strRunWayEndPointName =  self.runway.getName() + '-' + self.airport.getName()
        runWayEndPoint = WayPoint ( Name             = strRunWayEndPointName, 
                                    LatitudeDegrees  = self.runway.getLatitudeDegrees(),
                                    LongitudeDegrees = self.runway.getLongitudeDegrees(),
                                    AltitudeMeanSeaLevelMeters = self.airport.getFieldElevationAboveSeaLevelMeters())
        
        #logging.info( self.className + " : departure runway = "+ str (runWayEndPoint))
        ''' run-way true heading '''
        runwayTrueHeadingDegrees = self.runway.getTrueHeadingDegrees()
        #logging.info ( self.className + " : runway true heading = " + str(runwayTrueHeadingDegrees) + " degrees")
        
        ''' call base class Graph to build Climb Ramp core of the route '''
        index = 0
        self.addVertex(runWayEndPoint)
        index += 1
        
        ''' departure ground run => initial speed is null '''
        trueAirSpeedMetersSecond = 0.1
        ''' ground run leg distance '''
        self.totalLegDistanceMeters = 0.0
        
        ''' 9th September 2023 - add characteristic point '''
        self.aircraft.initStateVector( elapsedTimeSeconds = elapsedTimeSeconds, 
                                       flightPhase        = self.airport.getName()+"/"+self.runway.getName(),
                                       flightPathAngleDegrees     = 0.0,
                                       trueAirSpeedMetersSecond   = trueAirSpeedMetersSecond,
                                       altitudeMeanSeaLevelMeters = self.airport.getFieldElevationAboveSeaLevelMeters())
        ''' 
        Usually, the lift-off speed is designated to be 1.2 * Vstall 
        at a given weight, an aircraft will rotate and climb, stall or fly at an approach to landing at approx the same CAS.
        regardless of the elevation (height above sea level) , even though the true airspeed and ground-speed may differ significantly.
        These V speeds are normally published as IAS rather than CAS so they can be read directly from the airspeed indicator.
        '''
        VStallSpeedCASKnots = self.aircraft.getDefaultTakeOffCASknots()
        #logging.info ( self.className + ': V stall Calibrated AirSpeed= {0:.2f} knots'.format(VStallSpeedCASKnots) )
        
        ''' loop until 1.2 * Stall CAS speed reached '''
        endOfSimulation = False
        
        ''' initialize the aircraft altitude '''
        self.altitudeMeters = self.airport.getFieldElevationAboveSeaLevelMeters()
        
        while ((endOfSimulation == False) and
               ( tas2cas(tas = trueAirSpeedMetersSecond ,
                         altitude = self.airport.getFieldElevationAboveSeaLevelMeters(),
                         temp = 'std', speed_units = 'm/s', alt_units = 'm') * MeterPerSecond2Knots )  < (1.0 * VStallSpeedCASKnots)):
            ''' initial loop index '''
            #logging.info( self.className + " : flight list index = {0}".format( index ))

            if index == 1:
                #logging.info( self.className + " : flight list index = {0}".format( index ))
                intermediateWayPoint = runWayEndPoint
                
            
            ''' fly => increase in true air speed '''
            ''' during ground run => all the energy is used to increase the Kinetic energy => no potential energy increase '''
            endOfSimulation , deltaDistanceMeters , self.altitudeMeters = self.aircraft.fly(
                                                                     elapsedTimeSeconds       = elapsedTimeSeconds,
                                                                     deltaTimeSeconds         = deltaTimeSeconds, 
                                                                     totalDistanceFlownMeters = self.totalLegDistanceMeters ,
                                                                     altitudeMSLmeters        = self.altitudeMeters,
                                                                     distanceStillToFlyMeters = distanceStillToFlyMeters,
                                                                     currentPosition          = intermediateWayPoint,
                                                                     distanceToLastFixMeters  = distanceToLastFixMeters)
            #logging.info( self.className + " - back from fly step")
            #logging.info( self.className + " - altitude {0} meters".format( self.altitudeMeters ))
            trueAirSpeedMetersSecond = self.aircraft.getCurrentTrueAirSpeedMetersSecond()
            
            assert (((self.airport.getFieldElevationAboveSeaLevelMeters() - 10.0) <= self.altitudeMeters) and
                    ( self.altitudeMeters <= (self.airport.getFieldElevationAboveSeaLevelMeters() + 10.0)))
            #logging.debug self.className + ': delta distance= ' + str(deltaDistanceMeters) + ' meters'
            # name of the next point            
            self.totalLegDistanceMeters += deltaDistanceMeters
            distanceStillToFlyMeters -= deltaDistanceMeters
            distanceToLastFixMeters -= deltaDistanceMeters
            
            Name = ''
            if index == 1:
                Name = 'groundRun-{0}'.format( self.runway.getName() )
                #logging.info( self.className + " - " + Name )
                
            #bearingDegrees = math.fmod ( runwayTrueHeadingDegrees + 180.0 , 360.0 )
            bearingDegrees = runwayTrueHeadingDegrees
            newIntermediateWayPoint = intermediateWayPoint.getWayPointAtDistanceBearing(Name = Name, 
                                                                                  DistanceMeters = deltaDistanceMeters, 
                                                                                  BearingDegrees = bearingDegrees)
            ''' during the ground run - altitude = airport field elevation '''
            newIntermediateWayPoint.setAltitudeMeanSeaLevelMeters(self.airport.getFieldElevationAboveSeaLevelMeters())
            
            ''' update route way-point '''
            elapsedTimeSeconds += deltaTimeSeconds
            newIntermediateWayPoint.setElapsedTimeSeconds(elapsedTimeSeconds)
            self.elapsedTimeSeconds = elapsedTimeSeconds
 
            ''' insert in the as-is computed trajectory '''
            self.addVertex(newIntermediateWayPoint)
            
            ''' copy the intermediate way-point '''
            intermediateWayPoint = newIntermediateWayPoint 
            ''' increment the index '''
            index += 1
            
        ''' rename last point as take-off '''
        Name = 'takeOff-{0:.1f}-meters'.format(self.totalLegDistanceMeters)
        intermediateWayPoint.setName(Name)
        #logging.info( self.className + " - last ground run point = {0}".format( Name ) )
        # keep the last true airspeed
        self.lastTrueAirSpeedMetersSecond = self.aircraft.getCurrentTrueAirSpeedMetersSecond()
        
   
    def getElapsedTimeSeconds(self):
        return self.elapsedTimeSeconds

    def getTotalLegDistanceMeters(self):
        return self.totalLegDistanceMeters
    
    def getLastTrueAirSpeedMetersSecond(self):
        return self.lastTrueAirSpeedMetersSecond
