'''
Created on 27 avr. 2025

@author: robert
'''

'''
Created on 3 february 2015

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

@TODO: create a flight list class (with only fixes) , a flight plan with Lat-Long and the flight Path

manage a flight path built from a flight plan
ensure a decoupling between the lateral path and the vertical path.

To simplify the prediction process, the horizontal and vertical components of the trajectory are de-coupled.

The flight legs are consisting of a series of straight-line (great-circle) segments connected by constant radius turns.
The aircraft speed is used to calculate a turn radius.
'''

import math
import logging

from trajectory.Environment.Atmosphere.AtmosphereFile import Atmosphere
from trajectory.Environment.Earth.EarthFile import Earth
from trajectory.Environment.Utils import logElapsedRealTime

from trajectory.Guidance.FlightPlanFile import FlightPlan
from trajectory.GuidanceOpenap.GroundRunLegOpenapFile import GroundRunLeg
from trajectory.GuidanceOpenap.ClimbRampOpenapFile import ClimbRamp

from trajectory.GuidanceOpenap.TurnLegOpenapFile import TurnLeg
from trajectory.GuidanceOpenap.GreatCircleRouteOpenapFile import GreatCircleRoute
from trajectory.GuidanceOpenap.DescentGlideSlopeOpenapFile import DescentGlideSlope

from trajectory.Guidance.ConstraintsFile import ArrivalRunWayTouchDownConstraint, TargetApproachConstraint

from trajectory.Openap.AircraftMainFile import OpenapAircraft


from trajectory.Environment.Constants import Meter2Feet , GravityMetersPerSquareSeconds , Meter2NauticalMiles #= 0.000539956803 # One Meter = 0.0005 nautical miles
from trajectory.Environment.Constants import Kilogram2Pounds # = 2.20462262 # 1 kilogram = 2.204 lbs
from trajectory.Environment.Constants import DescentGlideSlopeThreeDegrees, DescentGlideSlopeDistanceNauticalMiles


class FlightPathOpenap(FlightPlan):
    
    flightPlan = None
    aircraftICAOcode = ''
    abortedFlight = False
    reducedClimbPowerCoeff = 0.0
    
    def __init__(self, 
                 route, 
                 aircraftICAOcode = 'A320', 
                 RequestedFlightLevel  = 330.0, 
                 cruiseMach            = 0.8, 
                 takeOffMassKilograms  = 62000.0,
                 reducedClimbPowerCoeff = 0.0):
        
        ''' The root logger always defaults to WARNING level. '''
        logging.getLogger().setLevel(logging.INFO)
        logging.info("----- FlightPathOpenap init -------")
        
        self.className = self.__class__.__name__
        self.abortedFlight = False
        
        ''' initialize mother class '''
        FlightPlan.__init__(self, route)
        
        ''' first bad and incomplete flight length '''
        ''' missing last turn and glide slope '''
        self.flightLengthMeters = self.computeLengthMeters() 
        
        self.aircraftICAOcode = aircraftICAOcode
        
        ''' aircraft object '''
        self.aircraft = OpenapAircraft( str(aircraftICAOcode).lower() , Earth() , Atmosphere() , initialMassKilograms = None)
        
        assert isinstance(self.aircraft, OpenapAircraft) and not(self.aircraft is None)
        self.aircraft.setAircraftMassKilograms(takeOffMassKilograms)
        self.aircraft.setInitialMassKilograms(takeOffMassKilograms)
        
        #logging.info ( self.className + " : Max TakeOff Weight kilograms = {0:.2f}".format(self.aircraft.getMaximumTakeOffMassKilograms() ) )   
        #logging.info ( self.className + " : Max Cruise Altitude Feet = {0:.2f}".format(self.aircraft.getMaxCruiseAltitudeFeet() ) )   
        #logging.info ( self.className + " : Max Speed MMO Cruise Mach = {0:.2f}".format(self.aircraft.getMaximumSpeedMmoMach() ) )
        
        #self.RequestedFlightLevel = self.aircraft.getMaxCruiseAltitudeFeet() / 100.0
        ''' requested flight level such as FL 330 '''
        self.RequestedFlightLevel = RequestedFlightLevel
        
        ''' sanity checks '''
        #assert self.RequestedFlightLevel >= MinFlightLevel and self.RequestedFlightLevel <= MaxFlightLevel
        #logging.info( self.className +  " - set cruise level = {0:.2f} feet".format( self.RequestedFlightLevel * 100.0 ))
        
        self.aircraft.setCruiseLevelFeet(self.RequestedFlightLevel * 100.0)
        
        #logging.info( self.className +  " - set target cruise mach")
        self.aircraft.setTargetCruiseMach(targetCruiseMach = cruiseMach)
        # 17th july 2023
        #self.aircraft.setReducedClimbPowerCoeff( reducedClimbPowerCoeff )
        
        self.arrivalAirport = self.getArrivalAirport()
        if (self.arrivalAirport is None):
            logging.info (self.className + ' there is no arrival airport => flight is out-bound !!!')
        #assert isinstance(self.arrivalAirport, Airport) and not(self.arrivalAirport is None)
        #assert isinstance ( self.arrivalAirport , Airport ) and not ( self.arrivalAirport is None)
        self.aircraft.setArrivalRunwayMSLmeters( self.arrivalAirport.getFieldElevationAboveSeaLevelMeters())
        
        self.departureAirport = self.getDepartureAirport()
        #assert isinstance(self.departureAirport, Airport) and not(self.departureAirport is None)
        self.aircraft.setDepartureRunwayMSLmeters ( self.departureAirport.getFieldElevationAboveSeaLevelMeters())
        
    def getAircraft(self):
        if ( self.aircraft ):
            return self.aircraft
        else:
            raise ValueError(self.className + ' aircraft not found= ' + self.aircraftICAOcode)

    def printPassedWayPoint(self, finalWayPoint):
        
        distanceFlownNautics = self.finalRoute.getLengthMeters() * Meter2NauticalMiles
        strMsg = ' passing way-point: {0} - alt= {1:.2f} meters - alt= {2:.2f} feet - already flown distance= {3:.2f} Nm'.format(
                                                    finalWayPoint.getName(),
                                                    finalWayPoint.getAltitudeMeanSeaLevelMeters(),
                                                    finalWayPoint.getAltitudeMeanSeaLevelMeters() * Meter2Feet,
                                                    distanceFlownNautics)
        logging.info  ( "{0} - {1}".format (self.className, strMsg ) )
        elapsedTimeSeconds = finalWayPoint.getElapsedTimeSeconds()
        logElapsedRealTime ( self.className  , elapsedTimeSeconds)
        
    def turnAndFly(self, 
                   tailWayPoint, 
                   headWayPoint,
                   initialHeadingDegrees,
                   headWayPointIndex):
        ''' 
        execute a turn to align true heading and then fly a great circle 
        '''    
        #logging.debug (' ================== one Turn Leg for each fix in the list =============== ')
        turnLeg = TurnLeg(  initialWayPoint = tailWayPoint,
                            finalWayPoint = headWayPoint,
                            initialHeadingDegrees = initialHeadingDegrees,
                            aircraft = self.aircraft,
                            reverse = False)
        
        distanceToLastFixMeters = self.computeDistanceToLastFixMeters(currentPosition = tailWayPoint, fixListIndex = headWayPointIndex)
        #logging.info ( self.className + ' distance to last fix= {0:.2f} Nm'.format(distanceToLastFixMeters * Meter2NauticalMiles) )
        
        distanceStillToFlyMeters = self.flightLengthMeters - self.finalRoute.getLengthMeters()
        #logging.info ( self.className + ' still to fly= {0:.2f} Nm'.format(distanceStillToFlyMeters * Meter2NauticalMiles) )

        self.endOfSimulation = turnLeg.buildTurnLeg(deltaTimeSeconds         = self.deltaTimeSeconds,
                                                    elapsedTimeSeconds       = tailWayPoint.getElapsedTimeSeconds(), 
                                                    distanceStillToFlyMeters = distanceStillToFlyMeters,
                                                    distanceToLastFixMeters  = distanceToLastFixMeters)
        self.finalRoute.addGraph(turnLeg)

        if (self.endOfSimulation == False):
            #logging.debug ( self.className + ' ==================== end of turn leg  ==================== ' )
                
            endOfTurnLegWayPoint = self.finalRoute.getLastVertex().getWeight()
            lastLeg = self.finalRoute.getLastEdge()
            #logging.debug ( self.className + ' end of turn orientation= {0:.2f} degrees'.format(lastLeg.getBearingTailHeadDegrees()) )
    
            '''==================== check if anticipated turn or fly by is applicable '''
            anticipatedTurnWayPoint = None
            if (self.flightListIndex + 2) < len(self.fixList):
                ''' still another fix in the list '''
                firstAngleDegrees  = endOfTurnLegWayPoint.getBearingDegreesTo(headWayPoint)
                secondAngleDegrees = headWayPoint.getBearingDegreesTo(self.wayPointsDict[self.fixList[self.flightListIndex+2]])
                firstAngleRadians  = math.radians(firstAngleDegrees)
                secondAngleRadians = math.radians(secondAngleDegrees)
    
                angleDifferenceDegrees = math.degrees(math.atan2(math.sin(secondAngleRadians-firstAngleRadians), math.cos(secondAngleRadians-firstAngleRadians)))
                #logging.debug ( self.className + ' difference= {0:.2f} degrees'.format(angleDifferenceDegrees) )
    
                tasMetersPerSecond = self.aircraft.getCurrentTrueAirSpeedMetersSecond()
                radiusOfTurnMeters = (tasMetersPerSecond * tasMetersPerSecond) / ( GravityMetersPerSquareSeconds * math.tan(math.radians(15.0)))
    
                anticipatedTurnStartMeters = radiusOfTurnMeters * math.tan(math.radians((180.0 - abs(angleDifferenceDegrees))/2.0))
                #logging.debug ( self.className + ' anticipated turn start from end point= {0:.2f} meters'.format(anticipatedTurnStartMeters) )
            
                if ((endOfTurnLegWayPoint.getDistanceMetersTo(headWayPoint) > (1.1 * anticipatedTurnStartMeters) 
                    and abs(angleDifferenceDegrees) > 30.)):
                    
                    logging.debug ( self.className + ' Envisage anticipated Fly By turn !!!' )
                    bearingDegrees = math.fmod ( firstAngleDegrees + 180.0 , 360.0 )
                    anticipatedTurnWayPoint = headWayPoint.getWayPointAtDistanceBearing(Name = 'flyby-' + headWayPoint.getName(),
                                                                                        DistanceMeters = anticipatedTurnStartMeters,
                                                                                        BearingDegrees = bearingDegrees)
                    headWayPoint = anticipatedTurnWayPoint
            
            #logging.info ( self.className + ' ==================== great circle ======================== ' )
            greatCircle = GreatCircleRoute( initialWayPoint = endOfTurnLegWayPoint,
                                            finalWayPoint = headWayPoint,
                                            aircraft = self.aircraft)
            
            distanceToLastFixMeters = self.computeDistanceToLastFixMeters(currentPosition = endOfTurnLegWayPoint,
                                                                          fixListIndex = headWayPointIndex)
            #logging.info ( self.className + ' distance to last fix= {0:.2f} Nm'.format(distanceToLastFixMeters * Meter2NauticalMiles) )
            
            distanceStillToFlyMeters = self.flightLengthMeters - self.finalRoute.getLengthMeters()
            #logging.info ( self.className + ' still to fly= {0:.2f} Nm'.format(distanceStillToFlyMeters * Meter2NauticalMiles) )
            
            self.endOfSimulation = greatCircle.computeGreatCircle(deltaTimeSeconds           = self.deltaTimeSeconds,
                                                                    elapsedTimeSeconds       = endOfTurnLegWayPoint.getElapsedTimeSeconds(),
                                                                    distanceStillToFlyMeters = distanceStillToFlyMeters,
                                                                    distanceToLastFixMeters  = distanceToLastFixMeters)
            ''' update final route '''
            self.finalRoute.addGraph(greatCircle)
                    
            #logging.debug ( ' ================== end of great circle ================== ' )
            
            finalWayPoint = self.finalRoute.getLastVertex().getWeight()
            
            lastLeg = self.finalRoute.getLastEdge()
            finalHeadingDegrees = lastLeg.getBearingTailHeadDegrees()
    
            distanceStillToFlyMeters = self.flightLengthMeters - self.finalRoute.getLengthMeters()
            #logging.debug ( self.className + ' still to fly= {0:.2f} meters - still to fly= {1:.2f} Nm'.format(distanceStillToFlyMeters, distanceStillToFlyMeters * Meter2NauticalMiles) )
            ''' logging.debug the way point that has been passed right now '''
            #self.printPassedWayPoint(finalWayPoint)
            
        ''' return to caller '''
        return self.endOfSimulation, finalHeadingDegrees, finalWayPoint.getElapsedTimeSeconds(), anticipatedTurnWayPoint
        
    def loopThroughFixList(self, 
                           initialHeadingDegrees, 
                           elapsedTimeSeconds):
        
        #print ( self.className + " : loop through fix list")
        anticipatedTurnWayPoint = None
        ''' start loop over the fix list '''
        ''' assumption: fix list does not contain departure and arrival airports '''
        self.flightListIndex = 0
        ''' loop over the fix list '''
        self.endOfSimulation = False
        
        while (self.endOfSimulation == False) and (self.flightListIndex < len(self.fixList)):
                
            ''' get the next fix '''
            if (anticipatedTurnWayPoint is None):
                fix = self.fixList[self.flightListIndex]
                ''' tail way point to reach '''
                tailWayPoint = self.wayPointsDict[fix]
            else:
                ''' we do not use the next fix but the anticipated turn way point '''
                tailWayPoint = anticipatedTurnWayPoint
                
            tailWayPoint.setElapsedTimeSeconds(elapsedTimeSeconds)
                                
            if (self.flightListIndex + 1) < len(self.fixList):  
                ''' next way point is still in the fix list => not yet the arrival airport '''
                headWayPoint = self.wayPointsDict[self.fixList[self.flightListIndex+1]]
                #logging.debug ( headWayPoint )
                ''' turn and fly '''
                self.endOfSimulation, initialHeadingDegrees , elapsedTimeSeconds , anticipatedTurnWayPoint = self.turnAndFly(
                                                                            tailWayPoint = tailWayPoint,
                                                                            headWayPoint = headWayPoint,
                                                                            initialHeadingDegrees = initialHeadingDegrees,
                                                                            headWayPointIndex =  self.flightListIndex)
            ''' prepare for next loop '''
            self.flightListIndex += 1

        ''' return final heading of the last great circle '''
        return self.endOfSimulation, initialHeadingDegrees
    
    def buildDeparturePhase(self):
        ''' this function manages the departure phases with a ground run and a climb ramp  '''
        logging.info ( self.className + ' : ============== build the departure ground run =========== '  )
        
        self.finalRoute = GroundRunLeg( runway   = self.departureRunway, 
                                        aircraft = self.aircraft,
                                        airport  = self.departureAirport)
        
        distanceToLastFixMeters = self.computeDistanceToLastFixMeters(currentPosition = self.departureAirport,
                                                                      fixListIndex = 0)
        #logging.info(self.className + " : distance to last fix {0} meters".format(distanceToLastFixMeters))
        
        distanceStillToFlyMeters = self.flightLengthMeters - self.finalRoute.getLengthMeters()
        #logging.info(self.className + " : distance still to fly {0} meters".format(distanceToLastFixMeters))

        elapsedTimeSeconds = 0.0
        self.finalRoute.buildDepartureGroundRun(deltaTimeSeconds         = self.deltaTimeSeconds,
                                                elapsedTimeSeconds       = elapsedTimeSeconds,
                                                distanceStillToFlyMeters = distanceStillToFlyMeters,
                                                distanceToLastFixMeters  = distanceToLastFixMeters)
        
        #logging.info( self.className + " - end of departure ground run")
        distanceStillToFlyMeters = self.flightLengthMeters - self.finalRoute.getLengthMeters()
        
        ''' return values as expected at the end of this take off phase ( ground run PLUS climb ramp) '''
        lastLeg = self.finalRoute.getLastEdge()
        initialHeadingDegrees = lastLeg.getBearingTailHeadDegrees()
        initialWayPoint = self.finalRoute.getLastVertex().getWeight()
        self.endOfSimulation = False
        
        ''' check if runway overshoot '''
        if ( self.finalRoute.getTotalLegDistanceMeters() > self.departureRunway.getLengthMeters()):
            #print ("ground run length = {0:.2f} meters - runway length = {1:.2f} meters".format( self.finalRoute.getTotalLegDistanceMeters() , self.departureRunway.getLengthMeters()))
            logging.info ( self.className + " -----> runway overshoot---------")
            self.endOfSimulation = True
            
        else:
                        
            distanceToFirstFixNautics = initialWayPoint.getDistanceMetersTo(self.getFirstWayPoint()) * Meter2NauticalMiles
            ''' next lateral phase is a climb ramp '''
            climbRamp = ClimbRamp(  initialWayPoint  = initialWayPoint,
                                    runway           = self.departureRunway, 
                                    aircraft         = self.aircraft,
                                    departureAirport = self.departureAirport)
            
            ''' climb ramp of 5.0 nautical miles is not possible if first fix placed in between '''
            ''' @TODO - First fix must not be allowed nearer to 5 Nautical Miles '''
            climbRampLengthNautics = min(distanceToFirstFixNautics / 2.0 , 5.0)
            ''' build the climb ramp '''
            climbRamp.buildClimbRamp(deltaTimeSeconds         = self.deltaTimeSeconds,
                                     elapsedTimeSeconds       = initialWayPoint.getElapsedTimeSeconds(),
                                     distanceStillToFlyMeters = distanceStillToFlyMeters ,
                                     distanceToLastFixMeters  = distanceToLastFixMeters,
                                     climbRampLengthNautics   = climbRampLengthNautics ,
                                     totalDistanceFlownMeters = self.finalRoute.getTotalLegDistanceMeters())
            self.finalRoute.addGraph(climbRamp)
            
            initialWayPoint = self.finalRoute.getLastVertex().getWeight()
            lastLeg = self.finalRoute.getLastEdge()
            initialHeadingDegrees = lastLeg.getBearingTailHeadDegrees()
            #logging.info ( self.className + ' last leg orientation= {0:.2f} degrees'.format(initialHeadingDegrees) )
            
            #'''============= add way point in the fix list =============== '''
            self.insert(position = 'begin', wayPoint= initialWayPoint )
            
        return self.endOfSimulation , initialHeadingDegrees , initialWayPoint
        
    def buildSimulatedArrivalPhase(self):
        
        #logging.info( self.className + " - build simulated arrival phase")
        
        self.aircraft.setArrivalRunwayMSLmeters( self.arrivalAirport.getFieldElevationAboveSeaLevelMeters() )

        ''' simulated arrival phase to compute fix at 10Nm of arrival srunway '''
        #logging.info ( self.className + '=========== add final turn, descent and ground run ===================' )
        arrivalGroundRun = GroundRunLeg( runway   = self.arrivalRunway,
                                         aircraft = self.aircraft,
                                         airport  = self.arrivalAirport )
        self.touchDownWayPoint = arrivalGroundRun.computeTouchDownWayPoint()
        # add touch down to constraint list
        self.constraintsList.append(ArrivalRunWayTouchDownConstraint(self.touchDownWayPoint))
        
        #logging.info ( self.className + " - " + str(self.touchDownWayPoint) )
        ''' distance from last fix to touch down '''
        distanceToLastFixNautics = self.touchDownWayPoint.getDistanceMetersTo(self.getLastWayPoint()) * Meter2NauticalMiles
        
        #logging.info ( self.className + '===================== final 3 degrees descending glide slope ================' )
        descentGlideSlope = DescentGlideSlope( runway   = self.arrivalRunway,
                                               aircraft = self.aircraft,
                                               arrivalAirport = self.arrivalAirport,
                                               descentGlideSlopeDegrees = DescentGlideSlopeThreeDegrees)
        
        ''' if there is a fix nearer to 5 Nm of the touch-down then limit size of simulated glide slope '''
        descentGlideSlopeSizeNautics = min(distanceToLastFixNautics / 2.0 , DescentGlideSlopeDistanceNauticalMiles)
        
        ''' build simulated glide slope '''
        descentGlideSlope.buildSimulatedGlideSlope(descentGlideSlopeSizeNautics)
        #logging.info( self.className + " - Simulated arrival descent glide slope is correctly finished")
        
        self.firstGlideSlopeWayPoint = descentGlideSlope.getVertex(v=0).getWeight()
        #logging.info ( self.className + ' top of arrival glide slope= {0}'.format(str(self.firstGlideSlopeWayPoint)) )
        
        #logging.info ( self.className + ' ================= need a turn leg to find the junction point the last way-point in the fix list to the top of the final glide slope' )
        '''
        initial heading is the orientation of the run-way
        '''
        lastFixListWayPoint = self.wayPointsDict[self.fixList[-1]]
        initialHeadingDegrees = self.arrivalRunway.getTrueHeadingDegrees()
        
        #logging.info ( "=====> arrival runway - true heading degrees = {0}".format(initialHeadingDegrees))

        lastTurnLeg = TurnLeg( initialWayPoint = self.firstGlideSlopeWayPoint, 
                           finalWayPoint = lastFixListWayPoint,
                           initialHeadingDegrees = initialHeadingDegrees, 
                           aircraft = self.aircraft,
                           reverse = True)
        
        ''' 16th January 2022 - final radius of turn '''
        finalRadiusOfTurnMeters = lastTurnLeg.buildNewSimulatedArrivalTurnLeg(
                                                 deltaTimeSeconds         = self.deltaTimeSeconds,
                                                 elapsedTimeSeconds       = 0.0,
                                                 distanceStillToFlyMeters = 0.0,
                                                 simulatedAltitudeSeaLevelMeters = self.firstGlideSlopeWayPoint.getAltitudeMeanSeaLevelMeters(),
                                                 flightPathAngleDegrees = 3.0,
                                                 bankAngleDegrees = 5.0)
        descentGlideSlope.addGraph(lastTurnLeg)
        ''' prepare next step '''
        beginOfLastTurnLeg = lastTurnLeg.getVertex(v=0).getWeight()
        #logging.info ( self.className + ' begin last turn= {0}'.format(str(beginOfLastTurnLeg) ))
        
        ''' add to constraint list '''
        self.constraintsList.append(TargetApproachConstraint(beginOfLastTurnLeg))
        
        ''' add the three last way-points in the fix list '''
        self.insert(position = 'end', wayPoint = beginOfLastTurnLeg )
        
        ''' update the length of the flight path '''
        self.distanceFromApproachToTouchDownMeters = descentGlideSlope.getLengthMeters()
        self.flightLengthMeters = self.computeLengthMeters() + descentGlideSlope.getLengthMeters()
        
        #logging.info ( self.className + ' updated flight path length= {0:.2f} Nm'.format(self.flightLengthMeters * Meter2NauticalMiles ) )
        
        ''' target approach fix is equal to the begin of the SIMULATED last turn leg '''
        self.aircraft.setTargetApproachWayPoint(beginOfLastTurnLeg)
        
        self.aircraft.setArrivalRunwayTouchDownWayPoint(self.touchDownWayPoint)
        #logging.info ( self.className + ' arrival runway touch down point = {0}'.format(str(self.touchDownWayPoint) ))
        
        logging.info ( self.className + ' fix list= {0}'.format(self.fixList) )
        
        ''' 16th January 2022 - Robert - return the final radius of turn '''
        return finalRadiusOfTurnMeters
        

    def buildArrivalPhase(self, initialHeadingDegrees , finalRadiusOfTurnMeters):
        
        #logging.info ( self.className + ' initial heading= {0:.2f} degrees'.format(initialHeadingDegrees) )
        #logging.info ( self.className + ' ==================== add last turn ==================== ' )
        if self.isDomestic() or self.isInBound():
            
            endOfLastGreatCircleWayPoint = self.finalRoute.getLastVertex().getWeight()
                
            finalHeadingDegrees = self.arrivalRunway.getTrueHeadingDegrees()
            finalHeadingDegrees = math.fmod ( finalHeadingDegrees + 180.0 , 360.0 )
            #logging.info ( self.className + ' runway final heading = {0:.2f} deg'.format(finalHeadingDegrees) )
            
            turnLeg = TurnLeg(  initialWayPoint           = endOfLastGreatCircleWayPoint,
                                    #finalWayPoint    = self.firstGlideSlopeWayPoint,
                                    finalWayPoint         = self.touchDownWayPoint,
                                    initialHeadingDegrees = initialHeadingDegrees,
                                    aircraft              = self.aircraft,
                                    reverse               = False)
            
            distanceStillToFlyMeters = self.flightLengthMeters - self.finalRoute.getLengthMeters()
            distanceToLastFixMeters = self.computeDistanceToLastFixMeters(currentPosition = endOfLastGreatCircleWayPoint,
                                                                          fixListIndex    = self.flightListIndex)
            distanceToLastFixMeters = distanceStillToFlyMeters
            ''' for the last turn => final heading towards the runway orientation '''
            deltaTimeSeconds = 1.0
            turnLeg.buildTurnLeg(deltaTimeSeconds         = deltaTimeSeconds,
                                 elapsedTimeSeconds       = endOfLastGreatCircleWayPoint.getElapsedTimeSeconds(),
                                 distanceStillToFlyMeters = distanceStillToFlyMeters,
                                 distanceToLastFixMeters = distanceToLastFixMeters,
                                 finalHeadingDegrees = finalHeadingDegrees,
                                 lastTurn = True,
                                 bankAngleDegrees = 5.0,
                                 arrivalRunway = self.arrivalRunway,
                                 finalRadiusOfTurnMeters = finalRadiusOfTurnMeters)
            self.finalRoute.addGraph(turnLeg)
                
            endOfTurnLegWayPoint = self.finalRoute.getLastVertex().getWeight()
            ''' ============= use touch-down way-point to compute distance to fly ============='''
            distanceStillToFlyMeters = endOfTurnLegWayPoint.getDistanceMetersTo(self.touchDownWayPoint)
            #logging.info ( self.className + ' distance still to fly= {0:.2f} Nm'.format(distanceStillToFlyMeters * Meter2NauticalMiles) )
    
            #logging.debug '==================== add descent slope ================= '
            descentGlideSlope = DescentGlideSlope( runway   = self.arrivalRunway,
                                                        aircraft = self.aircraft,
                                                        arrivalAirport = self.arrivalAirport,
                                                        descentGlideSlopeDegrees = 3.0)
                  
            flownDistanceMeters = self.finalRoute.getLengthMeters()    
            distanceStillToFlyMeters = self.flightLengthMeters - self.finalRoute.getLengthMeters()
            distanceToLastFixMeters = self.computeDistanceToLastFixMeters(currentPosition = endOfTurnLegWayPoint,
                                                                          fixListIndex    = self.flightListIndex)
            distanceToLastFixMeters = distanceStillToFlyMeters

            descentGlideSlope.buildGlideSlope( deltaTimeSeconds         = self.deltaTimeSeconds,
                                               elapsedTimeSeconds       = endOfTurnLegWayPoint.getElapsedTimeSeconds(), 
                                               initialWayPoint          = endOfTurnLegWayPoint, 
                                               flownDistanceMeters      = flownDistanceMeters, 
                                               distanceStillToFlyMeters = distanceStillToFlyMeters ,
                                               distanceToLastFixMeters  = distanceToLastFixMeters)
            
            self.finalRoute.addGraph(descentGlideSlope)
            endOfDescentGlideSlope = self.finalRoute.getLastVertex().getWeight()
            
            #logging.debug '================= add arrival ground run ================'
            arrivalGroundRun = GroundRunLeg( runway   = self.arrivalRunway,
                                          aircraft = self.aircraft,
                                          airport  = self.arrivalAirport )
            
            arrivalGroundRun.buildArrivalGroundRun(deltaTimeSeconds = self.deltaTimeSeconds,
                                                   elapsedTimeSeconds = endOfDescentGlideSlope.getElapsedTimeSeconds(),
                                                    initialWayPoint = endOfDescentGlideSlope)
            
            self.finalRoute.addGraph(arrivalGroundRun)
            ''' set total elapsed time seconds '''
            #logging.debug ("------------------- elapsed time = {0} seconds -------------".format( int ( arrivalGroundRun.getElapsedTimeSeconds() ) ) ) 
            logElapsedRealTime( self.className , int ( arrivalGroundRun.getElapsedTimeSeconds() ) )
            #logging.debug ("------------------- end of arrival ground run ----------")
            self.elapsedTimeSeconds = arrivalGroundRun.getElapsedTimeSeconds()
        
      
    def computeFlight(self, deltaTimeSeconds):
        #logging.info ( self.className + " : compute flight")
        ''' 
        main entry to compute a whole flight 
        '''
        self.deltaTimeSeconds = deltaTimeSeconds
        
        #assert not( self.aircraft is None)
        #assert not( self.departureRunway is None)
        #assert not( self.departureAirport is None)
        
        #logging.info ( self.className + " : start computing the trajectory ")
        try:
            if self.isDomestic() or self.isOutBound():
                
                #logging.info ( self.className + " - build departure phase")
                self.endOfSimulation, initialHeadingDegrees , initialWayPoint = self.buildDeparturePhase()
                
                #logging.info( self.className + " - end of departure phase")
                
            ''' end of simulation = True means the flight is aborted '''
            if ( self.endOfSimulation == False ) and ( self.isDomestic() or self.isInBound() ):
                
                #assert not(self.arrivalAirport is None)
                
                #logging.info ( self.className + " : build simulated arrival phase")
                finalRadiusOfTurnMeters = self.buildSimulatedArrivalPhase()
                #logging.info ( self.className + " - final radius of turn = {0} meters".format(finalRadiusOfTurnMeters))
                #sys.exit()
            
            #logging.debug ( self.className + "==================== Loop over the fix list ====================")
            if (self.endOfSimulation == False):
                #logging.info ( self.className + " : loop through fix list")
                
                self.endOfSimulation, initialHeadingDegrees = self.loopThroughFixList(initialHeadingDegrees = initialHeadingDegrees,
                                                                                      elapsedTimeSeconds    = initialWayPoint.getElapsedTimeSeconds())
            
            if (self.endOfSimulation == False):
                #logging.debug '=========== build arrival phase =============='
                self.buildArrivalPhase(initialHeadingDegrees, finalRadiusOfTurnMeters)
                
            if (self.endOfSimulation == False):
                logging.info ( self.className + ' ========== delta mass status ==============' )

                logging.info ( self.className + ' initial mass= {0:.2f} kilograms = {1:.2f} pounds'.format(self.aircraft.getTakeOffMassKilograms(),
                                                                                                   self.aircraft.getTakeOffMassKilograms()*Kilogram2Pounds) )
                logging.info ( self.className + ' final mass= {0:.2f} kilograms = {1:.2f} pounds'.format(self.aircraft.getCurrentMassKilograms(),
                                                                                                 self.aircraft.getCurrentMassKilograms()*Kilogram2Pounds) )
                logging.info ( self.className + ' diff mass= {0:.2f} kilograms = {1:.2f} pounds'.format(self.aircraft.getTakeOffMassKilograms()-self.aircraft.getCurrentMassKilograms(),
                                                                                                (self.aircraft.getTakeOffMassKilograms()-self.aircraft.getCurrentMassKilograms())*Kilogram2Pounds) )
                logging.info ( self.className + ' ========== delta mass status ==============' )
                
            if ( self.endOfSimulation ==  True ):
                self.abortedFlight = True
                return False
            
            return True
        
        except Exception as e:
            logging.error ("----> flight did not go to a normal end ---> {0}".format(e))
            self.abortedFlight = True
            return False
            
    def createXlsOutputFile(self):
        self.finalRoute.createXlsxOutputFile(self.abortedFlight, self.aircraftICAOcode, self.departureAirport.getICAOcode(), self.arrivalAirport.getICAOcode())
            
    def createFlightOutputFiles(self):
        ''' build outputs '''
        #self.finalRoute.createXlsxOutputFile(self.abortedFlight, self.aircraftICAOcode, self.departureAirport.getICAOcode(), self.arrivalAirport.getICAOcode())
        kmlXmlDocument = self.finalRoute.createKmlOutputFile(self.abortedFlight, self.aircraftICAOcode, self.departureAirport.getICAOcode(), self.arrivalAirport.getICAOcode())
        ''' add a prefix to the file path to identify the departure and arrival airport '''
        
        #self.aircraft.createStateVectorOutputFile(self.abortedFlight, self.aircraftICAOcode, self.departureAirport.getICAOcode(), self.arrivalAirport.getICAOcode())
        logging.debug (  '{0} - final route length = {1:.2f} Nm'.format(self.className, self.finalRoute.getLengthMeters()*Meter2NauticalMiles) )
        return kmlXmlDocument
    
    def createStateVectorHistoryFile(self):
        fileName = "{0}-{1}-{2}-Aborted-{3}".format( self.aircraftICAOcode , self.departureAirport.getICAOcode() , self.arrivalAirport.getICAOcode() , self.abortedFlight )
        return self.aircraft.createStateVectorHistoryFile( fileName )

    def createStateVectorOutputSheet(self, workbook):
        self.aircraft.createStateVectorOutputSheet(workbook, self.abortedFlight, self.aircraftICAOcode, self.departureAirport.getICAOcode(), self.arrivalAirport.getICAOcode())

    def createCsvAltitudeTimeProfile(self):
        csvAltitudeTimeProfile = self.finalRoute.createCsvAltitudeTimeProfile(self.abortedFlight, self.aircraftICAOcode, self.departureAirport.getICAOcode(), self.arrivalAirport.getICAOcode())
        return csvAltitudeTimeProfile
    
    def createKMLfileLike(self, memoryFile):
        ''' no return statement - result is written into memoryFile '''
        self.finalRoute.createKmlFileLike(memoryFile, self.abortedFlight, self.aircraftICAOcode, self.departureAirport.getICAOcode(), self.arrivalAirport.getICAOcode())
    
    def createKmlXmlDocument(self):
        kmlXmlDocument = self.finalRoute.createKmlOutputFile(self.abortedFlight, self.aircraftICAOcode, self.departureAirport.getICAOcode(), self.arrivalAirport.getICAOcode())
        return kmlXmlDocument
    
    def createKmlXmlPureDocument(self):
        kmlXmlDocument = self.finalRoute.createKmlXmlPureDocument(self.abortedFlight, self.aircraftICAOcode, self.departureAirport.getICAOcode(), self.arrivalAirport.getICAOcode())
        return kmlXmlDocument
    
    def getAircraftCurrentMassKilograms(self):
        return self.aircraft.getAircraftCurrentMassKilograms()
    
    def getFlightDurationSeconds(self):
        return self.elapsedTimeSeconds
