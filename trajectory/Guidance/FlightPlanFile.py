# -*- coding: UTF-8 -*-

'''
Created on 25 janvier 2015

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

@note: typical flight plan 

    strRoute = 'ADEP/LFBO-TOU-ALIVA-TOU37-FISTO-LMG-PD01-PD02-AMB-AMB01-AMB02-PD03-PD04-OLW11-OLW83-ADES/LFPO'

purpose : build a fix list from a route expressed as a sequence of names

@ TODO: it should be possible to insert in the flight plan
1) a lat-long expressed point such as N88-55-66W001-02-03
2) a condition such as before a given fix , a speed condition is reached (below 10.000 feet speed is lower to 250knots)

'''

import math
import logging

from trajectory.Environment.WayPoints.WayPointsDatabaseFile import WayPointsDatabase
from trajectory.Environment.Airports.AirportDatabaseFile import AirportsDatabase
from trajectory.Environment.Runways.RunWaysDatabaseFile import RunWayDataBase

from trajectory.Guidance.WayPointFile import WayPoint, Airport
from trajectory.Environment.Runways.RunWayFile import RunWay

#from trajectory.Guidance.ConstraintsFile import analyseConstraint
from trajectory.Environment.Constants import Meter2NauticalMiles

from trajectory.Guidance.FixListClass import FixList


class FlightPlan(FixList):
    
    className = ''
    
    wayPointsDict = {}
    constraintsList = []
    
    departureAirportIcaoCode = ''
    departureAirport = None
    arrivalAirportIcaoCode = ''
    arrivalAirport = None
    
    def __init__(self, strRoute):
        logging.info("----- Flight Plan init ------")
        
        self.className = self.__class__.__name__
        FixList.__init__(self, strRoute)
        
        self.wayPointsDb = WayPointsDatabase()
        assert self.wayPointsDb.exists() ==  True
        assert self.wayPointsDb.read() ==  True
        
        self.airportsDb = AirportsDatabase()
        assert self.airportsDb.read() == True
        
        self.runwaysDb = RunWayDataBase()
        assert self.runwaysDb.exists() == True
        assert self.runwaysDb.read() == True
        
        self.buildFixList()
         
    def getArrivalAirport(self):
        assert ( not(self.arrivalAirport is None) and isinstance(self.arrivalAirport, Airport))
        return self.arrivalAirport
    
    def getDepartureAirport(self):
        assert ( not(self.departureAirport is None) and isinstance(self.departureAirport, Airport))
        return self.departureAirport

    def buildFixList(self):
        #print ( self.className + " : Build the fix list")
        '''
        from the route build a fix list and from the fix list build a way point list
        '''
        self.wayPointsDict = {}
        
        ''' fill self departure airport and self.arrivalAirport '''
        self.createFixList()
        for fix in self.getFix():
            
            logging.info(self.className + ": next fix = " + fix)
            
            wayPoint = self.wayPointsDb.getWayPoint(fix)
            if (wayPoint):
                logging.info("waypoint = {0} in wayPoints database".format(wayPoint))
                self.wayPointsDict[fix] = wayPoint
            else:
                self.deleteFix(fix)
        
        ''' self.arrivalAirportICAOcode obtained from base class fix list '''
        self.arrivalAirport = self.airportsDb.getAirportFromICAOCode(ICAOcode = self.arrivalAirportICAOcode)
        assert ( not (self.arrivalAirport is None) and isinstance( self.arrivalAirport, Airport))
        
        logging.info( self.className + " : arrival airport : " + str(self.arrivalAirport))
        logging.info( self.className + " : arrival runway : " + str(self.arrivalRunwayName))
        
        self.arrivalRunway =  self.runwaysDb.getFilteredRunWays(airportICAOcode = self.arrivalAirportICAOcode, runwayName = self.arrivalRunwayName)
        assert ( not (self.arrivalRunway is None) and isinstance(self.arrivalRunway, RunWay ))
        
        self.arrivalRunway = RunWay(Name               = self.arrivalRunway.Name,
                                    airportICAOcode    = self.arrivalAirport.ICAOcode,
                                    LengthFeet         = self.arrivalRunway.LengthFeet,
                                    TrueHeadingDegrees = self.arrivalRunway.TrueHeadingDegrees,
                                    LatitudeDegrees    = self.arrivalRunway.LatitudeDegrees,
                                    LongitudeDegrees   = self.arrivalRunway.LongitudeDegrees)
        
        logging.info ( self.className + " : arrival runway : " + str(self.arrivalRunway) )

        self.departureAirport = self.airportsDb.getAirportFromICAOCode(ICAOcode = self.departureAirportICAOcode)
        assert ( not (self.departureAirport is None) and isinstance( self.departureAirport, Airport))
        
        logging.info( self.className + " : departure airport : " + str(self.departureAirport))
        #self.departureRunway = runwaysDb.getFilteredRunWays(airportICAOcode = self.departureAirportICAOcode, runwayName = self.departureRunwayName)
        self.departureRunway = self.runwaysDb.getFilteredRunWays(airportICAOcode = self.departureAirportICAOcode, runwayName = self.departureRunwayName)
        assert ( not (self.departureRunway is None) and isinstance(self.departureRunway, RunWay ))
        
        self.departureRunway = RunWay(Name             = self.departureRunway.Name,
                                    airportICAOcode    = self.departureAirport.ICAOcode,
                                    LengthFeet         = self.departureRunway.LengthFeet,
                                    TrueHeadingDegrees = self.departureRunway.TrueHeadingDegrees,
                                    LatitudeDegrees    = self.departureRunway.LatitudeDegrees,
                                    LongitudeDegrees   = self.departureRunway.LongitudeDegrees)
        logging.info ( self.className + " : departure runway : " + str(self.departureRunway) )

        #logging.debug self.className + ': fix list= ' + str(self.fixList)
        assert (self.allAnglesLessThan90degrees(minIntervalNautics = 10.0))
        
    def insert(self, position, wayPoint):
        ''' 
        insert a waypoint in the list and add the waypoint to the flight plan dictionary 
        '''
        assert (isinstance(wayPoint, WayPoint))

        if position == 'begin':
            self.fixList.insert(0, wayPoint.getName())
        elif position == 'end':
            self.fixList.insert(len(self.fixList), wayPoint.getName())
        else:
            if isinstance(position, int):
                self.fixList.insert(position, wayPoint.getName())

        # need to ensure that the same name does not appear twice in the list
        self.wayPointsDict[wayPoint.getName()] = wayPoint

    def getFirstWayPoint(self):
        ''' 
        if fix list is empty , need at least an arrival airport 
        '''
        if len(self.fixList) > 0:
            firstFix = self.fixList[0]
            return self.wayPointsDict[firstFix]
        else:
            ''' fix list is empty => need a departure airport at least '''
            assert not(self.departureAirport is None) and isinstance( self.departureAirport, Airport)
            return self.departureAirport
      
    def getLastWayPoint(self):
        ''' if fix list is empty, return arrival airport '''
        if len(self.fixList) > 0:
            lastFix = self.fixList[-1]
            return self.wayPointsDict[lastFix]
        else:
            assert (not(self.arrivalAirport is None)) and isinstance(self.arrivalAirport , Airport)
            return self.arrivalAirport
        
    def isOverFlight(self):
        return (self.departureAirport is None) and (self.arrivalAirport is None)
    
    def isDomestic(self):
        return not(self.departureAirport is None) and not(self.arrivalAirport is None)
    
    def isInBound(self):
        return (self.departureAirport is None) and not(self.arrivalAirport is None)
        
    def isOutBound(self):
        return not(self.departureAirport is None) and (self.arrivalAirport is None)
    
    def checkAnglesGreaterTo(self, 
                             firstWayPoint, 
                             secondWayPoint, 
                             thirdWayPoint, 
                             maxAngleDifferenceDegrees = 45.0):
        
        logging.debug (self.className + ': {0} - {1} - {2}'.format(firstWayPoint.getName(), secondWayPoint.getName(), thirdWayPoint.getName()) )
        firstAngleDegrees = firstWayPoint.getBearingDegreesTo(secondWayPoint)
        secondAngleDegrees = secondWayPoint.getBearingDegreesTo(thirdWayPoint)
                
        assert (firstAngleDegrees >= 0.0) and (secondAngleDegrees >= 0.0)
        
        logging.debug ( self.className + ': first angle= {0:.2f} degrees and second angle= {1:.2f} degrees'.format(firstAngleDegrees, secondAngleDegrees) )
        firstAngleRadians = math.radians(firstAngleDegrees)
        secondAngleRadians = math.radians(secondAngleDegrees)

        angleDifferenceDegrees = math.degrees(math.atan2(math.sin(secondAngleRadians-firstAngleRadians), math.cos(secondAngleRadians-firstAngleRadians))) 
        logging.debug (self.className + ': difference= {0:.2f} degrees'.format(angleDifferenceDegrees) )
        
        if abs(angleDifferenceDegrees) > maxAngleDifferenceDegrees:
            logging.debug ( self.className + ': WARNING - angle difference=  {0:.2f} greater to {1:.2f} degrees'.format(angleDifferenceDegrees, maxAngleDifferenceDegrees) )
            return False
        
        firstIntervalDistanceNm = firstWayPoint.getDistanceMetersTo(secondWayPoint) * Meter2NauticalMiles
        secondIntervalDistanceNm = secondWayPoint.getDistanceMetersTo(thirdWayPoint) * Meter2NauticalMiles
        if (firstIntervalDistanceNm < 20.0):
            logging.debug ( self.className + ': WARNING - distance between {0} and {1} less than 20 Nm = {2:.2f}'.format(firstWayPoint.getName(), secondWayPoint.getName(), firstIntervalDistanceNm) )
        if (secondIntervalDistanceNm < 20.0):
            logging.debug ( self.className + ': WARNING - distance between {0} and {1} less than 20 Nm = {2:.2f}'.format(secondWayPoint.getName(), thirdWayPoint.getName(), secondIntervalDistanceNm) )

        return True

    def isDistanceLessThan(self, 
                           firstIndex, 
                           secondIndex, 
                           minIntervalNautics = 10.0):
        '''
        check distance between two index in the fix list 
        '''
        assert (len(self.fixList) > 0) 
        assert  (firstIndex >= 0) and (firstIndex < len(self.fixList))
        assert  (secondIndex >= 0) and (secondIndex < len(self.fixList))
        assert (firstIndex != secondIndex)
        
        firstWayPoint = self.wayPointsDict[self.fixList[firstIndex]]
        secondWayPoint = self.wayPointsDict[self.fixList[secondIndex]]
        IntervalDistanceNm = firstWayPoint.getDistanceMetersTo(secondWayPoint) * Meter2NauticalMiles
        if IntervalDistanceNm < ( minIntervalNautics - 1.0):
            logging.debug ( self.className + ': WARNING - distance between {0} and {1} less than 10 Nm = {2:.2f}'.format(firstWayPoint.getName(), secondWayPoint.getName(), IntervalDistanceNm) )
            return True
        return False

    def allAnglesLessThan90degrees(self, minIntervalNautics = 10.0):
        ''' returns True if all contiguous angles lower to 90 degrees '''
        ''' suppress point not compliant with the distance interval rules '''
        
        ''' Note: need 3 way-points to build 2 contiguous angles '''
        oneFixSuppressed = True
        while oneFixSuppressed:
            index = 0
            oneFixSuppressed = False
            for fix in self.fixList:
                logging.debug ( self.className + ': fix= {0}'.format(fix) )
                
                if index == 1 and not(self.departureAirport is None):
                    firstWayPoint = self.departureAirport
                    logging.debug ( firstWayPoint )
                    secondWayPoint = self.wayPointsDict[self.fixList[index-1]]
                    logging.debug ( secondWayPoint )
                    thirdWayPoint = self.wayPointsDict[self.fixList[index]]
                    logging.debug ( thirdWayPoint )
                    if (self.isDistanceLessThan(firstIndex = index-1, 
                                             secondIndex = index, 
                                             minIntervalNautics = minIntervalNautics) == True):
                        ''' suppress the point from the fix list '''
                        logging.debug ( self.className + ': fix suppressed= {0}'.format(self.fixList[index]) )
                        self.fixList.pop(index)
                        oneFixSuppressed = True
                        
                    if oneFixSuppressed:
                        logging.debug ( self.className + ': start the whole loop again from the very beginning ' )
                        break
                    else:
                        self.checkAnglesGreaterTo(firstWayPoint, 
                                              secondWayPoint, 
                                              thirdWayPoint,
                                              maxAngleDifferenceDegrees = 30.0)
    
                if index >= 2:
                    
                    firstWayPoint = self.wayPointsDict[self.fixList[index-2]]
                    logging.debug ( firstWayPoint )
                    secondWayPoint = self.wayPointsDict[self.fixList[index-1]]
                    logging.debug ( secondWayPoint )
                    if (self.isDistanceLessThan(firstIndex = index - 2, 
                                             secondIndex = index - 1, 
                                             minIntervalNautics = minIntervalNautics) == True):
                        ''' suppress the point from the fix list '''
                        logging.debug ( self.className + ': fix suppressed= {0}'.format(self.fixList[index-1]) )
                        self.fixList.pop(index-1)
                        oneFixSuppressed = True
                        
                    thirdWayPoint = self.wayPointsDict[self.fixList[index]]
                    logging.debug ( thirdWayPoint )
                    if (self.isDistanceLessThan(firstIndex = index - 1, 
                                             secondIndex = index, 
                                             minIntervalNautics = minIntervalNautics) == True) and not (self.indexIsTheLast(index)):
                        ''' suppress the point from the fix list '''
                        logging.debug ( self.className + ': fix suppressed= {0}'.format(self.fixList[index]) )
                        self.fixList.pop(index)
                        oneFixSuppressed = True
                    
                    if oneFixSuppressed:
                        logging.debug ( self.className + ': start the whole loop again from the very beginning ' )
                        break
                    else:
                        self.checkAnglesGreaterTo(firstWayPoint, 
                                              secondWayPoint, 
                                              thirdWayPoint,
                                              maxAngleDifferenceDegrees = 30.0)
    
                logging.debug ( self.className + '============ index = {0} ==========='.format(index) )
                index += 1
        return True
    
    def computeLengthNauticalMiles(self):
        return self.computeLengthMeters() * Meter2NauticalMiles
    
    def computeLengthMeters(self):
        '''
        returns a float corresponding to the length of the route in Meters 
        if there is a arrival airport , distance from last fix to arrival airport is added
        '''
        lengthMeters = 0.0
        index = 0
        for fix in self.fixList:
            #logging.debug fix
            if not(self.departureAirport is None) and isinstance(self.departureAirport, Airport ): 
                if index == 0:
                    lengthMeters += self.departureAirport.getDistanceMetersTo(self.wayPointsDict[fix])
                    previousWayPoint = self.wayPointsDict[fix]

                else:
                    lengthMeters += previousWayPoint.getDistanceMetersTo(self.wayPointsDict[fix])
                    previousWayPoint = self.wayPointsDict[fix]

            else:
                ''' no departure airport '''
                if index == 0:
                    previousWayPoint = self.wayPointsDict[fix]
                else:
                    lengthMeters += previousWayPoint.getDistanceMetersTo(self.wayPointsDict[fix]) 
                    previousWayPoint = self.wayPointsDict[fix]

            index += 1
            
        ''' add distance from last fix to arrival airport if applicable '''
        if not(self.arrivalAirport is None) and isinstance(self.arrivalAirport, Airport):
            #logging.debug self.className + ': last fix= ' + self.fixList[-1]
            if len(self.wayPointsDict) > 0:
                lengthMeters += self.wayPointsDict[self.fixList[-1]].getDistanceMetersTo(self.arrivalAirport)
            else:
                raise self.className + " - wayPoints Dictionary is empty !!!"
                if (not(self.departureAirport is None) and isinstance( self.departureAirport, Airport )):
                    lengthMeters += self.departureAirport.getDistanceMetersTo(self.arrivalAirport)
            
        return lengthMeters 

    def computeDistanceToLastFixMeters(self, currentPosition, fixListIndex):
        '''
        compute length to fly from the provided index in the fix list
        '''
        lengthMeters = 0.0
        if fixListIndex == len(self.fixList):
            return 0.0

        assert (len(self.fixList) > 0) 
        assert (fixListIndex >= 0) 
        assert (fixListIndex < len(self.fixList))
        assert (len(self.wayPointsDict) > 0)
        if len(self.fixList) == 1:
            return 0.0

        for index in range(fixListIndex, len(self.fixList)):
            #logging.debug index
            if index == fixListIndex:
                firstWayPoint = currentPosition
            else:
                firstWayPoint = self.wayPointsDict[self.fixList[index]]
            #logging.debug firstWayPoint
            if index + 1 < len(self.fixList):
                secondWayPoint = self.wayPointsDict[self.fixList[index+1]]
                #logging.debug secondWayPoint
                lengthMeters += firstWayPoint.getDistanceMetersTo(secondWayPoint)
                #logging.debug self.className + ': first wayPoint= {0} - second wayPoint= {1}'.format(firstWayPoint, secondWayPoint)
        
        ''' do not count distance from last fix to arrival airport '''
#         if not(self.arrivalAirport is None):
#             lengthMeters += self.wayPointsDict[self.fixList[-1]].getDistanceMetersTo(self.arrivalAirport)
        return lengthMeters
    
