'''
Created on 6 mai 2025

@author: robert
'''

from trajectory.Openap.AircraftFlightPhasesFile import OpenapAircraftFlightPhases
from trajectory.Guidance.WayPointFile import WayPoint


class OpenapAircraftMiscelleaneous(OpenapAircraftFlightPhases):
    pass

    def __init__(self, aircraftICAOcode):
        
        self.className = self.__class__.__name__
        self.aircraftICAOcode = aircraftICAOcode
        
        super().__init__(aircraftICAOcode)
        
    def getMaximumNumberOfPassengers(self):
        return self.aircraft['pax']['max']
        
        
    def getLandingLengthMeters(self):
        self.LandingLengthMetersDict = self.wrap.landing_distance()
        self.LandingLengthMeters     = self.LandingLengthMetersDict['default'] * 1000.0
        #logging.info( self.className + " - landing length = {0} meters".format(self.LandingLengthMeters))
        return self.LandingLengthMeters
    
    def setTargetApproachWayPoint(self , approachWayPoint):
        ''' this point is the end of the descent '''
        ''' it is the top of the last turn before the descent glide slope to the arrival runway '''
        assert ( approachWayPoint and isinstance(approachWayPoint, WayPoint) )
        self.approachWayPoint = approachWayPoint
        
    def getTargetApproachWayPoint(self):
        ''' it is the top of the last turn before the descent glide slope to the arrival runway '''
        return self.approachWayPoint
    
    ''' this point is the end of the descent glide slope '''
    def setArrivalRunwayTouchDownWayPoint(self , touchDownWayPoint ):
        assert ( touchDownWayPoint and isinstance(touchDownWayPoint, WayPoint) )
        self.touchDownWayPoint = touchDownWayPoint
        
    def getArrivalRunwayTouchDownPoint(self):
        return self.touchDownWayPoint