'''
Created on 12 nov. 2024

@author: robert
'''

''' Thrust expressed in Newtons '''


#sys.path.append("C:/Users/rober/git/openap/") #replace PATH with the path to Foo

from openap import Thrust

from trajectory.Openap.AircraftDragFile import OpenapAircraftDrag

import logging 
logger = logging.getLogger(__name__)


class OpenapAircraftThrust(OpenapAircraftDrag):

    def __init__(self, aircraftICAOcode ):
        self.className = self.__class__.__name__
        super().__init__(aircraftICAOcode)
        
        self.thrust = Thrust(ac=str( aircraftICAOcode ).lower() , eng=None)
        
    def getTakeOffThrustNewtons(self, tasKnots , altitudeMSLfeet ):
        takeOffThrustNewtons = self.thrust.takeoff(tas = tasKnots, alt = altitudeMSLfeet)
        #logger.info ( self.className + ': take off thrust = {0:.2f} newtons - tas = {1:.2f} knots at MSL altitude {2:.2f} feet'.format(takeOffThrustNewtons , tasKnots , altitudeMSLfeet) )
        return takeOffThrustNewtons
    
    def getClimbThrustNewtons(self , tasKnots, altitudeMSLfeet , rateOfClimbFeetMinutes ):
        climbThrustNewtons = self.thrust.climb( tas = tasKnots, alt = altitudeMSLfeet , roc = rateOfClimbFeetMinutes )
        #logger.info ( self.className + ': climb thrust = {0:.2f} newtons - tas = {1:.2f} knots at MSL altitude {2:.2f} feet'.format(climbThrustNewtons , tasKnots , altitudeMSLfeet) )
        return climbThrustNewtons
    
    def getCruiseThrustNewtons(self , tasKnots , altitudeMSLfeet):
        cruiseThrustNewtons = self.thrust.cruise ( tas = tasKnots , alt = altitudeMSLfeet)
        #logger.info ( self.className + ': cruise thrust = {0:.2f} newtons - tas = {1:.2f} knots at MSL altitude {2:.2f} feet'.format(cruiseThrustNewtons , tasKnots , altitudeMSLfeet) )
        return cruiseThrustNewtons
    
    def getDescentIdleThrustNewtons(self , tasKnots , altitudeMSLfeet):
        descentIdleThrustNewtons = self.thrust.descent_idle ( tas = tasKnots , alt = altitudeMSLfeet)
        #logger.info ( self.className + ': descent idle thrust = {0:.2f} newtons - tas = {1:.2f} knots at MSL altitude {2:.2f} feet'.format(descentIdleThrustNewtons , tasKnots , altitudeMSLfeet) )
        return descentIdleThrustNewtons

    def computeThrustNewtons(self , tasKnots, altitudeMSLfeet , rateOfClimbFeetMinutes = 0.0):
        thrustNewtons = None
        if self.isTakeOff():
            thrustNewtons = self.getTakeOffThrustNewtons( tasKnots        = tasKnots, 
                                                          altitudeMSLfeet = altitudeMSLfeet)
            #logging.info(self.className + " - takeoff thrust = {0:.2f} newtons".format( thrustNewtons ))
            
        elif self.isInitialClimb():
            thrustNewtons = self.getClimbThrustNewtons( tasKnots               = tasKnots ,  
                                                        altitudeMSLfeet        = altitudeMSLfeet , 
                                                        rateOfClimbFeetMinutes = rateOfClimbFeetMinutes)
        elif self.isClimb():
            thrustNewtons = self.getClimbThrustNewtons( tasKnots               = tasKnots ,  
                                                        altitudeMSLfeet        = altitudeMSLfeet , 
                                                        rateOfClimbFeetMinutes = rateOfClimbFeetMinutes)
        elif self.isCruise():
            thrustNewtons = self.getCruiseThrustNewtons( tasKnots        = tasKnots , 
                                                         altitudeMSLfeet = altitudeMSLfeet )
        elif self.isDescent() or self.isApproach() or self.isLanding():
            thrustNewtons = self.getDescentIdleThrustNewtons( tasKnots        = tasKnots , 
                                                              altitudeMSLfeet = altitudeMSLfeet )
            #logging.info(self.className + " - descent idle thrust = {0:.2f} newtons".format( thrustNewtons ))
            
        else:
            raise ValueError("not yet implemented")

        return thrustNewtons