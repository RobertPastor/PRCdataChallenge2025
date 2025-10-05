'''
Created on 12 nov. 2024

@author: robert

'''

#sys.path.append("C:/Users/rober/git/openap/") #replace PATH with the path to Foo

from openap import FuelFlow
import logging 
logger = logging.getLogger(__name__)

from trajectory.Openap.AircraftVerticalRate import OpenapAircraftVerticalRate

class OpenapAircraftFuelFlow(OpenapAircraftVerticalRate):
    
    aircraftICAOcode = ""

    def __init__(self , aircraftICAOcode ):
        
        self.className = self.__class__.__name__
        self.aircraftICAOcode = aircraftICAOcode
        
        super().__init__(aircraftICAOcode)
        self.fuelFlow = FuelFlow(ac=aircraftICAOcode)
        
    def getFuelFlowAtTakeOffKgSeconds(self, TASknots , aircraftAltitudeMSLfeet ):
        fuelFlowKgSeconds = self.fuelFlow.takeoff(tas = TASknots, alt = aircraftAltitudeMSLfeet, throttle=1)
        #logger.info(self.className + " - fuel flow takeOff {0:.2f} kilograms per second".format( fuelFlowKgSeconds ))
        return fuelFlowKgSeconds
    
    def getFuelFlowClimbKgSeconds(self , aircraftMassKilograms , TASknots , aircraftAltitudeMSLfeet , verticalRateFeetMinutes , accelerationMetersSecondsSquare ):
        fuelFlowKgSeconds = self.fuelFlow.enroute(mass=aircraftMassKilograms, tas=TASknots, alt=aircraftAltitudeMSLfeet, vs=verticalRateFeetMinutes, acc=accelerationMetersSecondsSquare, limit=True)
        #logger.info(self.className + " - fuel flow climb {0:.2f} kilograms per second".format( fuelFlowKgSeconds ))
        return fuelFlowKgSeconds
    
    def getFuelFlowCruiseKgSeconds(self , aircraftMassKilograms , TASknots , aircraftAltitudeMSLfeet , verticalRateFeetMinutes , accelerationMetersSecondsSquare ):
        fuelFlowKgSeconds = self.fuelFlow.enroute(mass=aircraftMassKilograms, 
                                                  tas=TASknots, 
                                                  alt=aircraftAltitudeMSLfeet, 
                                                  vs=verticalRateFeetMinutes, 
                                                  acc=accelerationMetersSecondsSquare, 
                                                  limit=True)
        #logger.info(self.className + " - fuel flow cruise {0:.2f} kilograms per second".format( fuelFlowKgSeconds ))
        return fuelFlowKgSeconds
    
    def getFuelFlowDescentKgSeconds(self , aircraftMassKilograms , TASknots, aircraftAltitudeMSLfeet , verticalRateFeetMinutes , accelerationMetersSecondsSquare):
        #fuelFlowKgSeconds = self.fuelFlow.at_thrust( acthr = thrustNewtons , alt=aircraftAltitudeMSLfeet, limit=True)
       
        fuelFlowKgSeconds = self.fuelFlow.enroute(mass=aircraftMassKilograms, 
                                                  tas=TASknots, 
                                                  alt=aircraftAltitudeMSLfeet, 
                                                  vs=verticalRateFeetMinutes,
                                                  acc=accelerationMetersSecondsSquare, 
                                                  limit=True)

        #logger.info(self.className + " - fuel flow descent {0:.2f} kilograms per second".format( fuelFlowKgSeconds ))
        return fuelFlowKgSeconds
    
    def computeFuelFlowDescentKilogramsSeconds(self , descentIdleThrustNewtons , aircraftAltitudeMSLfeet ):
        fuelFlowKgSeconds = self.fuelFlow.at_thrust ( acthr = descentIdleThrustNewtons , 
                                                      alt = aircraftAltitudeMSLfeet )
        #logging.info( self.className + " - descent idle fuel flow = {0:.2f} kg/s".format( fuelFlowKgSeconds ))
        return fuelFlowKgSeconds

    def computeFuelFlowKilogramsSeconds(self , 
                                        TASknots , 
                                        aircraftAltitudeMSLfeet , 
                                        aircraftMassKilograms           = 0.0, 
                                        verticalRateFeetMinutes         = 0.0, 
                                        accelerationMetersSecondsSquare = 0.0,
                                        thrustNewtons                   = 0.0):
        fuelFlowKilogramSeconds = None
        
        if self.isTakeOff():
            fuelFlowKilogramSeconds = self.getFuelFlowAtTakeOffKgSeconds( TASknots                = TASknots, 
                                                                          aircraftAltitudeMSLfeet = aircraftAltitudeMSLfeet )
            
        elif self.isInitialClimb():
            fuelFlowKilogramSeconds = self.getFuelFlowClimbKgSeconds( aircraftMassKilograms           = aircraftMassKilograms , 
                                                                      TASknots                        = TASknots , 
                                                                      aircraftAltitudeMSLfeet         = aircraftAltitudeMSLfeet , 
                                                                      verticalRateFeetMinutes         = verticalRateFeetMinutes, 
                                                                      accelerationMetersSecondsSquare = accelerationMetersSecondsSquare)

        elif self.isClimb():
            fuelFlowKilogramSeconds = self.getFuelFlowClimbKgSeconds( aircraftMassKilograms           = aircraftMassKilograms , 
                                                                      TASknots                        = TASknots , 
                                                                      aircraftAltitudeMSLfeet         = aircraftAltitudeMSLfeet , 
                                                                      verticalRateFeetMinutes         = verticalRateFeetMinutes, 
                                                                      accelerationMetersSecondsSquare = accelerationMetersSecondsSquare)
        elif self.isCruise():
            fuelFlowKilogramSeconds = self.getFuelFlowCruiseKgSeconds( aircraftMassKilograms            = aircraftMassKilograms ,  
                                                                       TASknots                         = TASknots , 
                                                                       aircraftAltitudeMSLfeet          = aircraftAltitudeMSLfeet , 
                                                                       verticalRateFeetMinutes          = verticalRateFeetMinutes, 
                                                                       accelerationMetersSecondsSquare  = accelerationMetersSecondsSquare)
            
        elif self.isDescent() or self.isApproach() or self.isLanding():
            fuelFlowKilogramSeconds = self.getFuelFlowCruiseKgSeconds( aircraftMassKilograms            = aircraftMassKilograms ,  
                                                                       TASknots                         = TASknots , 
                                                                       aircraftAltitudeMSLfeet          = aircraftAltitudeMSLfeet , 
                                                                       verticalRateFeetMinutes          = verticalRateFeetMinutes, 
                                                                       accelerationMetersSecondsSquare  = accelerationMetersSecondsSquare)

            #fuelFlowKilogramSeconds = self.getFuelFlowDescentKgSeconds( aircraftAltitudeMSLfeet          = aircraftAltitudeMSLfeet , 
            #                                                            thrustNewtons                    = thrustNewtons )
            
        else:
            fuelFlowKilogramSeconds = None

        return fuelFlowKilogramSeconds