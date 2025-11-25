'''
Created on 12 nov. 2024

@author: robert
'''
from trajectory.Openap.AircraftFuelFlowFile import OpenapAircraftFuelFlow

import logging
# create logger
logger = logging.getLogger()


class OpenapAircraftMass(OpenapAircraftFuelFlow):

    takeOfMassKilograms = 0.0 
    initialMassKilograms = 0.0
    currentMassKilograms = 0.0
    fuelCapacityKilograms = 0.0
    
    referenceMassKilograms = 0.0
    minimumMassKilograms = 0.0
    maximumMassKilograms = 0.0
    
    def __init__(self , aircraftICAOcode ):
        
        logger.setLevel(logging.INFO)
        self.className = self.__class__.__name__
        
        super().__init__(aircraftICAOcode)
        
        self.maximumTakeOffMassKilograms   = self.aircraft['mtow']
        self.maxLandingMassKilograms       = self.aircraft['mlw']
        self.operatingEmptyWeightKilograms = self.aircraft['oew']* 0.75
        self.referenceMassKilograms        = self.maximumTakeOffMassKilograms 
        self.minimumMassKilograms          = self.aircraft['oew'] 
        
    def getReferenceMassKilograms (self):
        return self.referenceMassKilograms
        
    def setInitialMassKilograms(self, initialMassKilograms):
        #logger.info ( self.className + " --- set initial mass = {0} kilograms".format(initialMassKilograms))
        self.takeOfMassKilograms  = initialMassKilograms
        self.initialMassKilograms = initialMassKilograms
        self.currentMassKilograms = initialMassKilograms
        
    def getCurrentMassKilograms(self):
        #logger.info ( self.className + " --- current mass = {0:.2f} kilograms".format(self.currentMassKilograms))
        return self.currentMassKilograms
    
    def setAircraftMassKilograms(self, aircraftMassKilograms ):
        ''' @TODO add check that current mass not lower to minimum mass '''
        if ( aircraftMassKilograms < self.operatingEmptyWeightKilograms ):
            raise ValueError("Error - aircraft mass {0} -> lower than Operating Empty weight = {1} kilograms".format( aircraftMassKilograms , self.operatingEmptyWeightKilograms ))
        self.currentMassKilograms = aircraftMassKilograms
    
    def getTakeOffMassKilograms(self):
        return self.initialMassKilograms
        
    def getInitialMassKilograms(self):
        return self.initialMassKilograms
        
    def getMinimumMassKilograms(self):
        return self.minimumMassKilograms
    
    def getMaximumTakeOffMassKilograms(self):
        return self.maximumTakeOffMassKilograms
    
    