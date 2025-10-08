'''
Created on 8 oct. 2025

@author: robert
'''


class FlightListClass(object):
    
    def __init__(self , flight_date , aircraft_type):
        pass
        self.className = self.__class__.__name__
    
        self.flight_date = flight_date
        self.aircraft_type = aircraft_type
        
        
    def getFlightDate(self):
        return self.flight_date
    
    def getAircraftType(self):
        return self.aircraft_type
    