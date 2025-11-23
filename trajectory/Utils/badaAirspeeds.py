'''
Created on 22 nov. 2025

@author: robert
'''
import logging

''' 
https://github.com/eurocontrol-bada/pybada
'''

from pyBADA import atmosphere as atm

from trajectory.Environment.Constants import Knot2MetersPerSecond , Feet2Meter,\
    MeterSecond2Knots

def tas2mach( tasKnots, altitude_feet , deltaTemperatureCelsius ):
    tasMetersPerSeconds = tasKnots * Knot2MetersPerSecond
    altitude_meters = altitude_feet * Feet2Meter
    ''' expected altitude in meters '''
    theta = atm.theta ( h = altitude_meters , deltaTemp = deltaTemperatureCelsius)
    return atm.tas2Mach( v = tasMetersPerSeconds , theta = theta )


''' standard atmosphere, temperature delta provided '''
def tas2cas ( tasKnots , altitude_feet , deltaTemperatureCelsius ):
    tasMetersPerSeconds = tasKnots * Knot2MetersPerSecond
    ''' expected altitude in meters '''
    altitude_meters = altitude_feet * Feet2Meter
    sigma = atm.sigma ( h = altitude_meters , deltaTemp = deltaTemperatureCelsius )
    delta = atm.delta ( h = altitude_meters , deltaTemp = deltaTemperatureCelsius )
    casMetersPerSeconds = atm.tas2Cas( tas = tasMetersPerSeconds , delta =  delta , sigma = sigma )
    return casMetersPerSeconds * MeterSecond2Knots


def mach2tas( mach , altitude_feet, deltaTemperatureCelsius ):
    ''' expected altitude in meters '''
    altitude_meters = altitude_feet * Feet2Meter
    theta = atm.theta( h = altitude_meters , deltaTemp = deltaTemperatureCelsius )
    tasMetersPerSeconds = atm.mach2Tas( mach, theta )
    return tasMetersPerSeconds * MeterSecond2Knots


def mach2cas ( mach , altitude_feet, deltaTemperatureCelsius ):
    altitude_meters = altitude_feet * Feet2Meter
    theta = atm.theta( h = altitude_meters , deltaTemp = deltaTemperatureCelsius )
    delta = atm.delta ( h = altitude_meters , deltaTemp = deltaTemperatureCelsius )
    sigma = atm.sigma ( h = altitude_meters , deltaTemp = deltaTemperatureCelsius )
    casMetersPerSeconds = atm.mach2Cas ( Mach = mach , theta = theta , delta = delta , sigma = sigma)
    return casMetersPerSeconds * MeterSecond2Knots
    
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    tasKnots = 350.0 
    altitude_feet = 29000.0
    altitude_feet = 29000.0
    deltaTemperatureCelsius = 0.0
    
    mach = tas2mach( tasKnots = tasKnots, altitude_feet =  altitude_feet ,  deltaTemperatureCelsius = deltaTemperatureCelsius ) 
    print(f"Input TAS {tasKnots} knots - at {altitude_feet} feet -> mach = {mach}")
    
    cas = tas2cas ( tasKnots = tasKnots, altitude_feet =  altitude_feet , deltaTemperatureCelsius = deltaTemperatureCelsius  )
    print(f"Input TAS {tasKnots} knots - at {altitude_feet} feet -> CAS = {cas} knots ")

    tas = mach2tas ( mach = mach , altitude_feet = altitude_feet, deltaTemperatureCelsius = deltaTemperatureCelsius )
    print(f"Input mach {mach} - at {altitude_feet} feet -> TAS = {tas} knots ")
    
    cas = mach2cas ( mach = mach , altitude_feet = altitude_feet, deltaTemperatureCelsius = deltaTemperatureCelsius )
    print(f"Input mach {mach} - at {altitude_feet} feet -> CAS = {cas} knots ")

