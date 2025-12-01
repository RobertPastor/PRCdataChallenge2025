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
    
