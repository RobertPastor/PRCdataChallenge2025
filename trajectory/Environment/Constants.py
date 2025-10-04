'''
Created on 17 déc. 2020

@author: robert

Define constants common to all modules

'''

gravity_meter_square_seconds = 9.80665 

Meter2Feet = 3.2808399 # one meter equals 3.28 feet
#Meter2Feet = 3.2808399 # one meter approx == 3 feet (3 feet 3⅜ inches)

Feet2Meter = 0.3048 # one feet equals 0.3048 meters

Meter2NauticalMiles = 0.000539956803 # One Meter = 0.0005 nautical miles

NauticalMiles2Meter = 1852.0 # in meters
NauticalMiles2Meters = 1852.0 # in meters

Knots2MetersSeconds   = 0.514444444 # meters / second
Knots2MetersPerSecond = 0.514444444 # meters / second
Knot2MetersPerSecond  = 0.514444444 # meters per second

MeterPerSecond2Knots = 1.9438444924406
MeterSecond2Knots    = 1.9438444924406

MeterSeconds2FeetMinutes = 196.85039370078738

# mass

Kilogram2Pounds = 2.20462262
Kilogram2Pounds = 2.20462262 # 1 kilogram = 2.204 lbs

KeroseneLiter2Kilograms = 0.817
KeroseneKilogram2Liter = 1.2345679012345678

# minimal radius of the final turn -> /flight-profile/trajectory/Guidance/TurnLegFile.py
FinalArrivalTurnRadiusNauticalMiles = 5.0
ThreeDegreesGlideSlope = 3.0

DescentGlideSlopeThreeDegrees = 3.0
DescentGlideSlopeDistanceNauticalMiles = 10.0

MaxRateOfClimbFeetPerMinutes = 2000.0
MaxRateOfDescentFeetPerMinutes = -2200.0

Kerosene_kilo_to_US_gallons = 0.33
US_gallon_to_US_dollars = 3.25

GravityMetersPerSquareSeconds = 9.81

MinFlightLevel = 15.0
MaxFlightLevel = 450.0

''' 23rd July 2023 - Reduced Climb Power - see BADA manual 3.10 '''
reducedClimbPowerJetCoeffMaxValue = 0.15

ConstantClimbRampLengthNauticalMiles = 5.0 # Nautical Miles
GlideSlopeStart2TouchDownNauticalMiles = 5.0 # 5 Nmfrom start of glide slope to runway touch down

RollingFrictionCoefficient = 0.035

ConstantTaxiSpeedCasKnots = 5.0 # Knots CAS ???

EarthRadiusMeters     = 6378135.0 # earth’s radius in meters
EarthMeanRadiusMeters = 6378135.0 # earth’s radius in meters

MaxSpeedNoiseRestrictionsKnots = 250.0
MaxSpeedNoiseRestrictionMeanSeaLevelFeet = 10000.0

FeetMinutes2MetersSeconds = 0.00508