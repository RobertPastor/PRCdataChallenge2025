'''
Created on 16 nov. 2025

@author: robert

'''
import math
import numpy as np
from trajectory.aerocalc.airspeed import  mach_alt2cas
from trajectory.aerocalc.airspeed import mach2tas 
from trajectory.aerocalc.airspeed import  tas2cas
from trajectory.Environment.Constants import Knots2MetersPerSecond , MetersToNauticalMiles


class TensorFlowSpeedBaseClass(object):
    pass

    def __init__(self):
        pass
    
    def isNotNoneAndNotNan(self , value ):
        return not (value is None) and (value != np.nan) and not( math.isnan ( value ))
    
    def compute_TAS_KnotsfromMach_atFuelStart(self , row):
        
        mach = row['aircraft_mach_at_fuel_start']
        aircraft_altitude_ft = row['aircraft_altitude_ft_at_fuel_start']
        ''' TAS computed from Java is correct - CAS is erroneous '''
        TAS = row['aircraft_TAS_at_fuel_start']
        groundSpeedKnots = row['aircraft_groundspeed_kt_at_fuel_start']

        if self.isNotNoneAndNotNan(mach) :
            if self.isNotNoneAndNotNan( aircraft_altitude_ft ):
                return mach2tas ( mach=mach, altitude=aircraft_altitude_ft , temp='std' ,alt_units='ft', speed_units='kt')
        else:
            pass
            #if self.isNotNoneAndNotNan(groundSpeedKnots):
            #    ''' here TODO - take into account the wind '''
            #   return groundSpeedKnots
        return np.nan

    def compute_TAS_KnotsfromMach_atFuelEnd(self , row):
        
        mach = row['aircraft_mach_at_fuel_end']
        aircraft_altitude_ft = row['aircraft_altitude_ft_at_fuel_end']
        ''' TAS computed from Java is correct - CAS is erroneous '''
        TAS = row['aircraft_TAS_at_fuel_end']
        groundSpeedKnots = row['aircraft_groundspeed_kt_at_fuel_end']

        if self.isNotNoneAndNotNan(mach) :
            if self.isNotNoneAndNotNan( aircraft_altitude_ft ):
                return mach2tas ( mach=mach, altitude=aircraft_altitude_ft )
        else:
            pass
            #if self.isNotNoneAndNotNan(groundSpeedKnots):
            #    ''' @TODO here take into account the wind speed and direction '''
            #    return groundSpeedKnots
        return np.nan

        ''' compute CAS from mach '''
    def compute_CAS_KnotsfromMach_atFuelStart(self, row ):
        
        mach = row['aircraft_mach_at_fuel_start']
        aircraft_altitude_ft = row['aircraft_altitude_ft_at_fuel_start']
        ''' TAS computed from Java is correct - CAS is erroneous '''
        CAS = row['aircraft_CAS_at_fuel_start']
        groundSpeedKnots = row['aircraft_groundspeed_kt_at_fuel_start']

        if self.isNotNoneAndNotNan(mach) :
            if self.isNotNoneAndNotNan( aircraft_altitude_ft ):
                ''' assumption is that altitude is always provided -> no altitude missings content '''
                return mach_alt2cas ( mach = mach, altitude = aircraft_altitude_ft , alt_units='ft', speed_units='kt')
        else:
            pass
            #if self.isNotNoneAndNotNan(groundSpeedKnots) and self.isNotNoneAndNotNan( aircraft_altitude_ft ):
            #    tas = groundSpeedKnots
            #    if ( tas < 661.48):
            #        return tas2cas( tas = tas , altitude = aircraft_altitude_ft , temp = 'std' , alt_units='ft', speed_units='kt')
        return np.nan
    
    def compute_CAS_KnotsfromMach_atFuelEnd(self, row ):
        
        mach = row['aircraft_mach_at_fuel_end']
        aircraft_altitude_ft = row['aircraft_altitude_ft_at_fuel_end']
        ''' TAS computed from Java is correct - CAS is erroneous '''
        CAS = row['aircraft_CAS_at_fuel_end']
        groundSpeedKnots = row['aircraft_groundspeed_kt_at_fuel_end']
        
        if self.isNotNoneAndNotNan(mach) :
            if self.isNotNoneAndNotNan( aircraft_altitude_ft ):
                ''' assumption is that altitude is always provided -> no altitude missings content '''
                return  mach_alt2cas ( mach = mach,altitude = aircraft_altitude_ft , alt_units = 'ft' , speed_units = 'kt')
        else:
            pass
            #if self.isNotNoneAndNotNan(groundSpeedKnots) and self.isNotNoneAndNotNan( aircraft_altitude_ft ):
            #    tas = groundSpeedKnots
            #    if ( tas < 661.48):
            #        return tas2cas( tas = tas , altitude = aircraft_altitude_ft , temp = 'std' , alt_units='ft', speed_units='kt')
        return np.nan
    
    def computeDistanceFlownNm(self , row ):
        fuel_start_instant = row['start']
        fuel_end_instant = row['end']
        duration_seconds = abs ((fuel_end_instant - fuel_start_instant).total_seconds())
        groundSpeedStartKnots = row['aircraft_groundspeed_kt_at_fuel_start']
        groundSpeedEndKnots = row['aircraft_groundspeed_kt_at_fuel_end']
        groundSpeedAverageKnots = abs ( groundSpeedStartKnots + groundSpeedEndKnots ) / 2.0
        groundSpeedAverageMetersPerSecond = groundSpeedAverageKnots * Knots2MetersPerSecond
        DistanceFlownMeters = groundSpeedAverageMetersPerSecond * duration_seconds
        return DistanceFlownMeters * MetersToNauticalMiles
    
    ''' use python aerocal method to compute TAS and CAS from mach when TAS or CAS are null '''
    def computeTASnCASfromMachOrGroundSpeed(self, df):
        # Machine epsilon for single precision (32-bit)
        df['aircraft_TAS_at_fuel_start'] = df.apply( self.compute_TAS_KnotsfromMach_atFuelStart , axis = 1)
        df['aircraft_TAS_at_fuel_end']   = df.apply( self.compute_TAS_KnotsfromMach_atFuelEnd   , axis = 1)
        
        df['aircraft_CAS_at_fuel_start'] = df.apply( self.compute_CAS_KnotsfromMach_atFuelStart , axis = 1)
        df['aircraft_CAS_at_fuel_end']   = df.apply( self.compute_CAS_KnotsfromMach_atFuelEnd   , axis = 1)
        
        ''' distance flown using ground speed between fuel start and fuel end '''
        df['aircraft_distance_flown_start_end'] = df.apply( self.computeDistanceFlownNm , axis = 1)
        
        return df