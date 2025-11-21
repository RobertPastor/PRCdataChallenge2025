'''
Created on 16 nov. 2025

@author: robert

'''
import math
import numpy as np

from trajectory.Environment.Constants import MeterPerSecond2Knots, Meter2Feet
from trajectory.aerocalc.airspeed import  mach_alt2cas
from trajectory.aerocalc.airspeed import mach2tas 
from trajectory.aerocalc.airspeed import tas2cas
from trajectory.Environment.Constants import Knots2MetersPerSecond , MetersToNauticalMiles

from pyBADA import atmosphere as atm
from pyBADA.atmosphere import mach2Tas as Bada_mach2Tas
from pyBADA.atmosphere import mach2Cas as Bada_mach2Cas

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

        if self.isNotNoneAndNotNan(mach) and  self.isNotNoneAndNotNan( aircraft_altitude_ft ):
            return mach2tas ( mach=mach, altitude=aircraft_altitude_ft , temp='std' ,alt_units='ft', speed_units='kt')
        return TAS

    def compute_TAS_KnotsfromMach_atFuelEnd(self , row):
        
        mach = row['aircraft_mach_at_fuel_end']
        aircraft_altitude_ft = row['aircraft_altitude_ft_at_fuel_end']
        ''' TAS computed from Java is correct - CAS is erroneous '''
        TAS = row['aircraft_TAS_at_fuel_end']
        groundSpeedKnots = row['aircraft_groundspeed_kt_at_fuel_end']

        if self.isNotNoneAndNotNan(mach) and self.isNotNoneAndNotNan( aircraft_altitude_ft ):
            return mach2tas ( mach=mach, altitude=aircraft_altitude_ft ,temp='std' ,alt_units='ft', speed_units='kt')
        return TAS

        ''' compute CAS from mach '''
    def compute_CAS_KnotsfromMach_atFuelStart(self, row ):
        
        mach = row['aircraft_mach_at_fuel_start']
        aircraft_altitude_ft = row['aircraft_altitude_ft_at_fuel_start']
        ''' TAS computed from Java is correct - CAS is erroneous '''
        CAS = row['aircraft_CAS_at_fuel_start']
        groundSpeedKnots = row['aircraft_groundspeed_kt_at_fuel_start']

        if self.isNotNoneAndNotNan(mach) and self.isNotNoneAndNotNan( aircraft_altitude_ft ):
            ''' assumption is that altitude is always provided -> no altitude missings content '''
            try:
                return mach_alt2cas ( mach = mach, altitude = aircraft_altitude_ft , alt_units='ft', speed_units='kt')
            except:
                pass
        return CAS
    
    def compute_CAS_KnotsfromMach_atFuelEnd(self, row ):
        
        mach = row['aircraft_mach_at_fuel_end']
        aircraft_altitude_ft = row['aircraft_altitude_ft_at_fuel_end']
        ''' TAS computed from Java is correct - CAS is erroneous '''
        CAS = row['aircraft_CAS_at_fuel_end']
        groundSpeedKnots = row['aircraft_groundspeed_kt_at_fuel_end']
        
        if self.isNotNoneAndNotNan(mach) and self.isNotNoneAndNotNan( aircraft_altitude_ft ):
            ''' assumption is that altitude is always provided -> no altitude missings content '''
            try:
                return  mach_alt2cas ( mach = mach,altitude = aircraft_altitude_ft , alt_units = 'ft' , speed_units = 'kt')
            except:
                pass
        return CAS
    
    def computeDistanceFlownNm(self , row ):
        fuel_start_instant    = row['start']
        fuel_end_instant      = row['end']
        duration_seconds      = abs ((fuel_end_instant - fuel_start_instant).total_seconds())
        groundSpeedStartKnots = row['aircraft_groundspeed_kt_at_fuel_start']
        groundSpeedEndKnots     = row['aircraft_groundspeed_kt_at_fuel_end']
        groundSpeedAverageKnots = abs ( groundSpeedStartKnots + groundSpeedEndKnots ) / 2.0
        groundSpeedAverageMetersPerSecond = groundSpeedAverageKnots * Knots2MetersPerSecond
        DistanceFlownMeters               = groundSpeedAverageMetersPerSecond * duration_seconds
        return DistanceFlownMeters * MetersToNauticalMiles
    
    def compute_TAS_KnotsfromMach_Bada_atFuelStart(self, row):
        
        mach = row['aircraft_mach_at_fuel_start']
        aircraft_altitude_ft = row['aircraft_altitude_ft_at_fuel_start']
        aircraft_altitude_meters = aircraft_altitude_ft * Meter2Feet
        ''' BADA provides speed results in m/s '''
        if self.isNotNoneAndNotNan(mach) and self.isNotNoneAndNotNan( aircraft_altitude_ft ):
            # deviation from standard tamperature
            deltaTemp = 0.0
            ''' BADA provides speed results in m/s '''
            theta = atm.theta(aircraft_altitude_meters, deltaTemp)
            return Bada_mach2Tas ( mach , theta ) * MeterPerSecond2Knots
        return row['aircraft_TAS_at_fuel_start']
        
    def compute_TAS_KnotsfromMach_Bada_atFuelEnd(self, row):
        mach = row['aircraft_mach_at_fuel_end']
        aircraft_altitude_ft = row['aircraft_altitude_ft_at_fuel_end']
        aircraft_altitude_meters = aircraft_altitude_ft * Meter2Feet

        if self.isNotNoneAndNotNan(mach) and self.isNotNoneAndNotNan( aircraft_altitude_ft ):
            # deviation from standard tamperature
            deltaTemp = 0.0
            # theta Normalized temperature according to the ISA model
            theta = atm.theta(aircraft_altitude_meters, deltaTemp)
            return Bada_mach2Tas ( mach , theta ) * MeterPerSecond2Knots
        return row['aircraft_TAS_at_fuel_end']
        
    def compute_CAS_KnotsfromMach_Bada_atFuelStart(self , row):
        mach = row['aircraft_mach_at_fuel_start']
        aircraft_altitude_ft = row['aircraft_altitude_ft_at_fuel_start']
        aircraft_altitude_meters = aircraft_altitude_ft * Meter2Feet

        if self.isNotNoneAndNotNan(mach) and self.isNotNoneAndNotNan( aircraft_altitude_ft ):
            # deviation from standard tamperature
            deltaTemp = 0.0
            # theta: Normalized temperature
            # delta : normalized pressure
            # sigma: Normalized air density             
            theta , delta , sigma = atm.atmosphereProperties ( aircraft_altitude_meters , deltaTemp)
            
            '''' retunrs Cas in meters per seconds '''
            return Bada_mach2Cas ( mach , theta , delta, sigma ) * MeterPerSecond2Knots
        return row['aircraft_CAS_at_fuel_start']
        
    def compute_CAS_KnotsfromMach_Bada_atFuelEnd(self, row):
        mach = row['aircraft_mach_at_fuel_end']
        aircraft_altitude_ft = row['aircraft_altitude_ft_at_fuel_end']
        aircraft_altitude_meters = aircraft_altitude_ft * Meter2Feet

        if self.isNotNoneAndNotNan(mach) and self.isNotNoneAndNotNan( aircraft_altitude_ft ):
            deltaTemp = 0.0
            theta , delta , sigma = atm.atmosphereProperties ( aircraft_altitude_meters , deltaTemp)
            
            ''' returns CAS in meters per seconds '''
            return Bada_mach2Cas ( mach , theta , delta, sigma ) * MeterPerSecond2Knots
        return row['aircraft_CAS_at_fuel_end']
    
    ''' use python aerocal method to compute TAS and CAS from mach when TAS or CAS are null '''
    def computeTASnCASfromMachOrGroundSpeed(self, df):
        # Machine epsilon for single precision (32-bit)
        ''' using aero calc library '''
        df['aircraft_TAS_aerocalc_at_fuel_start'] = df.apply( self.compute_TAS_KnotsfromMach_atFuelStart , axis = 1)
        df['aircraft_TAS_aerocalc_at_fuel_end']   = df.apply( self.compute_TAS_KnotsfromMach_atFuelEnd   , axis = 1)
        
        df['aircraft_CAS_aerocalc_at_fuel_start'] = df.apply( self.compute_CAS_KnotsfromMach_atFuelStart , axis = 1)
        df['aircraft_CAS_aerocalc_at_fuel_end']   = df.apply( self.compute_CAS_KnotsfromMach_atFuelEnd   , axis = 1)
        
        ''' TAS from mach using BADA pyBADA '''
        df['aircraft_TAS_Bada_at_fuel_start'] = df.apply( self.compute_TAS_KnotsfromMach_Bada_atFuelStart , axis = 1)
        df['aircraft_TAS_Bada_at_fuel_end']   = df.apply( self.compute_TAS_KnotsfromMach_Bada_atFuelEnd   , axis = 1)
        
        ''' CAS from mach using BADA pyBADA '''
        df['aircraft_CAS_Bada_at_fuel_start'] = df.apply( self.compute_CAS_KnotsfromMach_Bada_atFuelStart , axis = 1)
        df['aircraft_CAS_Bada_at_fuel_end']   = df.apply( self.compute_CAS_KnotsfromMach_Bada_atFuelEnd   , axis = 1)
        
        ''' distance flown using ground speed between fuel start and fuel end '''
        df['aircraft_distance_flown_start_end'] = df.apply( self.computeDistanceFlownNm , axis = 1)
        
        return df