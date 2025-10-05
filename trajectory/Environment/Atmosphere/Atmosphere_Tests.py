'''
Created on 15 août 2023

@author: robert
'''

import unittest
import os
import xlsxwriter
from trajectory.Environment.Atmosphere import Atmosphere
from trajectory.aerocalc.airspeed import tas2cas, cas2tas

class Test_Main(unittest.TestCase):
    
    def test_tas_2_cas(self):
        pass
        atmos = Atmosphere()
        tas = 320.0 # knots
        for xAltDecaMeters in range (0, 2000 , 100):
            
            print ( xAltDecaMeters * 10.0 )
            cas_atmosphere = atmos.tas2cas( tas = tas, altitude =  xAltDecaMeters * 10.0, temp = 'std', speed_units = 'kt', alt_units = 'm')
            cas_aerocals = tas2cas ( tas = tas, altitude = xAltDecaMeters * 10.0, temp = 'std', speed_units = "kt", alt_units= "m", temp_units='C')
            
            print (" tas 2 cas = {0} - tas 2 cas = {1}".format(  cas_atmosphere , cas_aerocals ) )
            
    def test_cas_2_tas(self):
        atmos = Atmosphere()
        cas = 220.0
        for xAltDecaMeters in range (0, 2000 , 100):
            print ( xAltDecaMeters * 10.0 )
            tas_atmosphere = atmos.cas2tas(cas = cas, altitudeMeters = xAltDecaMeters * 10.0 , speed_units = "kt" , altitude_units = "m")
            tas_aerocals = cas2tas ( cas = cas, altitude = xAltDecaMeters * 10.0, temp = 'std', speed_units = "kt", alt_units= "m", temp_units='C')
    
            print (" cas 2 tas = {0} - cas 2 tas = {1}".format(  tas_atmosphere , tas_aerocals ) )

    def test_mains(self):

        fileName = "Tabular Atmosphere.xlsx"
        print ( "===================Tabular Atmosphere start======================" )
        fileName = os.path.dirname(__file__) + os.path.sep  + fileName
        print ( fileName )
        
        workbook = xlsxwriter.Workbook(fileName)
        worksheet = workbook.add_worksheet('Temperature Degrees')
        
        RowIndex = 0
        ColumnIndex = 0
        for header in ['Altitude-Meters', 
                       'Temperature-Degrees',
                       'Temperature-Kelvins',
                       'Air-density-kg-per-cubic-meters',
                       'speed-of-sound',
                       'pressure-hecto-pascals']:
            worksheet.write(RowIndex, ColumnIndex, header)
            ColumnIndex += 1
        
        RowIndex += 1
        # =======================
        atmos = Atmosphere()
        for xAltDecaMeters in range (0,2000):
            
            AltitudeMeters = xAltDecaMeters*10.
            ColumnIndex = 0
            #print  ( '====================================' )
            
            worksheet.write( RowIndex, ColumnIndex, AltitudeMeters )
            ColumnIndex += 1
            
            worksheet.write( RowIndex, ColumnIndex, atmos.getTemperatureDegrees(AltitudeMeters))
            ColumnIndex += 1
    
            worksheet.write( RowIndex, ColumnIndex, atmos.getTemperatureKelvins(AltitudeMeters))
            ColumnIndex += 1
    
            worksheet.write( RowIndex, ColumnIndex, atmos.getAirDensityKilogramsPerCubicMeters(AltitudeMeters))
            ColumnIndex += 1
    
            worksheet.write( RowIndex, ColumnIndex, atmos.getSpeedOfSoundMetersPerSecond(AltitudeMeters))
            ColumnIndex += 1
            
            worksheet.write( RowIndex, ColumnIndex, atmos.getPressurePascals(AltitudeMeters)/100.)
    
            RowIndex += 1
        
        workbook.close()
        print ( "===================Tabular Atmosphere end======================" )
    
if __name__ == '__main__':
    unittest.main()