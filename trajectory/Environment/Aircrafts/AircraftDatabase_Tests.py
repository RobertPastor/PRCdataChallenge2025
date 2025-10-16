'''
Created on 21 oct. 2024

@author: robert
'''
import os
from trajectory.Environment.Aircrafts.FAAaircraftDatabaseFile import FaaAircraftDatabase

import logging
import unittest

import pandas as pd

#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):

        logging.basicConfig(level=logging.INFO)

        print ( "--- read aircraft database ---")
        
        faaAircraftDatabase = FaaAircraftDatabase()
        assert faaAircraftDatabase.exists()
        
        if ( faaAircraftDatabase.read()):
            
            for aircraft in ['A320' , 'A20N']:
                #print ("--------------")
                #print ( str ( aircraft ))
                massMTOW = faaAircraftDatabase.getMTOW_lb(aircraft)
                massMLAW = faaAircraftDatabase.getMALW_lb(aircraft)
                #print ( "MTOW {0} lb --- MLAW {1} lb ".format( massMTOW , massMLAW) )
                #print ( "MTOW {0} kilograms --- MLAW {1} kilograms ".format( massMTOW * lbToKilograms , massMLAW * lbToKilograms) )
                if ( faaAircraftDatabase.isICAOcodeExisting(str ( aircraft ) ) == False ):
                    print ( aircraft + " -not in FAA database")
                
    def test_main_two(self):
        faaAircraftDatabase = FaaAircraftDatabase()
        assert faaAircraftDatabase.exists()
        
        if ( faaAircraftDatabase.read()):
            
            for aircraft in ['A320' , 'A20N']:
                #print ("--------------")
                #print ( str ( aircraft ))
                massMTOW_lb = faaAircraftDatabase.getMTOW_lb(aircraft)
                print(aircraft , " massMTOW = {0}".format ( massMTOW_lb ))
                massMLAW_lb = faaAircraftDatabase.getMALW_lb(aircraft)
                print(aircraft , " massMTOW = {0}".format ( massMLAW_lb ))

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    
    print("pandas version = " + pd. __version__)
    
    unittest.main()