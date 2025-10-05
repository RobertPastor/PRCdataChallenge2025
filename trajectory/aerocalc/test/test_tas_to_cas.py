'''
Created on 11 oct. 2024

@author: robert
'''

import unittest
import sys

from trajectory.aerocalc.airspeed import tas2cas

def RE(value, truth):
    """ Return the absolute value of the relative error.
...."""

    return abs((value - truth) / truth)

class Test_Tas_to_Cas(unittest.TestCase):
        
    def test_002(self):
        
        altitudeFeet = 31000.0 # feet
        tasKnots = 470.0

        casKnots = tas2cas( tasKnots , altitudeFeet , temp = 'std' , speed_units= 'kt' , alt_units= 'ft')
        print ( "ISA - At {0} feet - with Tas Knots {1} - Cas Knots {2}\n".format( altitudeFeet , tasKnots , casKnots ) )
        
        expectedCasKnots = 297.767964
        
        self.assertTrue( RE ( casKnots , expectedCasKnots) <= 1e-5 )
        
    def test_003(self):
        
        altitudeFeet = 31000.0 # feet
        tasKnots = 470.0
        temperaturyDegreesCelsius = -46.4
        
        casKnots = tas2cas( tasKnots , altitudeFeet , temp = temperaturyDegreesCelsius , speed_units= 'kt' , alt_units= 'ft' , temp_units= "C")
        print ( "\nISA - At {0} feet - temperature Degrees Celsius {1} - Tas Knots {2} - Cas Knots {3}\n".format( altitudeFeet , temperaturyDegreesCelsius , tasKnots , casKnots ) )

        expectedCasKnots = 297.755543
        
        self.assertTrue( RE ( casKnots , expectedCasKnots) <= 1e-5 )
        
    def test_004(self):
        
        altitudeFeet = 31000.0 # feet
        tasKnots = 470.0
        temperaturyDegreesCelsius = -55.0
        
        casKnots = tas2cas( tasKnots , altitudeFeet , temp = temperaturyDegreesCelsius , speed_units= 'kt' , alt_units= 'ft' , temp_units= "C")
        print ( "\nISA - At {0} feet - temperature Degrees Celsius {1} - Tas Knots {2} - Cas Knots {3}\n".format( altitudeFeet , temperaturyDegreesCelsius , tasKnots , casKnots ) )
        
        print ( casKnots )

        expectedCasKnots = 304.1641995
        
        self.assertTrue( RE ( casKnots , expectedCasKnots) <= 1e-5 )
        
        



        
        
        

# add test suites to main test suite, so all test results are in one block
main_suite = unittest.TestSuite()
suite1 = unittest.makeSuite(Test_Tas_to_Cas)

main_suite.addTest(suite1)



# run main test suite
# if we run the main test suite, we get a line for each test, plus any
# tracebacks from failures.

unittest.TextTestRunner(verbosity=5).run(main_suite)