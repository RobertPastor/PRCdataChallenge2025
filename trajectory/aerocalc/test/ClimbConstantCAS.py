'''
Created on 11 oct. 2024

@author: robert
'''

import unittest
import os

from trajectory.Environment.Atmosphere import Atmosphere
from trajectory.aerocalc.airspeed import tas2cas, cas2tas

class Test_Main(unittest.TestCase):
    
    def test_001(self):
        pass
    
        for xAltDecaFeet in range (1000, 31000 , 100):
            
            print ( xAltDecaFeet  )




if __name__ == '__main__':
    unittest.main()
    