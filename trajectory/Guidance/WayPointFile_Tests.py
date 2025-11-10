'''
Created on 10 nov. 2025

@author: robert
'''

import logging
import unittest
import pandas as pd
from trajectory.Guidance.WayPointFile import WayPoint
#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        
        print("------------ Waypoint test one----------------")

        logging.basicConfig(level=logging.INFO)
        
        
        
if __name__ == '__main__':
    logging.basicConfig()
    unittest.main()