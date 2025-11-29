'''
Created on 10 nov. 2025

@author: robert
'''


from trajectory.Guidance.GeographicalPointFile import GeographicalPoint

import logging
import unittest
import pandas as pd
from trajectory.Guidance.WayPointFile import WayPoint
#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        

        logging.basicConfig(level=logging.INFO)
        logging.info("------------ Waypoint test one----------------")

        
        
if __name__ == '__main__':
    logging.basicConfig()
    unittest.main()