'''
Created on 16 nov. 2025

@author: robert

'''
from time import time

import unittest

import sys
import logging

from trajectory.Environment.Earth.EarthFile import Earth
from trajectory.Environment.Atmosphere.AtmosphereFile import Atmosphere

from trajectory.Environment.Airports.AirportDatabaseFile import AirportsDatabase
from trajectory.Guidance.WayPointFile import Airport

from trajectory.Environment.Runways.RunWaysDatabaseFile import RunWaysDataBase
from openap import prop

from trajectory.Openap.AircraftMainFile import OpenapAircraft

from trajectory.GuidanceOpenap.FlightPathOpenapFile import FlightPathOpenap


#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        
        pass
        available_acs = prop.available_aircraft(use_synonym=True)
        listAc = []
        for ac in available_acs:
            print ( str(ac).upper() )
            listAc.append(str(ac).upper())
            
        listAc.sort()
        print("========")
        for ac in listAc:
            print ( ac )
            

    