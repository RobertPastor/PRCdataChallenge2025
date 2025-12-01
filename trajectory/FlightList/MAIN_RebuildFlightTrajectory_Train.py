'''
Created on 22 nov. 2025

@author: robert
'''

import platform
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import logging
from tabulate import tabulate
from pathlib import Path

from trajectory.FlightList.FlightListReader import FlightListDatabase
from trajectory.Flights.FlightsReader import FlightsDatabase
from trajectory.Guidance.WayPointFile import Airport 
from trajectory.Environment.Airports.AirportDatabaseFile import AirportsDatabase
from trajectory.Environment.Earth.EarthFile import Earth
from trajectory.Environment.Atmosphere.AtmosphereFile import Atmosphere
from trajectory.Openap.AircraftMainFile import OpenapAircraft

from trajectory.Environment.Runways.RunWaysDatabaseFile import RunWaysDataBase
from trajectory.GuidanceOpenap.FlightPathOpenapFile import FlightPathOpenap
from trajectory.Environment.WayPoints.WayPointsDatabaseFile import WayPointsDatabase

from trajectory.FlightList.MAIN_SUB_FlightTrajectoriesToRebuild import FlightIdsToRebuild
from trajectory.Utils.utils import readNumberOfCPUs


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("python version = " + platform.python_version())
    logging.info("tensorflow version = " + tf.__version__)
    logging.info("pandas version = " + pd. __version__)
    logging.info("numpy version = " + np. __version__)
        
    logging.basicConfig(level=logging.DEBUG)
    
    nbCPUs = readNumberOfCPUs()
    
    aircraftICAOcodeToFilter = None
    
    train_rank_final = "train"
    flightIdsToRebuild = FlightIdsToRebuild (train_rank_final)
    assert flightIdsToRebuild.readFlighIdsToRebuild()
    
    flight_ids_list_toRebuild = flightIdsToRebuild.getFlightIdsListToRebuild()

    errorsDict = {}
    #errorsDict = flightIdsToRebuild.rebuildAllFlightIds()
    
    print("---------errors-----------")
    for key, value in errorsDict.items():
        print ( "flight id = {0} - error = {1}".format(key, value))
        