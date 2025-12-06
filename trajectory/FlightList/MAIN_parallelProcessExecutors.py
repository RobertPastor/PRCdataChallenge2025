'''
Created on 3 déc. 2025

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
from time import time , sleep

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

from queue import Queue
from concurrent.futures import ProcessPoolExecutor

from pathos.multiprocessing import ProcessingPool

def whileQueueIsEmpty():
    queueIsEmpty = False
    while queueIsEmpty == False:
        pass
    
    
def execute_method( *args ):
    
    module = __import__('trajectory.FlightList.MAIN_SUB_FlightTrajectoriesToRebuild')
    #func = getattr(module, 'rebuildOneFlightId')

    obj , flight_id , aircraftICAOcodeToFilter = args
    print ( dir ( obj ) )
    
    assert ( isinstance ( flight_id , str))
    assert ( isinstance ( aircraftICAOcodeToFilter , str))
    #return func( flight_id , aircraftICAOcodeToFilter)

def fillInQueueWithFlightIds(   flight_ids_list_toRebuild ,aircraftICAOcodeToFilter):
    assert isinstance ( aircraftICAOcodeToFilter , str )
    assert isinstance ( flight_ids_list_toRebuild , list )
    ''' use a multi processes consumer queue '''
    dataQueue = Queue(len(flight_ids_list_toRebuild))
    for flight_id in flight_ids_list_toRebuild:
        dataQueue.put(item=(  flight_id, aircraftICAOcodeToFilter) , block=True, timeout=None)
    
    print("queue is filled")
    return dataQueue


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("python version = " + platform.python_version())
    logging.info("tensorflow version = " + tf.__version__)
    logging.info("pandas version = " + pd. __version__)
    logging.info("numpy version = " + np. __version__)
        
    logging.basicConfig(level=logging.DEBUG)
    
    
    train_rank_final = "rank"
    aircraftICAOcodeToFilter = "A359"
    
    flightIdsToRebuildObject = FlightIdsToRebuild (train_rank_final)
    assert flightIdsToRebuildObject.readFlighIdsToRebuild()
    
    flight_ids_list_toRebuild = flightIdsToRebuildObject.getFlightIdsListToRebuild()
    flight_ids_list_toRebuild_length = len(flight_ids_list_toRebuild)
    logging.info(f"size of the flight ids to rebuild list = {flight_ids_list_toRebuild_length}")
    
    aircraftICAOcodeToFilter = "A359"
    #flight_ids_queue = fillInQueueWithFlightIds( flight_ids_list_toRebuild , aircraftICAOcodeToFilter)
    
    dataArgumentsList = []
    for flight_id in flight_ids_list_toRebuild:
        dataArgumentsList.append( [ flight_id, aircraftICAOcodeToFilter ] )
    
    nbCPUs = readNumberOfCPUs()
    pool = ProcessingPool(nodes=nbCPUs)
    
    try:
        # Map the bound method directly — pathos supports this
        results = pool.map(flightIdsToRebuildObject.rebuildOneFlightId, dataArgumentsList)

        print("Input:", dataArgumentsList)
        print("Output:", results)

    except Exception as e:
        print("Error during parallel execution:", e)

    finally:
        # Always close and join the pool
        pool.close()
        pool.join()
        pool.clear()
    