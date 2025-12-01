'''
Created on 22 nov. 2025

@author: robert
'''

import logging
from time import  sleep
from trajectory.FlightList.MAIN_SUB_FlightTrajectoriesToRebuild import FlightIdsToRebuild
import platform
import tensorflow as tf
import pandas as pd
import numpy as np
from tabulate import tabulate
from trajectory.Utils.utils import readNumberOfCPUs
from pathos.multiprocessing import Pool

MaxFlightIdsInQueue =  5
Static_Object = None

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("python version = " + platform.python_version())
    logging.info("tensorflow version = " + tf.__version__)
    logging.info("pandas version = " + pd. __version__)
    logging.info("numpy version = " + np. __version__)
        
    logging.basicConfig(level=logging.DEBUG)
    errorsDict = {}

    train_rank_final = "rank"
    aircraftICAOcodeToFilter = "A359"
    
    flightIdsToRebuildObject = FlightIdsToRebuild (train_rank_final)
    assert flightIdsToRebuildObject.readFlighIdsToRebuild()
    
    Static_Object = flightIdsToRebuildObject
    
    flight_ids_list_toRebuild = flightIdsToRebuildObject.getFlightIdsListToRebuild()
    flight_ids_list_toRebuild_length = len(flight_ids_list_toRebuild)
    logging.info(f"size of the flight ids to rebuild list = {flight_ids_list_toRebuild_length}")
    #errorsDict = flightIdsToRebuildObject.rebuildAllFlightIds(aircraftICAOcodeToFilter)
    
    #print("---------errors-----------")
    #for key, value in errorsDict.items():
    #    print ( "flight id = {0} - error = {1}".format(key, value))
    
    aircraftICAOcodeToFilter = "A359"
    
    data = [(flight_id, "A359") for flight_id in flight_ids_list_toRebuild]
    print (data)
    
    nbCPUs = readNumberOfCPUs()
    logging.info(f"number of CPUs = {nbCPUs}")
    
    def call_method( *args):
        method , flight_id , aircraftICAOcodeToFilter = args[0]
        print(f"launching class method with arguments  = {flight_id} - {aircraftICAOcodeToFilter}")
        """Call the given method with provided arguments."""
        return method( flight_id , aircraftICAOcodeToFilter )
    
    counter = 0
    data = []
    for flight_id in flight_ids_list_toRebuild:
        counter = counter + 1
        if counter < 20:
            data.append( (flightIdsToRebuildObject.rebuildOneFlightId , flight_id, "A359") )
            
    results = []
    # do an asynchronous map, then get the results
    with Pool(processes=nbCPUs) as pool:
        data = data
        result  = pool.apply_async(call_method, data)
        results.append(result)

    # need to wait for all processes finish
    while True:
        sleep(1)
        # catch exception if results are not ready yet
        try:
            ready = [result.ready() for result in results]
            successful = [result.successful() for result in results]
        except Exception:
            continue
        # exit loop if all tasks returned success
        if all(successful):
            break
        # raise exception reporting exceptions received from workers
        if all(ready) and not all(successful):
            raise Exception(f'Workers raised following exceptions {[result._value for result in results if not result.successful()]}')
    
    print("[Main] All flight ids are rebuild - tasks completed.")
    