'''
Created on 6 nov. 2025

@author: robert

read flight parquet file one by one
using the train and rank fuel files
extend the flight Start and End timestamp with those needed to extrapolate

'''
from trajectory.Flights.FlightsReader import FlightsDatabase




class ExtendFlightTimeStampWithFuelStartEndAndInterpolate:
    """Exemple de classe simple"""
    
    def __init__(self ):
        pass        
    
    def loopThroughTrainFlightsFiles(self):
        pass
    
        flightsDatabase = FlightsDatabase()
        trainFolderPathStr = flightsDatabase.getTrainFlightsFolderPathStr()
        print ( trainFolderPathStr )
        
        from pathlib import Path

        pathlist = Path(trainFolderPathStr).rglob('*.parquet')
        for path in pathlist:
            # because path is object not string
            path_in_str = str(path)
            print(path_in_str)

        