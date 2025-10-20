'''
Created on 18 oct. 2025

@author: robert
'''


import logging
import unittest
import pandas as pd
import os
from pathlib import Path
from trajectory.Flights.FlightsReader import FlightsDatabase
from tabulate import tabulate

#============================================
class Test_Main(unittest.TestCase):
    
    def test_visit_train_flights(self):
        pass
    
        fligthsDatabase = FlightsDatabase()
        trainFlightsFolderStr = fligthsDatabase.getTrainFlightsFolderPathStr()
        trainFlightsFolderPath = Path(trainFlightsFolderStr)
        print ( trainFlightsFolderStr )
        onlyTrainfileNames = [f for f in os.listdir(trainFlightsFolderPath) if os.path.isfile(os.path.join(trainFlightsFolderPath, f))]
        count = 0
        listOfFlightIds = []
        for index , trainFileName in enumerate(onlyTrainfileNames):
            print ( str( trainFileName) )
            count = count + 1
            fligthsDatabase.readOneTrainFile(trainFileName)
            
            listOfFlightIds.append( fligthsDatabase.getFlightId() )
            if index > 10:
                break
            
        print("number of files = " + str(count))
        print("list of flight Ids : " + str(listOfFlightIds ))
        print("size of list of flight Identifierss = " + str( len ( listOfFlightIds )))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(pd. __version__)
    
    unittest.main()