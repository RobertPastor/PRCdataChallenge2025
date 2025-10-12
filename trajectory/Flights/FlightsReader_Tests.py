'''
Created on 9 oct. 2025

@author: rober
'''


import logging
import unittest
import pandas as pd
import os
from trajectory.Flights.FlightsReader import FlightsDatabase
from tabulate import tabulate


#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        logging.basicConfig(level=logging.INFO)

        print("---------------- test_main_one  ----------------")
        
        logging.info("Read Flight data")
        flightsDatabase = FlightsDatabase()
        assert flightsDatabase.readSomeTrainFiles(testMode = True) == True
        
    def test_main_two(self):
        logging.basicConfig(level=logging.INFO)

        print("---------------- test_main_two  ----------------")
        
        logging.info("Read Flight data")
        flightsDatabase = FlightsDatabase()
        assert flightsDatabase.readSomeTrainFiles(testMode = True) == True
        
        assert flightsDatabase.checkFlightsTrainHeaders()
        
    def test_main_three(self):
        logging.basicConfig(level=logging.INFO)

        print("---------------- test_main_three  ----------------")
        print("show aircraft type code distribution ")
        
        fileName = "prc770822360.parquet"
        flightsDatabase = FlightsDatabase()
        df = flightsDatabase.readOneTrainFile(fileName)
        
        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))


        print("---------show aircraft type code distribution ")
        logging.info( str ( df['aircraft_type_code'].value_counts()))
                
        print("---------------- show nulls  ----------------")
        logging.info ( str ( df.isnull().sum() ))

        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(pd. __version__)
    
    unittest.main()
        