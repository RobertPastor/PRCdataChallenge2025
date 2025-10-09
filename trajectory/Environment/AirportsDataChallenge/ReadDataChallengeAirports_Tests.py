'''
Created on 7 oct. 2025

@author: robert
'''




import logging
import unittest

from trajectory.Environment.AirportsDataChallenge.AirportsDataChallengeDatabaseFile import AirportsDataChallengeDatabase

#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        print("------------test_main_one----------------")

        logging.basicConfig(level=logging.INFO)
        
        logging.info("Read Data Challenge Airports")
        
        airportsDb = AirportsDataChallengeDatabase()
        assert airportsDb.read() == True
        
        if airportsDb.read():
            logging.info("Data Challenge Airports - correctly read")
        else:
            logging.error("Data Challenge Airports - read failed")
            
    def test_main_two(self):
        logging.basicConfig(level=logging.INFO)

        print("------------test_main_two----------------")

        airportsDb = AirportsDataChallengeDatabase()
        if airportsDb.read():
            ParisCDG = "LFPG"
            
            logging.info( airportsDb.getAirPort( ParisCDG ))
            assert ( not ( airportsDb.getAirPort( ParisCDG ) is None ))
            
    def test_main_three(self):
        
        logging.basicConfig(level=logging.INFO)

        print("------------test_main_three----------------")
        airportsDb = AirportsDataChallengeDatabase()
        if airportsDb.read():
            assert airportsDb.checkHeaders () == True
            logging.info("both expected and read column list are identical")
            
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()