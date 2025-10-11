'''
Created on 8 oct. 2025

@author: robert
'''

import logging
import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['figure.figsize'] = (16,6)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

from trajectory.Fuel.FuelReader import FuelDatabase

# Set the option to display all columns
pd.options.display.max_columns = None

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        logging.basicConfig(level=logging.INFO)

        logging.info("---------------- test_main_one  ----------------")

        fuelDatabase = FuelDatabase()
        
        assert fuelDatabase.readFuelTrain() == True
        
    def test_main_two(self):
        logging.basicConfig(level=logging.INFO)

        logging.info("---------------- test_main_two  ----------------")
        
        logging.info("Read Fuel Train")

        fuelDatabase = FuelDatabase()
        
        assert fuelDatabase.readFuelTrain() == True
        assert fuelDatabase.checkFuelTrainHeaders() == True
        
        logging.info( str ( fuelDatabase.getFuelTrainDataframe().sample(10) ))
        logging.info ( str ( fuelDatabase.getFuelTrainDataframe().isnull().sum() ))

        logging.info ( str ( fuelDatabase.getFuelTrainDataframe().dtypes ))
        
    def test_main_three(self):
        
        logging.basicConfig(level=logging.INFO)
        logging.info("---------------- test_main_three  ----------------")

        logging.info("Read Fuel Rank")
        fuelDatabase = FuelDatabase()
        
        assert fuelDatabase.readFuelRank() == True
        assert fuelDatabase.checkFuelRankHeaders() == True
        
        df = fuelDatabase.getFuelRankDataframe()
        print ( str ( df.shape ))
        print ( str ( df.sample(10) ))
        print ( str(  list ( df )) )
        
    def test_main_four(self):
        
        logging.basicConfig(level=logging.INFO)
        logging.info("---------------- analyse fuel distribution  ----------------")
        
        fuelDatabase = FuelDatabase()
        
        assert fuelDatabase.readFuelTrain() == True
        assert fuelDatabase.checkFuelTrainHeaders() == True
        df = fuelDatabase.getFuelTrainDataframe()
        plt.hist( df['fuel_kg'] , bins = 100)
        plt.show()
        
        logging.info("---------------- identify outliers  ----------------")

        plt.boxplot(df['fuel_kg'])
        plt.title("Boxplot to Identify Outliers")
        plt.show()
        
        logging.info("---------------- show statistics  ----------------")

        # Calculate statistics
        data = df['fuel_kg']
        data_min = np.min(data)
        data_max = np.max(data)
        data_std = np.std(data)
        
        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.plot(data, label='Data', marker='o', linestyle='-', alpha=0.7)
        
        # Annotate min, max, and std
        plt.axhline(data_min, color='blue', linestyle='--', label=f'Min: {data_min:.2f}')
        plt.axhline(data_max, color='red', linestyle='--', label=f'Max: {data_max:.2f}')
        plt.axhline(data.mean(), color='green', linestyle='-', label=f'Mean: {data.mean():.2f}')
        plt.fill_between(range(len(data)), data.mean() - data_std, data.mean() + data_std, 
                         color='yellow', alpha=0.2, label=f'Std Dev: ±{data_std:.2f}')
        
        # Add labels and legend
        plt.title('Data with Min, Max, and Standard Deviation')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Show the plot
        plt.show()
        

        # Plot x versus y
        x = df['time_diff_seconds']
        y = df['fuel_kg']
        plt.plot(x, y, marker='o', label='y = 2x')
        
        # Add labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('X versus Y Plot')
        
        # Add a legend
        plt.legend()
        
        # Show the plot
        plt.show()  
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print(pd. __version__)
    
    unittest.main()