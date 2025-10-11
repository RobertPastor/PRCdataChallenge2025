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
from tabulate import tabulate

from scipy import stats

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
    
    def test_main_zero(self):
        ''' check that there are no ZEROs '''
        
        logging.basicConfig(level=logging.INFO)
        logging.info("---------------- test for null values  ----------------")
        
        fuelDatabase = FuelDatabase()
        assert fuelDatabase.readFuelTrain() == True
        assert fuelDatabase.readFuelRank() == True
        
        df = fuelDatabase.getFuelTrainDataframe()
        logging.info ( str ( df.isnull().sum()  ))
        ''' check that sum of all column sums are still null '''
        assert ( df.isnull().sum().sum() == 0 )
        
        df = fuelDatabase.getFuelRankDataframe()
        logging.info ( str ( df.isnull().sum() ))
        ''' check that sum of all column sums are still null '''
        assert ( df.isnull().sum().sum() == 0 )


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
        
        #logging.info( str ( fuelDatabase.getFuelTrainDataframe().sample(10) ))
        df = fuelDatabase.getFuelTrainDataframe()

        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))

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
        ''' fuel burnt is NULL every where in the Rank data '''
        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        print ( str(  list ( df )) )

    def test_main_four(self):
        
        logging.basicConfig(level=logging.INFO)
        logging.info("---------------- analyse fuel flow distribution  ----------------")
        
        fuelDatabase = FuelDatabase()
        
        assert fuelDatabase.readFuelTrain() == True
        assert fuelDatabase.checkFuelTrainHeaders() == True
        
        df = fuelDatabase.getFuelTrainDataframe()
        '''
        plt.hist( df['fuel_kg'] , bins = 100)
        plt.title("Histogram - Fuel Kg - Fuel Train")
        
        plt.ylabel('count of occurrences')
        plt.xlabel('Fuel burnt Kg')
        
        plt.show()
        
        logging.info("---------------- identify outliers  ----------------")

        plt.boxplot(df['fuel_kg'])
        plt.title("Boxplot to Identify Outliers - Fuel Train")
        plt.ylabel('Fuel burnt Kg')
        plt.show()
        
        logging.info("---------------- show statistics  ----------------")

        # Calculate statistics
        #data = df.loc[(df['fuel_kg'] > 200000.0) & (df['time_diff_seconds'] > 3.0 * 3600.0 )]
        
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
        plt.title('Data with Min, Max, and Standard Deviation - Fuel Train')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Show the plot
        #plt.show()
        

        # Plot x versus y
        x = df['time_diff_seconds']
        y = df['fuel_kg']
        plt.plot(x, y, marker='o', label='y = 2x')
        
        # Add labels and title
        plt.xlabel('time_diff_seconds')
        plt.ylabel('fuel_burnt_kg')
        plt.title('delta time versus fuel burnt kg')
        
        # Add a legend
        plt.legend()
        # Show the plot
        plt.show()  
        '''
        
    def test_main_five(self):
        
        logging.basicConfig(level=logging.INFO)
        logging.info("---------------- analyze fuel distribution  ----------------")
        
        fuelDatabase = FuelDatabase()
        
        assert fuelDatabase.readFuelTrain() == True
        assert fuelDatabase.checkFuelTrainHeaders() == True
        df = fuelDatabase.getFuelTrainDataframe()
        
        # Pretty print the DataFrame
        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        
    def test_rank_distribution_time_differences(self):
        '''
        logging.basicConfig(level=logging.INFO)
        logging.info("---------------- analyze fuel distribution  ----------------")
        
        fuelDatabase = FuelDatabase()
        assert fuelDatabase.readFuelRank() == True
        assert fuelDatabase.checkFuelRankHeaders() == True
        
        df = fuelDatabase.getFuelRankDataframe()
        
        logging.info("---------------- identify outliers  ----------------")

        df = df[(df['time_diff_seconds'] < 600.0 )]

        plt.boxplot(df['time_diff_seconds'])
        plt.title("Boxplot to Identify Outliers - time diff seconds - Fuel Rank")
        plt.ylabel('time_diff_seconds')
        plt.show()
        
        logging.info("---------------- identify outliers  ----------------")
        
        df = fuelDatabase.getFuelRankDataframe()
        df = df[(df['time_diff_seconds'] < 600.0 )]

        plt.hist( df['time_diff_seconds'] , bins = 100)
        plt.title("Histogram - Time diff seconds - Fuel Rank")
        plt.xlabel('time_diff_seconds')
        plt.ylabel('occurrences')

        plt.show()
        '''
        
    def test_stats(self):
        
        logging.basicConfig(level=logging.INFO)
        logging.info("---------------- analyze statistics   ----------------")
        
        fuelDatabase = FuelDatabase()
        assert fuelDatabase.readFuelTrain() == True
        assert fuelDatabase.checkFuelTrainHeaders() == True
        
        df = fuelDatabase.getFuelTrainDataframe()
        ''' filter fuel_burnt kg lower to 2 tons '''
        #df = df[(df['fuel_flow_kg_sec'] < 2.0 * 1000.0)]
                # Find min and max of column 'A'
        min_value = df['fuel_flow_kg_sec'].min()
        max_value = df['fuel_flow_kg_sec'].max()
        
        print(f"Minimum value in column 'fuel_flow_kg_sec': {min_value}")
        print(f"Maximum value in column 'fuel_flow_kg_sec': {max_value}")

        print ( str ( df.shape ))
        print ( str ( list ( df.shape )))

        df['fuel_flow_kg_sec_z'] = np.abs( stats.zscore ( df['fuel_flow_kg_sec'] ) )
        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        
        # Find min and max of column 'A'
        min_value = df['fuel_flow_kg_sec_z'].min()
        max_value = df['fuel_flow_kg_sec_z'].max()
        
        print(f"Minimum value in column 'fuel_burnt_kg_z': {min_value}")
        print(f"Maximum value in column 'fuel_burnt_kg_z': {max_value}")
        
        print ( str ( df.shape ))
        print ( str ( list ( df )))
        
        ''' filter outliers '''
        df = df[df['fuel_flow_kg_sec_z'] <= 3.0 ]
        
        ''' commercial airliner 2500 kg / hour '''
        #max_fuel_flow_kg_sec  = 2500.0 / ( 60. * 60. )
        max_fuel_flow_kg_sec  = 10000.0 / ( 60. * 60. )
        print ("max fuel flow kg per seconds = {0}".format (max_fuel_flow_kg_sec))
        df = df[df['fuel_flow_kg_sec'] <= max_fuel_flow_kg_sec ]
        
        print ( str ( df.shape ))
        print ( str ( list ( df )))
        
        plt.hist( df['fuel_flow_kg_sec'] , bins = 100)
        plt.title("Histogram - Fuel flow Kg per seconds - Fuel Train")
        
        plt.ylabel('count of occurrences')
        plt.xlabel('Fuel flow Kg per seconds')
        
        plt.show()
        
    def test_fuel_extension_with_flight_start(self):
        
        logging.basicConfig(level=logging.INFO)
        logging.info("---------------- analyze fuel extended with flight start   ----------------")
        
        fuelDatabase = FuelDatabase()
        assert fuelDatabase.readFuelTrain() == True
        assert fuelDatabase.checkFuelTrainHeaders() == True
        
        assert fuelDatabase.extendFuelTrainWithFlightTakeOff()
        
        df = fuelDatabase.getFuelTrainDataframe()
        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        
        assert fuelDatabase.readFuelRank() == True
        assert fuelDatabase.extendFuelRankFithFlightTakeOff()
        
        df = fuelDatabase.getFuelRankDataframe()
        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))


        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print(pd. __version__)
    
    unittest.main()