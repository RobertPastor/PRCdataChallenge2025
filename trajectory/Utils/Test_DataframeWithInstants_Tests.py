'''
Created on 23 nov. 2025

@author: robert
'''


import logging
import unittest
from tabulate import tabulate
from datetime import datetime
from datetime import timedelta
import pytz
from tabulate import tabulate

import pandas as pd
#============================================
class Test_Main(unittest.TestCase):

    def test_main_one(self):
        
        timestamp_list = []
        
        # create a datetime object representing March 1, 2023 at 9:30 AM
        start_datetime = datetime(2023, 3, 1, 9, 30)
        print (f"without timezone =  {start_datetime} " )
        print (f"with TimeZone = {start_datetime.replace(tzinfo=pytz.utc)}" )

        # create a timedelta object representing 3 hours and 15 minutes
        event_duration = timedelta(seconds=3)
        
        # add the duration to a start time to get an end time
        end_datetime = start_datetime + event_duration
        print (end_datetime)
        
        timestamp_list.append( start_datetime )
        timestamp_list.append( end_datetime )
        
        timestamp_Series = {'timestamp': timestamp_list}
        print ( timestamp_Series )
        df_timestamp = pd.DataFrame(timestamp_Series, index=[1,2])
        print ( df_timestamp )
        
        flight_id_Series = {"flight_id" : ["prc999888777","prc999888777"] }
        print ( flight_id_Series )
        df_flight_id = pd.DataFrame(flight_id_Series , index=[1,2])
        print ( df_flight_id)
        
        df = pd.merge(df_flight_id , df_timestamp , left_index=True, right_index=True)
        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))

        print("-"*120)
        print("with UTC")
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    unittest.main()
    