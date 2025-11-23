'''
Created on 23 nov. 2025

@author: robert
'''

from trajectory.Openap.AircraftStateVectorFile import OpenapAircraftStateVector
import pandas as pd
import pytz
from trajectory.Utils.badaAirspeeds import tas2cas, tas2mach
from datetime import datetime , timedelta
from trajectory.Environment.Constants import Meter2Feet , MeterSecond2Knots
from tabulate import tabulate

class OpenapAircraftPRCchallenge(OpenapAircraftStateVector):
    pass

    def __init__(self , aircraftICAOcode):
        super().__init__(aircraftICAOcode)
    
    def createAircraftPRCdataChallengeFlightDataframe(self , abortedFlight  , aircraftICAOcode, 
                                                      flight_id , takeOffInstant):
        assert (type(abortedFlight) == bool )
        self.flight_id = flight_id
        self.takeOffInstant = takeOffInstant
        assert isinstance( takeOffInstant , datetime  )
        
        takeOffInstant_UTC = takeOffInstant.replace(tzinfo=pytz.utc)
        print(f"Current takeOff UTC time: {takeOffInstant_UTC}")
        
        index_list = []
        timestamp_list = []
        flight_id_list = []
        source_list = []
        typecode_list = []
        altitude_feet_list = []
        TAS_knots_list = []
        groundSpeed_knots_list = []
        CAS_knots_list = []
        mach_list = []
        maxIndex = 1000
        index = 0
        for stateVectorHistory in self.aircraftStateHistory:
            if (index > maxIndex):
                break
            index_list.append( index )
            index = index + 1
            for elapsedTimeSeconds, valueList in stateVectorHistory.items():
                ''' access to each value '''
                event_duration = timedelta(seconds=elapsedTimeSeconds)
                end_datetime = self.takeOffInstant + event_duration
                timestamp_list.append( end_datetime )
                
                flight_id_list.append(self.flight_id)
                source_list.append("computed")
                typecode_list.append(aircraftICAOcode)
                
                ''' altitude feet '''
                altitudeMeanSeaLevelMeters = valueList[1]
                altitude_feet_list.append(altitudeMeanSeaLevelMeters * Meter2Feet)
                
                ''' TAS speeds '''
                trueAirSpeedMetersSecond = valueList[2]
                TAS_knots_list.append( trueAirSpeedMetersSecond * MeterSecond2Knots )
                ''' ground speed '''
                groundSpeed_knots_list.append( trueAirSpeedMetersSecond * MeterSecond2Knots )
                
                ''' CAS speeds'''
                CAS_knots_list.append( tas2cas ( tasKnots = trueAirSpeedMetersSecond * MeterSecond2Knots ,
                                                 altitude_feet = altitudeMeanSeaLevelMeters * Meter2Feet ,
                                                 deltaTemperatureCelsius = 0.0))
                ''' mach speed '''
                mach_list.append( tas2mach ( tasKnots = trueAirSpeedMetersSecond * MeterSecond2Knots ,
                                             altitude_feet = altitudeMeanSeaLevelMeters * Meter2Feet ,
                                             deltaTemperatureCelsius = 0.0))
        
        df_timestamp         = pd.DataFrame({"timestamp"     : timestamp_list}         , index=index_list)
        df_flight_id         = pd.DataFrame({"flight_id"     : flight_id_list}         , index=index_list)
        df_altitude_feet     = pd.DataFrame({"altitude"      : altitude_feet_list}     , index=index_list)
        df_TAS_knots         = pd.DataFrame({"TAS"           : TAS_knots_list}         , index=index_list)
        df_groundSpeed_knots = pd.DataFrame({"groundspeed"   : groundSpeed_knots_list} , index=index_list)
        df_CAS_knots         = pd.DataFrame({"CAS"           : CAS_knots_list}         , index=index_list)
        df_mach              = pd.DataFrame({"mach"          : mach_list}              , index=index_list)
        df_source            = pd.DataFrame({"source"        : source_list}            , index=index_list)
        df_typecode          = pd.DataFrame({"typecode"      : typecode_list}          , index=index_list)

        df = pd.merge(df_flight_id , df_timestamp , left_index=True, right_index=True)        
        df = pd.merge(df , df_altitude_feet , left_index=True, right_index=True)        
        df = pd.merge(df , df_TAS_knots , left_index=True, right_index=True)        
        df = pd.merge(df , df_groundSpeed_knots , left_index=True, right_index=True)        
        df = pd.merge(df , df_CAS_knots , left_index=True, right_index=True)        
        df = pd.merge(df , df_mach , left_index=True, right_index=True)        
        df = pd.merge(df , df_source , left_index=True, right_index=True)        
        df = pd.merge(df , df_typecode , left_index=True, right_index=True)        

        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        for columnName in ["flight_id","source","typecode"]:
            df[columnName] = df[columnName].astype(str)

        for columnName in ["altitude","TAS","CAS","mach"]:
            df[columnName] = df[columnName].astype(float)

        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))
        return df