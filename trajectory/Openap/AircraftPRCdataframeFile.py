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
from trajectory.Guidance.GraphFile import Graph

class OpenapAircraftPRCchallenge(OpenapAircraftStateVector):
    
    lastValidLatitudeDegrees = 0.0
    lastValidLongitudeDegrees = 0.0
    lastValidAltitudeMSLmeters = 0.0

    def __init__(self , aircraftICAOcode):
        self.aircraftICAOcode = aircraftICAOcode
        super().__init__(aircraftICAOcode)
        
    def extractLatitudeDegrees(self , finalRoute , index ):
        assert type(index) == int
        try:
            if index < finalRoute.getNumberOfVertices():
                vertex = finalRoute.getVertex(index)
                wayPoint = vertex.getWeight()
                self.lastValidLatitudeDegrees = wayPoint.getLatitudeDegrees()
                return self.lastValidLatitudeDegrees 
        except:
            return self.lastValidLatitudeDegrees 
        return self.lastValidLatitudeDegrees 
    
    def extractLongitudeDegrees(self , finalRoute , index ):
        assert type(index) == int
        try:
            if index < finalRoute.getNumberOfVertices():
                vertex = finalRoute.getVertex(index)
                wayPoint = vertex.getWeight()
                self.lastValidLongitudeDegrees = wayPoint.getLongitudeDegrees()
                return self.lastValidLongitudeDegrees
        except:
            return self.lastValidLongitudeDegrees
        return self.lastValidLongitudeDegrees 
    
    def extractAltitudeMSLmeters(self , finalRoute, index ):
        assert type(index) == int
        try:
            if index < finalRoute.getNumberOfVertices():
                vertex = finalRoute.getVertex(index)
                wayPoint = vertex.getWeight()
                self.lastValidAltitudeMSLmeters = wayPoint.getAltitudeMeanSeaLevelMeters()
                return self.lastValidAltitudeMSLmeters
        except:
            return self.lastValidAltitudeMSLmeters
        return self.lastValidAltitudeMSLmeters
    
    def extractTrackCourseAngleDegrees(self , finalRoute, index):
        assert type(index) == int
        edge = finalRoute.getEdge(index)
        return edge.getBearingTailHeadDegrees()
    
    def computeRateOfClimbDescentFeetPerMinutes(self ):
        if (self.elapsedTimeSeconds - self.previousElapsedTimeSeconds)>0.0:
            altitudeDifferenceFeet = (self.altitudeMeanSeaLevelFeet - self.previousAltitudeMeanSeaLevelFeet)
            rateOfClimbDescentFeetSeconds= altitudeDifferenceFeet / (self.elapsedTimeSeconds - self.previousElapsedTimeSeconds)
            rateOfClimbDescentFeetMinute = rateOfClimbDescentFeetSeconds / 60.0
        else:
            rateOfClimbDescentFeetMinute = 0.0
        return rateOfClimbDescentFeetMinute

    def createAircraftPRCdataChallengeFlightDataframe(self , finalRoute ,abortedFlight  , aircraftICAOcode, 
                                                      flight_id , takeOffInstant):
        assert ( isinstance ( finalRoute , Graph ) )
        print (f"number of vertices =  {finalRoute.getNumberOfVertices()}" )
        print (f"number of state history items = {len(self.aircraftStateHistory)}" )
        
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
        latitude_degrees_list = []
        longitude_degrees_list = []
        track_degrees_list = []
        altitude_feet_list = []
        vertical_rate_feet_minutes_list = []
        TAS_knots_list = []
        groundSpeed_knots_list = []
        CAS_knots_list = []
        mach_list = []
        
        index = 0
        maxIndex = len(self.aircraftStateHistory)
        for stateVectorHistory in self.aircraftStateHistory:
            if index > (maxIndex - 10):
                break
            index_list.append( index )
            
            flight_id_list.append(self.flight_id)
            source_list.append("computed")
            typecode_list.append(aircraftICAOcode)
            
            ''' latitude and longitude from finalRoute '''
            latitude_degrees_list.append ( self.extractLatitudeDegrees( finalRoute, index) )
            longitude_degrees_list.append ( self.extractLongitudeDegrees( finalRoute, index) )
            track_degrees_list.append( self.extractTrackCourseAngleDegrees(finalRoute, index) )

            for elapsedTimeSeconds, valueList in stateVectorHistory.items():
                ''' access to each value in the state vector '''
                self.elapsedTimeSeconds = elapsedTimeSeconds
                ''' altitude meters '''
                self.altitudeMeanSeaLevelMeters = valueList[1]
                ''' TAS speeds meters per second '''
                self.trueAirSpeedMetersSecond = valueList[2]
                
            ''' insert values in the list '''
            event_duration = timedelta(seconds=self.elapsedTimeSeconds)
            end_datetime = self.takeOffInstant + event_duration
            timestamp_list.append( end_datetime )
            ''' altitude '''
            self.altitudeMeanSeaLevelFeet = self.altitudeMeanSeaLevelMeters * Meter2Feet
            altitude_feet_list.append(self.altitudeMeanSeaLevelFeet)
            ''' TAS '''
            trueAirSpeedKnots = self.trueAirSpeedMetersSecond * MeterSecond2Knots
            TAS_knots_list.append( trueAirSpeedKnots )
            ''' ground speed '''
            groundSpeed_knots_list.append( trueAirSpeedKnots )
            ''' CAS speeds'''
            CAS_knots_list.append( tas2cas ( tasKnots = trueAirSpeedKnots ,
                                                 altitude_feet = self.altitudeMeanSeaLevelFeet ,
                                                 deltaTemperatureCelsius = 0.0))
            ''' mach speed '''
            mach_list.append( tas2mach ( tasKnots = self.trueAirSpeedMetersSecond * MeterSecond2Knots ,
                                             altitude_feet = self.altitudeMeanSeaLevelFeet ,
                                             deltaTemperatureCelsius = 0.0))
            
            ''' compute rate of climb descent feet per minutes '''
            if ( index == 0 ):
                self.previousElapsedTimeSeconds = self.elapsedTimeSeconds
                self.previousAltitudeMeanSeaLevelFeet = self.altitudeMeanSeaLevelMeters * Meter2Feet
                vertical_rate = 0.0
            else:
                vertical_rate = self.computeRateOfClimbDescentFeetPerMinutes()
                self.previousElapsedTimeSeconds = self.elapsedTimeSeconds
                self.previousAltitudeMeanSeaLevelFeet = self.altitudeMeanSeaLevelMeters * Meter2Feet

            vertical_rate_feet_minutes_list.append( vertical_rate )
            
            index = index + 1

        df_timestamp         = pd.DataFrame({"timestamp"     : timestamp_list}         , index=index_list)
        df_flight_id         = pd.DataFrame({"flight_id"     : flight_id_list}         , index=index_list)
        df_latitude_degrees  = pd.DataFrame({"latitude"      : latitude_degrees_list}  , index=index_list)
        df_longitude_degrees = pd.DataFrame({"longitude"     : longitude_degrees_list}          , index=index_list)
        df_track_degrees     = pd.DataFrame({"track"         : track_degrees_list}              , index=index_list)
        df_altitude_feet     = pd.DataFrame({"altitude"      : altitude_feet_list}              , index=index_list)
        df_TAS_knots         = pd.DataFrame({"TAS"           : TAS_knots_list}                  , index=index_list)
        df_groundSpeed_knots = pd.DataFrame({"groundspeed"   : groundSpeed_knots_list}          , index=index_list)
        df_vertical_rate     = pd.DataFrame({"vertical_rate" : vertical_rate_feet_minutes_list} , index=index_list)
        df_CAS_knots         = pd.DataFrame({"CAS"           : CAS_knots_list}                  , index=index_list)
        df_mach              = pd.DataFrame({"mach"          : mach_list}              , index=index_list)
        df_source            = pd.DataFrame({"source"        : source_list}            , index=index_list)
        df_typecode          = pd.DataFrame({"typecode"      : typecode_list}          , index=index_list)

        df = pd.merge(df_flight_id , df_timestamp , left_index=True, right_index=True)
        df = pd.merge(df , df_longitude_degrees   , left_index=True, right_index=True)
        df = pd.merge(df , df_latitude_degrees    , left_index=True, right_index=True)
        df = pd.merge(df , df_altitude_feet       , left_index=True, right_index=True)
        df = pd.merge(df , df_groundSpeed_knots   , left_index=True, right_index=True)
        df = pd.merge(df , df_track_degrees       , left_index=True, right_index=True)
        df = pd.merge(df , df_vertical_rate       , left_index=True, right_index=True)
        df = pd.merge(df , df_mach                , left_index=True, right_index=True)
        df = pd.merge(df , df_typecode            , left_index=True, right_index=True)
        df = pd.merge(df , df_TAS_knots           , left_index=True, right_index=True)
        df = pd.merge(df , df_CAS_knots           , left_index=True, right_index=True)
        df = pd.merge(df , df_source              , left_index=True, right_index=True)

        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        for columnName in ["flight_id","source","typecode"]:
            df[columnName] = df[columnName].astype(str)

        for columnName in ["altitude","TAS","CAS","mach"]:
            df[columnName] = df[columnName].astype(float)

        ''' suppress index '''
        print(df.dtypes)
        print ("take off instant = {0}".format(takeOffInstant) )
        print(tabulate(df[-10:], headers='keys', tablefmt='grid' , showindex=False , ))
        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))

        return df