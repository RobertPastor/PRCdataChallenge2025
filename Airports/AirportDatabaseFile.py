# -*- coding: UTF-8 -*-

'''
Created on 6 September 2014

@author: PASTOR Robert

        Written By:
                Robert PASTOR 
                @Email: < robert [--DOT--] pastor0691 (--AT--) gmail [--DOT--] com >

        @http://trajectoire-predict.monsite-orange.fr/ 
        @copyright: Copyright 2015 Robert PASTOR 

        This program is free software; you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation; either version 3 of the License, or
        (at your option) any later version.
 
        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.
 
        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.

this file builds the airport database
http://openflights.org/data.html 

As of January 2012, the OpenFlights Airports Database contains 6977 airports spanning the globe, as shown in the map above. Each entry contains the following information:
Airport ID     Unique OpenFlights identifier for this airport.
Name     Name of airport. May or may not contain the City name.
City     Main city served by airport. May be spelled differently from Name.
Country     Country or territory where airport is located.
IATA/FAA     3-letter FAA code, for airports located in Country "United States of America".
3-letter IATA code, for all other airports.
Blank if not assigned.
ICAO     4-letter ICAO code.
Blank if not assigned.
Latitude     Decimal degrees, usually to six significant digits. Negative is South, positive is North.
Longitude     Decimal degrees, usually to six significant digits. Negative is West, positive is East.
Altitude     In feet.
Timezone     Hours offset from UTC. Fractional hours are expressed as decimals, eg. India is 5.5.
DST     Daylight savings time. One of E (Europe), A (US/Canada), S (South America), O (Australia), Z (New Zealand), N (None) or U (Unknown). See also: Help: Time

The data is ISO 8859-1 (Latin-1) encoded, with no special characters.

Note: Rules for daylight savings time change from year to year and from country to country. 
The current data is an approximation for 2009, built on a country level. 
Most airports in DST-less regions in countries that generally observe DST (eg. AL, HI in the USA, NT, QL in Australia, parts of Canada) are marked incorrectly.
Sample entries

507,"Heathrow","London","United Kingdom","LHR","EGLL",51.4775,-0.461389,83,0,"E"
26,"Kugaaruk","Pelly Bay","Canada","YBB","CYBB",68.534444,-89.808056,56,-6,"A"
3127,"Pokhara","Pokhara","Nepal","PKR","VNPK",28.200881,83.982056,2712,5.75,"N"

'''
import os
import csv
import logging

from trajectory.Guidance.WayPointFile import Airport
from trajectory.Environment.Constants import Meter2Feet, Meter2NauticalMiles
from trajectory.Guidance.Haversine import points2distanceMeters

fieldNames = ["Airport ID", "Airport Name" , "City", "Country", "IATA/FAA", "ICAO Code",
                "LatitudeDegrees", "LongitudeDegrees", "AltitudeFeet", "TimeZone", "DST"]

feetToMeters = 0.3048 #meters

class AirportsDatabase(object):
    className = ''
    airportsDb = {}
    countriesDb = []
    FilePath = ""
    airportsFilesFolder = None

    def __init__(self):
        '''
        Open the file by calling open and then csv.DictReader.
        '''
        self.className = self.__class__.__name__
        ''' file name with extension '''
        self.FilePath = "Airports.csv"
        
        self.airportsFilesFolder = os.path.dirname(__file__)

        logging.info ( self.className + ': file folder= {0}'.format(self.airportsFilesFolder) )
        self.FilePath = os.path.join(self.airportsFilesFolder , self.FilePath)
        logging.info ( self.className + ': file path= {0}'.format(self.FilePath) )

    def read(self):
        try:
            dictReader = csv.DictReader(open(self.FilePath, encoding='utf-8'), fieldnames=fieldNames , delimiter=";")
            for row in dictReader:
                #print ( row )
                airport = {}
                for field in fieldNames:
                    #print ( field , row[field])
                    airport[field] = row[field]
                    if 'Country' in field:
                        country = row[field]
                        if not(country in self.countriesDb):
                            self.countriesDb.append(country)
                            
                #print ( row["ICAO Code"] )
                self.airportsDb[row["ICAO Code"]] = airport
            return True
        except Exception as e:
            logging.info ( e )
            return False
            
    def getAirportsFromCountry(self, Country = ''):
        for row in self.airportsDb.values():
            if row['Country'] == Country and len(row['ICAO Code'])>0 :
                airport = Airport(
                                Name=row['City']+'-'+row['Airport Name'] ,
                                LatitudeDegrees = row['LatitudeDegrees'] , 
                                LongitudeDegrees = row['LongitudeDegrees'] , 
                                fieldElevationAboveSeaLevelMeters = float(row['AltitudeFeet'])*feetToMeters,
                                ICAOcode = row['ICAO Code'] ,
                                Country = row['Country'] )
                yield airport
            
    def getAirports(self):
        for row in self.airportsDb.values():
            if len(row['ICAO Code'])>0 and len(row['Country'])>0:
                airport = Airport(
                                    Name = row['City']+'-'+row['Airport Name'] ,
                                    LatitudeDegrees = row['LatitudeDegrees'] , 
                                    LongitudeDegrees = row['LongitudeDegrees'] , 
                                    fieldElevationAboveSeaLevelMeters = float(row['AltitudeFeet'])*feetToMeters,
                                    ICAOcode = row['ICAO Code'] ,
                                    Country = row['Country'] )
                yield airport
    
    def dump(self):
        if self.airportsDb is None: 
            return
        for row in self.airportsDb:
            logging.info ( "{0} - {1}".format(self.className , row ))
    
    
    def getNumberOfAirports(self):
        if self.airportsDb is None: return 0
        return len(self.airportsDb.keys())
            
            
    def dumpCountry(self, Country="France"):
        if self.airportsDb is None: return
        for key , airport in self.airportsDb.items():
            if str(key).startswith('LF') and airport['Country'] == Country :
                logging.info (  airport )
               
                
    def getICAOCode(self, airportName = ''):
        if self.airportsDb is None: return ""
        airportsIcaoCodeList = []
        for key, airport in self.airportsDb.items():
            if str(airportName).lower() in str(airport["Airport Name"]).lower():
                airportsIcaoCodeList.append(key)
        if len(airportsIcaoCodeList)==1: return airportsIcaoCodeList[0]
        return airportsIcaoCodeList
    
    def getAirportElevationMeters(self, ICAOcode = ""):
        if self.airportsDb is None: return 0.0
        
        ''' internal airport is a dictionary '''
        for key, airportInternal in self.airportsDb.items():
            if key == ICAOcode:
                return float(airportInternal["AltitudeFeet"])*feetToMeters
        return 0.0
    
    def getAirportLatitudeDegrees(self, ICAOcode = ""):
        if self.airportsDb is None: return 0.0
        
        ''' internal airport is a dictionary '''
        for key, airportInternal in self.airportsDb.items():
            if key == ICAOcode:
                return float(airportInternal["LatitudeDegrees"])
        return 0.0
                 
    def getAirportLongitudeDegrees(self, ICAOcode = ""):
        if self.airportsDb is None: return 0.0
        
        ''' internal airport is a dictionary '''
        for key, airportInternal in self.airportsDb.items():
            if key == ICAOcode:
                return float(airportInternal["LongitudeDegrees"])
        return 0.0
    
    def isAirportICAOcodeInDB(self , ICAOcode = ""):
        if self.airportsDb is None: return None
        for key, airportInternal in self.airportsDb.items():
            if key == ICAOcode:
                return True
        return False
               
    def getAirportFromICAOCode(self, ICAOcode=""):
        if self.airportsDb is None: return None
        airport = None
        ''' internal airport is a dictionary '''
        for key, airportInternal in self.airportsDb.items():
            if key == ICAOcode:
                airport = Airport(
                                Name = airportInternal['City']+'-'+airportInternal["Airport Name"] ,
                                LatitudeDegrees = airportInternal["LatitudeDegrees"] , 
                                LongitudeDegrees = airportInternal["LongitudeDegrees"] , 
                                fieldElevationAboveSeaLevelMeters = float(airportInternal["AltitudeFeet"])*feetToMeters,
                                ICAOcode = ICAOcode,
                                Country = airportInternal['Country'] )
                return airport
        return None
        
    def getCountries(self):
        assert not( self.airportsDb is None)
        for country in self.countriesDb:
            yield country
            
    def computeDistanceNm(self, adepICAOcode, adesICAOcode):
        if self.airportsDb is None: return 0.0
        
        adepLatitudeDegrees = self.getAirportLatitudeDegrees(adepICAOcode)
        adepLongitudeDegrees = self.getAirportLongitudeDegrees(adepICAOcode)
        
        adesLatitudeDegrees = self.getAirportLatitudeDegrees(adesICAOcode)
        adesLongitudeDegrees = self.getAirportLongitudeDegrees(adesICAOcode)
        
        return points2distanceMeters( [adepLatitudeDegrees,adepLongitudeDegrees], [adesLatitudeDegrees, adesLongitudeDegrees])*Meter2NauticalMiles

        
if __name__ == '__main__':
    
    logging.basicConfig(level=logging.DEBUG)

    airportsDatabase = AirportsDatabase()
    ret = airportsDatabase.read()
    print ( ret )
    #for country in airportsDatabase.getCountries():
    #    print ( country )
        
    ICAOcode = "LFPG"
    airport = airportsDatabase.getAirportFromICAOCode(ICAOcode)
    if airport: 
        print ( "airport = {0} - elevation = {1} meters".format( airport.Name , airport.fieldElevationAboveSeaLevelMeters))
        
    ICAOcode = "MMMX"
    airport = airportsDatabase.getAirportFromICAOCode(ICAOcode)
    if airport: 
        print ( "airport = {0} - elevation = {1} meters".format( airport.Name , airport.fieldElevationAboveSeaLevelMeters))