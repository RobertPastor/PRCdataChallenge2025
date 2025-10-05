# -*- coding: UTF-8 -*-

'''
@since: Created on 26 aout 2014

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

create a KML output file that is readable in Google Earth
'''
import os
import xml.dom.minidom
import logging

class KmlOutput(object):
    
    fileName = ""
    kmlDoc = None
    documentElement = None
    
    def __init__(self, fileName, abortedFlight, aircraftICAOcode, AdepICAOcode, AdesICAOcode):
        
        self.AbortedFlight = abortedFlight
        self.AircraftICAOcode = aircraftICAOcode
        self.AdepICAOcode = AdepICAOcode
        self.AdesICAOcode = AdesICAOcode
        
        self.className = self.__class__.__name__

        self.fileName = str(fileName).replace('/', '-')
        if ( not ( str(self.fileName).endswith(".kml") )):
            self.fileName = self.fileName + ".kml"
        #sanity check : filename shall not contain "/"
        assert (not ('/' in self.fileName))
        
        self.fileName = fileName
        
        self.FilesFolder = os.path.dirname(__file__)
        
        logging.info ( self.className + ': file folder= {0}'.format(self.FilesFolder) )
        self.filePath = os.path.abspath( os.path.join ( self.FilesFolder , ".." , "ResultsFiles" , self.fileName ) )
        logging.info ( self.className + ': file path= {0}'.format(self.filePath) )
        
        # This constructs the KML document
        self.kmlDoc = xml.dom.minidom.Document()
        kmlElement = self.kmlDoc.createElementNS('http://earth.google.com/kml/2.2', 'kml')
        kmlElement.setAttribute('xmlns', 'http://earth.google.com/kml/2.2')
        kmlElement = self.kmlDoc.appendChild(kmlElement)
        self.documentElement = self.kmlDoc.createElement('Document')
        self.documentElement = kmlElement.appendChild(self.documentElement)
        
    def write(self, 
              name,
              LongitudeDegrees, 
              LatitudeDegrees, 
              AltitudeAboveSeaLevelMeters):
        
        #assert isinstance(name, (str))
        #assert isinstance(LongitudeDegrees, float)
        #assert isinstance(LatitudeDegrees, float)
        #assert isinstance(AltitudeAboveSeaLevelMeters, float)
        
        placemarkElement = self.kmlDoc.createElement('Placemark')
        
        nameElement = self.kmlDoc.createElement('name')
        nameElement.appendChild(self.kmlDoc.createTextNode(name))
        placemarkElement.appendChild(nameElement)
        
        pointElement = self.kmlDoc.createElement('Point')
        placemarkElement.appendChild(pointElement)
        
        extrudeElement = self.kmlDoc.createElement('extrude')
        extrudeElement.appendChild(self.kmlDoc.createTextNode("1"))
        pointElement.appendChild(extrudeElement)
        
        altitudeModeElement = self.kmlDoc.createElement('altitudeMode')
        altitudeModeElement.appendChild(self.kmlDoc.createTextNode("absolute"))
        pointElement.appendChild(altitudeModeElement)
        
        coordinates = str(float(LongitudeDegrees))+","+str(float(LatitudeDegrees))+","+str(AltitudeAboveSeaLevelMeters)
        coorElement = self.kmlDoc.createElement('coordinates')
        coorElement.appendChild(self.kmlDoc.createTextNode(coordinates))
        pointElement.appendChild(coorElement)
        
        self.documentElement.appendChild(placemarkElement)
        

    ''' warning - this folder cleaning does not work in heroku '''
    def cleanKmlFolder(self):
        self.FilesFolder = os.path.dirname(__file__)
        self.FilesFolder =  os.path.join( self.FilesFolder , '..' , 'static' , 'kml')
        for f in os.listdir(self.FilesFolder):
            os.remove(os.path.join(self.FilesFolder, f))
        
        
    def close(self):
        ''' always write in the static kml folder '''
        #self.FilesFolder = os.path.dirname(__file__)
        #self.FilesFolder =  os.path.join( self.FilesFolder , '..' , 'static' , 'kml')
        
        #self.filePath = os.path.join( self.FilesFolder , self.fileName )
        #self.filePath = os.path.join( "/static/kml" , self.fileName)
        #self.filePath = "static/kml/" + self.fileName
        logging.info ( self.className + ': file path= {0}'.format(self.filePath) )
        kmlFile = open( self.filePath , 'wb')
        kmlXmlDocument = self.kmlDoc.toprettyxml('  ', newl = '\n', encoding = 'utf-8')
        kmlFile.write ( kmlXmlDocument )
        kmlFile.close()
        #return kmlXmlDocument
        
    def getFilePath(self):
        return self.filePath
    
    def getKmlFileName(self):
        return self.fileName
    
class KmlFileLike(KmlOutput):
    
    def close(self, memoryFile):
        indent="\t";
        addindent = ""
        newl="\n"
        encoding='utf-8'
        self.kmlDoc.writexml(memoryFile, indent, addindent, newl, encoding)
        
    