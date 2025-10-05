'''
Created on 23 juil. 2025

@author: robert

pure document version of KmlOutput
'''

import os
import xml.dom.minidom
import logging

class KmlOutputPureDocument(object):
    
    kmlDoc = None
    documentElement = None
    
    def __init__(self, abortedFlight, aircraftICAOcode, AdepICAOcode, AdesICAOcode):
        
        self.AbortedFlight = abortedFlight
        self.AircraftICAOcode = aircraftICAOcode
        self.AdepICAOcode = AdepICAOcode
        self.AdesICAOcode = AdesICAOcode
        
        self.className = self.__class__.__name__

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
        
    def retrieveBytesLikeObject(self):
        return self.kmlDoc.toprettyxml('  ', newl = '\n', encoding = 'utf-8')
        