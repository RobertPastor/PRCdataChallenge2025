# -*- coding: UTF-8 -*-
'''
Created on Mar 18, 2015

@author: Robert PASTOR

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

'''
import logging
from datetime import datetime
import pandas as pd
from tabulate import tabulate

from trajectory.Guidance.WayPointFile import WayPoint, Airport

from trajectory.OutputFiles.KmlOutput import KmlOutput, KmlFileLike
from trajectory.OutputFiles.KmlOutputPureDocument import KmlOutputPureDocument
from trajectory.OutputFiles.GroundTrackOutput import GroundTrackOutput

class Vertex(object):
    
    def __init__(self, vertex):
        self.className = self.__class__.__name__
        self._vertex = vertex
        
    def getWeight(self):
        return self._vertex
    
    def __str__(self):
        return self.className + ': vertex= {0}'.format(str(self._vertex))

class Edge(object):
    _tail = None
    _head = None
    distanceTailHeadMeters = 0.0
    bearingTailHeadDegrees = 0.0
    
    def __init__(self, tail, head):
        self._tail = tail
        self._head = head
    
    def getTail(self):
        return self._tail
    
    def getHead(self):
        return self._head

    def getDistanceTailHeadMeters(self):
        if ( isinstance(self._tail, (WayPoint, Airport)) and isinstance(self._head, (WayPoint , Airport)) ):
            self.distanceTailHeadMeters = self._tail.getDistanceMetersTo(self._head)
        return self.distanceTailHeadMeters

    def getBearingTailHeadDegrees(self):
        if ( isinstance(self._tail, (WayPoint , Airport)) and isinstance(self._head, (WayPoint , Airport)) ):
            self.bearingTailHeadDegrees = self._tail.getBearingDegreesTo(self._head)
        return self.bearingTailHeadDegrees

class Graph(object):
    _vertex = []
    _edge = []
    lengthMeters = 0.0
    
    def __init__(self):
        self.className = self.__class__.__name__
        self._vertex = []
        self._edge = []
        self.lengthMeters = 0.0

    def __str__(self):
        return self.className + ': number of vertices= {0}'.format(len(self._vertex))

    def addGraph(self, otherGraph):
        ''' add the vertices of a another graph to self '''
        assert isinstance(otherGraph, Graph)
        for vertex in otherGraph._vertex:
            ''' add the vertex '''
            self.addVertex( vertex.getWeight() )
        
    def addVertex(self, *args):
        if len(args) == 1:
            ''' position in the oriented graph '''
            weight = args[0]
            self._vertex.append(Vertex(weight))
            ''' add edge here '''
            numberOfVertices = len(self._vertex)
            if numberOfVertices > 1:
                tail = self._vertex[numberOfVertices-2].getWeight()
                head = self._vertex[numberOfVertices-1].getWeight()
                self.addEdge(Edge(tail, head))
        else:
            assert (isinstance(args[0], int))
            index = args[0]
            if (index >= 0) and (index <= len(self._vertex)):
                weight = args[1]
                self._vertex.insert(index, Vertex(weight))
                ''' need to re build the list of Edges '''
                ''' remove edge with index -1 '''
                if index == 0:
                    self._edge.pop(0)
                else:
                    self._edge.pop(index-1)
                ''' need to rebuild two edges '''
                tail1 = self._vertex[index-1].getWeight()
                head1 = self._vertex[index].getWeight()
                self.insertEdge(index, Edge(tail1, head1))
                
                tail2 = self._vertex[index].getWeight()
                head2 = self._vertex[index+1].getWeight()
                self.insertEdge(index+1, Edge(tail2, head2))
            else:
                raise ValueError(self.className + ': insert index= {0] not in the limits 0..len'.format(index, len(self._vertex)))

    def getVertex(self, v):
        """
        (Graph, int) -> Vertex
        Returns the specified vertex of this graph.
        v is an ordered key related to the position in the ordered graph '''
        """
        assert isinstance(v, int)
        if v < 0 or v >= len(self._vertex):
            raise ValueError(self.className + ': getVertex: vertex index out of bounds!!!')
        return self._vertex[v]
    
    def getLastVertex(self):
        numberOfVertices = len(self._vertex)
        if numberOfVertices > 0:
            return self._vertex[numberOfVertices-1]
        return None
    
    def insertEdge(self, index, baseEdge):
        assert(isinstance(index, int))
        if (index >= 0) and (index <= len(self._edge)):
            if (isinstance(baseEdge, Edge)):
                ''' modify the list '''
                self._edge.insert(index, baseEdge)
                ''' update the graph length '''
                self.lengthMeters += baseEdge.getDistanceTailHeadMeters()
            else:
                raise ValueError('Graph: insert edge - edge must be of class BaseEdge !!!')
        else:
            raise ValueError('Graph: getVertex: vertex index out of bounds!!!')
    
    def addEdge(self, baseEdge):
        '''logging.debug 'Graph: add edge'''
        if (isinstance(baseEdge, Edge)):
            self._edge.append(baseEdge)
            ''' update the graph length '''
            self.lengthMeters += baseEdge.getDistanceTailHeadMeters()
        else:
            raise ValueError('Graph: add edge - edge must be of class BaseEdge !!!')
    
    def getLastEdge(self):
        numberOfEdges = len(self._edge)
        if  numberOfEdges > 0:
            return self._edge[numberOfEdges-1]
        return None

    def getEdge(self, position):
        '''
        (Graph, int) -> Edge
        '''
        assert isinstance(position, int)
        if position < 0 or position > len(self._edge):
            raise ValueError('Graph: getEdge: edge index out of bounds !!!')
        return self._edge[position]
    
    def getNumberOfVertices(self):
        """
        (Graph) -> int
        Returns the number of vertices in this graph.
        """
        return len(self._vertex)

    def getNumberOfEdges(self):
        """
        (Graph) -> int
        Returns the number of edges in this graph.
        """
        return len(self._edge)
    
    def getVertices(self):
        ''' returns an iterator on the vertices '''
        for vertex in self._vertex:
            yield vertex

    def getEdges(self):
        ''' returns an iterator on the edges '''
        for edge in self._edge:
            yield edge
            
    def hideSomeVertices(self, kmlFileLike , nbHidden):
        count = 0
        for vertex in self.getVertices():
            wayPoint = vertex.getWeight()
            if ( len(wayPoint.getName()) > 0):
                # if waypoint has a name -> reset counter
                count = 0
                kmlFileLike.write( wayPoint.getName(),
                                    wayPoint.getLongitudeDegrees(),
                                    wayPoint.getLatitudeDegrees(), 
                                    wayPoint.getAltitudeMeanSeaLevelMeters())
            else:
                count = count + 1
                if (count < nbHidden):
                    pass
                else:
                    # reset counter every nbHidden vertex
                    count = 0
                    kmlFileLike.write( wayPoint.getName(),
                                    wayPoint.getLongitudeDegrees(),
                                    wayPoint.getLatitudeDegrees(), 
                                    wayPoint.getAltitudeMeanSeaLevelMeters())
        return kmlFileLike
    
    def createKmlFileLike(self, memoryFile, abortedFlight, aircraftICAOcode, AdepICAOcode, AdesICAOcde):
        ''' create a memory file like to download '''
        self.AbortedFlight = abortedFlight
        self.AircraftICAOcode = aircraftICAOcode
        self.AdepICAOcode = AdepICAOcode
        self.AdesICAOcode = AdesICAOcde
        
        assert ( type(abortedFlight) == bool )
        if self.getNumberOfVertices() > 1:
            ''' need at least two vertices '''
            tail = self.getVertex(0)
            head = self.getVertex(self.getNumberOfVertices()-1)
            assert isinstance(tail.getWeight(), WayPoint)
            assert isinstance(head.getWeight(), WayPoint)
            tailWayPoint = tail.getWeight()
            headWayPoint = head.getWeight()
            
            strFileName = ""
            if abortedFlight:
                strFileName = "ABORTED-"
            strFileName += str(aircraftICAOcode) + "-" + AdepICAOcode + "-" + AdesICAOcde
            strFileName += "-" + tailWayPoint.getName()+'-'+headWayPoint.getName()
            ''' replace '''
            strFileName = str(strFileName).replace(' ', '-')
            strFileName += '-{0}.kml'.format(datetime.now().strftime("%d-%b-%Y-%Hh%Mm%S"))
            
            kmlFileLike = KmlFileLike( strFileName, abortedFlight, aircraftICAOcode, AdepICAOcode, AdesICAOcde)
            kmlFileLike = self.hideSomeVertices(kmlFileLike, 10)
            ''' this is where the xml / kml document is pushed into the StringIO '''
            kmlFileLike.close(memoryFile)
            ''' no need to return anything as the memoryFile is directly written in '''
            logging.debug ( "{0} - {1}".format(self.className , strFileName) )
        
        return  ValueError("GraphFile - createKmlOutputFile - number of vertices is 0")
    
    def createKmlXmlPureDocument(self, abortedFlight, aircraftICAOcode, AdepICAOcode, AdesICAOcde):
        self.AbortedFlight = abortedFlight
        self.AircraftICAOcode = aircraftICAOcode
        self.AdepICAOcode = AdepICAOcode
        self.AdesICAOcode = AdesICAOcde
        
        kmlOutputPureDocument = None
        
        assert ( type(abortedFlight) == bool )
        if self.getNumberOfVertices() > 1:
            ''' need at least two vertices '''
            #tail = self.getVertex(0)
            #head = self.getVertex(self.getNumberOfVertices()-1)
            #assert isinstance(tail.getWeight(), WayPoint)
            #assert isinstance(head.getWeight(), WayPoint)
            #tailWayPoint = tail.getWeight()
            #headWayPoint = head.getWeight()
            
            kmlOutputPureDocument = KmlOutputPureDocument( abortedFlight, aircraftICAOcode, AdepICAOcode, AdesICAOcde)
            kmlOutputPureDocument = self.hideSomeVertices(kmlOutputPureDocument, 10)

            for vertex in self.getVertices():
                wayPoint = vertex.getWeight()
                kmlOutputPureDocument.write(wayPoint.getName(),
                                    wayPoint.getLongitudeDegrees(),
                                    wayPoint.getLatitudeDegrees(), 
                                    wayPoint.getAltitudeMeanSeaLevelMeters())
            return kmlOutputPureDocument.retrieveBytesLikeObject()
    
        return  ValueError("GraphFile - createKmlOutputPureDocument - number of vertices is 0")
    
    def createKmlOutputFile(self, abortedFlight, aircraftICAOcode, 
                            AdepICAOcode, AdesICAOcde):
        self.AbortedFlight = abortedFlight
        self.AircraftICAOcode = aircraftICAOcode
        self.AdepICAOcode = AdepICAOcode
        self.AdesICAOcode = AdesICAOcde
        
        assert ( type(abortedFlight) == bool )
        if self.getNumberOfVertices() > 1:
            ''' need at least two vertices '''
            tail = self.getVertex(0)
            head = self.getVertex(self.getNumberOfVertices()-1)
            assert isinstance(tail.getWeight(), WayPoint)
            assert isinstance(head.getWeight(), WayPoint)
            tailWayPoint = tail.getWeight()
            headWayPoint = head.getWeight()
            
            strFileName = ""
            if abortedFlight:
                strFileName = "ABORTED-"
            strFileName += str(aircraftICAOcode) + "-" + AdepICAOcode + "-" + AdesICAOcde
            strFileName += "-" + tailWayPoint.getName()+'-'+headWayPoint.getName()
            ''' replace '''
            strFileName = str(strFileName).replace(' ', '-')
            strFileName += '-{0}.kml'.format(datetime.now().strftime("%d-%b-%Y-%Hh%Mm%S"))
            
            kmlOutputFile = KmlOutput(strFileName, abortedFlight, aircraftICAOcode, AdepICAOcode, AdesICAOcde)
            kmlOutputFile = self.hideSomeVertices(kmlOutputFile, 10)
            for vertex in self.getVertices():
                wayPoint = vertex.getWeight()
                kmlOutputFile.write(wayPoint.getName(),
                                    wayPoint.getLongitudeDegrees(),
                                    wayPoint.getLatitudeDegrees(), 
                                    wayPoint.getAltitudeMeanSeaLevelMeters())
            kmlOutputFile.close()
            logging.debug ( "{0} - {1}".format(self.className , strFileName) )
            return kmlOutputFile
    
        return ValueError("GraphFile - createKmlOutputFile - number of vertices is 0")
    
    def createPRCdataChallengeFlightDataframe(self , abortedFlight , flight_id, takeOffInstant ):
        assert (type(abortedFlight) == bool )
        
        flight_id_series = pd.Series(name="flight_id")
        timestamp_series = pd.Series(name="timestamp")
        
        maxVertexWritten = 10000
        vertexCounter = 0
        courseAngleDegrees = 0.0
        if self.getNumberOfVertices() > 1:
            ''' loop '''
            index = 0
            for vertex in self.getVertices():
                vertexCounter = vertexCounter + 1
                if vertexCounter > maxVertexWritten:
                    break
                ''' build an edge having two consecutive vertices as tail and head '''
                edge = None
                if index > 0:
                    edge = Edge(self.getVertex(index-1).getWeight(), self.getVertex(index).getWeight())
                if not (edge is None):
                    courseAngleDegrees = edge.getBearingTailHeadDegrees()
                    
                flight_id_series.add(flight_id, axis = 1)
                timestamp_series.add()
                
        df = pd.DataFrame()
        df = pd.concat([df, flight_id_series.to_frame()], ignore_index=True)
        df = pd.concat([df, timestamp_series.to_frame()], ignore_index=True)
        print(tabulate(df[:10], headers='keys', tablefmt='grid' , showindex=False , ))

        return df
    
    def createXlsxOutputFile(self, abortedFlight, aircraftICAOcode, AdepICAOcode, AdesICAOcode):
        assert (type(abortedFlight) == bool )
        
        if self.getNumberOfVertices() > 1:
            ''' need at least two vertices '''
            tail = self.getVertex(0)
            head = self.getLastVertex()
            assert isinstance(tail.getWeight(), WayPoint)
            assert isinstance(head.getWeight(), WayPoint)
            tailWayPoint = tail.getWeight()
            headWayPoint = head.getWeight()
            
            strFileName = ""
            if abortedFlight:
                strFileName = "ABORTED-"
            strFileName += str(aircraftICAOcode) + "-" + AdepICAOcode +  "-" + AdesICAOcode
            strFileName += "-" + tailWayPoint.getName()+'-'+headWayPoint.getName()
            strFileName = str(strFileName).replace(' ', '-')
            strFileName += '-{0}.xlsx'.format(datetime.now().strftime("%d-%b-%Y-%Hh%Mm%S"))
            
            groundTrackOutput = GroundTrackOutput(strFileName)
            groundTrackOutput.writeHeaders()
            ''' compute cumulated distance in Meters '''
            cumulatedDistanceMeters = 0.0
            ''' loop '''
            index = 0
            for vertex in self.getVertices():
                ''' build an edge having two consecutive vertices as tail and head '''
                edge = None
                if index > 0:
                    edge = Edge(self.getVertex(index-1).getWeight(), self.getVertex(index).getWeight())
                    
                deltaDistanceMeters = 0.0
                courseAngleDegrees = 0.0
                if not (edge is None):
                    deltaDistanceMeters = edge.getDistanceTailHeadMeters()
                    courseAngleDegrees = edge.getBearingTailHeadDegrees()
                    
                cumulatedDistanceMeters += deltaDistanceMeters
                wayPoint = vertex.getWeight()
                groundTrackOutput.write(wayPoint.getElapsedTimeSeconds(),
                                 wayPoint.getName(),
                                 
                                 wayPoint.getLongitudeDegrees(),
                                 wayPoint.getLatitudeDegrees(),
                                 
                                 wayPoint.getAltitudeMeanSeaLevelMeters(),
                                 deltaDistanceMeters,
                                 cumulatedDistanceMeters,
                                 courseAngleDegrees)
                index += 1
            groundTrackOutput.close()
            
            
    def createCsvAltitudeTimeProfile(self, abortedFlight, aircraftICAOcode, AdepICAOcode, AdesICAOcode):
        assert (type(abortedFlight) == bool )
        
        groundTrack = list()
        maxAltitudeMSLmeters = 0.0
        maxElapsedTimeSeconds = 0.0
        
        counter = 0
        
        if self.getNumberOfVertices() > 1:
            ''' need at least two vertices '''
            tail = self.getVertex(0)
            head = self.getLastVertex()
            assert isinstance(tail.getWeight(), WayPoint)
            assert isinstance(head.getWeight(), WayPoint)
            
            ''' loop '''
            for vertex in self.getVertices():
                ''' build an edge having two consecutive vertices as tail and head '''
                    
                wayPoint = vertex.getWeight()
                if ( wayPoint.getAltitudeMeanSeaLevelMeters() > maxAltitudeMSLmeters):
                    maxAltitudeMSLmeters = wayPoint.getAltitudeMeanSeaLevelMeters()
                
                maxElapsedTimeSeconds = wayPoint.getElapsedTimeSeconds()
                
                outputWayPoint = {
                    "x"  : wayPoint.getElapsedTimeSeconds(),
                    "y"  : round ( wayPoint.getAltitudeMeanSeaLevelMeters() , 1 )
                    }
                #outputWayPoint = [wayPoint.getElapsedTimeSeconds(), wayPoint.getAltitudeMeanSeaLevelMeters()]
                if ( ( counter % 10 ) == 0 ) or abortedFlight:
                    groundTrack.append(outputWayPoint)
                counter = counter + 1
                
        return {
            "groundTrack" : groundTrack,
            "MaxAltitudeMSLmeters" : maxAltitudeMSLmeters,
            "maxElapsedTimeSeconds" : maxElapsedTimeSeconds
            }
            
    def getLengthMeters(self):
        return self.lengthMeters
    
    def computeLengthMeters(self):
        self.lengthMeters = 0.0
        ''' assert that there only one way to visit this graph '''
        for edge in self.getEdges():
            self.lengthMeters +=  edge.getDistanceTailHeadMeters()
        return self.lengthMeters
    