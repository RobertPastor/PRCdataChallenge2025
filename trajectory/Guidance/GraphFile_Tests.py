

import unittest
import logging
import time 

from trajectory.Guidance.GraphFile import Graph, Vertex
from trajectory.Environment.Airports.AirportDatabaseFile import AirportsDatabase

class Test_Graph(unittest.TestCase):

    def test_main_one(self):    
        g1 = Graph()
        logging.info ( 'empty graph= ' + str( g1 ) )
        v1 = Vertex('Robert')
        g1.addVertex(v1)
        logging.info ( g1 )
        logging.info ( 'number of vertices: {0}'.format(g1.getNumberOfVertices()) )
        logging.info ( g1.getLastVertex().getWeight() )
        logging.info ( g1.getVertex(0).getWeight() )
        
    def test_main_two(self):    
        g1 = Graph()
        v1 = Vertex('Robert')
        v2 = Vertex('Francois')
        g1.addVertex(v1)
        g1.addVertex(v2)
        logging.info ( 'number of vertices: {0}'.format(g1.getNumberOfVertices()) )
        logging.info ( 'number of edges: {0}'.format(g1.getNumberOfEdges()) )
        logging.info ( g1.getLastVertex().getWeight() )
        
    def test_main_three(self):    

        g2 = Graph()
        v3 = Vertex('Marie')
        g2.addVertex(v3)
        
        g1 = Graph()
        v1 = Vertex('Robert')
        v2 = Vertex('Francois')
        g1.addVertex(v1)
        g1.addVertex(v2)
        
        g1.addGraph(g2)
        logging.info ( g1 )
        for vertex in g1.getVertices():
            logging.info ( str(vertex) )
        logging.info ( "=================" )
        for edge in g1.getEdges():
            logging.info ( str(edge.getTail()) + " --- " + str(edge.getHead() ) )

    def test_main_four(self):    

        logging.info ( " ========== AirportsDatabase testing ======= time start= " )
        airportsDb = AirportsDatabase()
        assert (airportsDb.read())
        airportsDb.dumpCountry(Country="France")
        logging.info ( "number of airports= " + str( airportsDb.getNumberOfAirports() ) )
        for ap in ['Orly', 'paris', 'toulouse', 'marseille' , 'roissy', 'blagnac' , 'provence' , 'de gaulle']:
            logging.info ( "ICAO Code of= " +  str(ap) + " ICAO code= " + str(airportsDb.getICAOCode(ap) ) )
        
        t1 = time.perf_counter()
        
        logging.info ( " ========== AirportsDatabase testing ======= time start= " + str( t1 ) )
        CharlesDeGaulleRoissy = airportsDb.getAirportFromICAOCode('LFPG')
        logging.info ( CharlesDeGaulleRoissy )
        MarseilleMarignane = airportsDb.getAirportFromICAOCode('LFML')
        logging.info ( MarseilleMarignane )
        
        g0 = Graph()
        for icao in [ 'LFPO', 'LFMY', 'LFAT', 'LFGJ']:
            airport = airportsDb.getAirportFromICAOCode(icao)
            g0.addVertex(airport)
        logging.info ( '================ g0 ================='  )
        for node in g0.getVertices():
            logging.info ( node )
        
        g1 = Graph()
        for icao in [ 'LFKC', 'LFBO' , 'LFKB']:
            airport = airportsDb.getAirportFromICAOCode(icao)
            g1.addVertex(airport)     
            
        logging.info ( '================ g1 ================='  )
        for node in g1.getVertices():
            logging.info ( node )
            
        logging.info ( ' ============== g0.add_graph(g1) ===============' )
        g0.addGraph(g1)
        for node in g0.getVertices():
            logging.info ( node )
            
        logging.info ( ' ============== g0.create XLS file ===============' )
    
        #g0.createXlsxOutputFile()
        #g0.createKmlOutputFile()
    
    def test_main_five(self):    

        airportsDb = AirportsDatabase()
        assert (airportsDb.read())
        
        logging.info ( ' ============== g3 performance ===============' )
        t0 = time.perf_counter()
        g3 = Graph()
        index = 0
        for airport in airportsDb.getAirports():
            logging.info ( airport )
            g3.addVertex(airport)
            index += 1
        t1 = time.perf_counter()
        logging.info ( 'number of airports= {0} - duration= {1} seconds'.format(index, t1-t0) )
     
        airport = airportsDb.getAirportFromICAOCode('LFPG')
        t2= time.perf_counter()
        g4 = Graph()
        for i in range (0,10000):
            g4.addVertex(airport)
        t3 = time.perf_counter()
        logging.info ( 'number of addVertex = {0} - duration= {1:.8f} seconds'.format(i , t3-t2) )
    
if __name__ == '__main__':
    logging.basicConfig()
    unittest.main()