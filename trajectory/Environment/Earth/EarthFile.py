# -*- coding: UTF-8 -*-

'''
Created on Jul 1, 2014

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
import math

from trajectory.Environment.Constants import EarthRadiusMeters

class Earth():
    
    radiusMeters = 6378135.0 # earth’s radius in meters
    omega = 2 * math.pi/ (23 * 3600 + 56 * 60 + 4.0905) # earth’s rot. speed (rad/s)
    mu = 3.986004e14 # mu = GMe %earth’s grav. const (m^3/s^2)
    
    def __init__(self, 
                 radius = 6378135.0, # in meters
                 omega = (2 * math.pi/ (23 * 3600 + 56 * 60 + 4.0905)) ,# earth’s rot. speed (rad/s)
                 mu = 3.986004e14 ): # mu = GMe %earth’s grav. const (m^3/s^2)

        self.className = self.__class__.__name__

        self.radiusMeters = radius
        self.omega = omega
        self.mu = mu
        
    def getRadiusMeters(self):
        return self.radiusMeters
    
    def gravityWelmec(self, heightMSLmeters, latitudeDegrees):
        latitudeRadians = math.radians(latitudeDegrees)
        gravity = 1 + ( 0.0053024 * math.sin(latitudeRadians) * math.sin(latitudeRadians) )
        gravity = gravity - ( 0.0000058 * math.sin( 2 * latitudeRadians) * math.sin(2 * latitudeRadians) )
        gravity = gravity * 9.780318
        gravity = gravity - ( 0.000003085 * heightMSLmeters )
        return gravity
        
    def gravity(self, radius, latitudeRadians):
        # returns gc gnorth
        # (c) 2006 Ashish Tewari
        
        phi = math.pi / 2.0 - latitudeRadians
        Re = self.radiusMeters
        
        J2 = 1.08263e-3
        J3 = 2.532153e-7
        J4 = 1.6109876e-7
        gc = self.mu * (1-1.5 * J2 * ( 3 * (math.cos(phi) ** 2) -1)*((Re/radius)** 2) - 2 * J3* math.cos(phi)*(5*math.cos(phi)**2-3)*(Re/radius) ** 3-(5/8) * J4 * (35 * (math.cos(phi) ** 4) - 30 * (math.cos(phi)**2) +3 )*((Re/radius)**4)) / (radius**2)
        gnorth = -3 * self.mu * math.sin(phi)* math.cos(phi) * (Re/radius) * (Re/radius) * (J2 + 0.5 * J3 * (5 * math.cos(phi) ** 2 - 1) * (Re/radius) / math.cos(phi) +(5/6) * J4 * (7 * math.cos(phi)**2-1) * (Re/radius) ** 2)/ (radius**2)
        return gc, gnorth

    def dump(self):
        print ("earth radius: ", self.radiusMeters, " meters")
        print ("earth's rotation speed: ", self.omega, " radians/sec")
        print ("earth's gravity constant: ", self.mu, " m^3/s^2")

    def __str__(self):
        strMsg = self.className + " earth radius= {0} meters".format( self.radiusMeters )
        strMsg += " - earth's rotation speed=  {0} radians/sec".format( self.omega,)
        strMsg += " - earth's gravity constant= {0} m^3/s^2".format( self.mu )
        return strMsg
