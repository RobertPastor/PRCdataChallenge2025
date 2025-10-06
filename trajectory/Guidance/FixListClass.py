'''
Created on 11 août 2023

@author: robert
'''


import logging


from trajectory.Guidance.ConstraintsFile import analyseConstraint

'''
this list is not exactly as an airline route
it might be extended with a top of descent or the start of the last turn
'''
class FixList(object):
    
    className = ""
    ''' ordered list of fixes '''
    fixList = []
    constraintsList = []
    
    strRoute = ""
    departureAirportICAOcode = ""
    arrivalAirportICAOcode = ""
    departureRunwayName = ""
    arrivalRunwayName = ""
    
    
    def __init__(self, strRoute):
        self.className = self.__class__.__name__
        self.fixList = []
        
        self.departureAirportIcaoCode = ""
        self.arrivalAirportICAOcode = ""
        
        self.departureRunwayName = ""
        self.arrivalRunwayName = ""
        
        assert isinstance(strRoute, (str))
        logging.debug (self.className + ': route= ' + strRoute)
        self.strRoute = strRoute
        
        
    def __str__(self):
        return self.className + ' fix list= ' + str(self.fixList)
    
    def getDepartureAirportICAOcode(self):
        return self.departureAirportICAOcode
    
    def getArrivalAirportICAOcode(self):
        return self.arrivalAirportICAOcode
    
    def getDepartureRunwayName(self):
        return self.departureRunwayName
    
    def getArrivalRunwayName(self):
        return self.arrivalRunwayName
    
    def getFix(self):
        for fix in self.fixList:
            yield fix
            
    def createFixList(self):

        logging.info (self.className + ': ================ create Fix List =================' )
        self.fixList = []
        index = 0
        for fix in self.strRoute.split('-'):
            fix = str(fix).strip()
            ''' first item is the Departure Airport '''
            if str(fix).startswith('ADEP'):
                ''' fix is the departure airport '''
                if index == 0:
                    ''' ADEP is the first fix of the route '''
                    if len(str(fix).split('/')) >= 2:
                        self.departureAirportICAOcode = str(fix).split('/')[1]
                        logging.debug (self.className + ': departure airport= {0}'.format( self.departureAirportICAOcode))
    
                    self.departureRunwayName = ''
                    if len(str(fix).split('/')) >= 3:
                        self.departureRunwayName = str(fix).split('/')[2]
                        
                else:
                    raise ValueError (self.className + ': ADEP must be the first fix in the route!!!')

                
            elif  str(fix).startswith('ADES'):
                ''' check if Destination Airport is last item of the list '''
                if index == (len(self.strRoute.split('-'))-1):
                    ''' ADES is the last fix of the route '''
                    if len(str(fix).split('/')) >= 2:
                        self.arrivalAirportICAOcode = str(fix).split('/')[1]
                        logging.debug (self.className + ': arrival airport= {0}'.format( self.arrivalAirportICAOcode))

                    self.arrivalRunwayName = ''
                    if len(str(fix).split('/')) >= 3:
                        self.arrivalRunwayName = str(fix).split('/')[2]
                    
                else:
                    raise ValueError (self.classeName + ': ADES must be the last fix of the route!!!' )

            else:
                ''' do not take the 1st one (ADEP) and the last one (ADES) '''
                constraintFound, levelConstraint, speedConstraint = analyseConstraint(index, fix)
                #logging.info self.className + ': constraint found= {0}'.format(constraintFound)
                if constraintFound == True:
                    constraint = {}
                    constraint['fixIndex'] = index
                    constraint['level'] = levelConstraint
                    constraint['speed'] = speedConstraint
                    self.constraintsList.append(constraint)
                else:
                    self.fixList.append(fix)

            index += 1             


    def deleteFix(self, thisFix):
        if thisFix in self.fixList:
            self.fixList.remove(thisFix)
        
    def indexIsTheLast(self, index):
        return index == len(self.fixList)-1