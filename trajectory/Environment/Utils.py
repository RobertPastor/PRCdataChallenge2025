'''
Created on 25 juin 2023

@author: robert
'''
import logging 

logger = logging.getLogger(__name__)

def logElapsedRealTime(className, elapsedTimeSeconds):
    
        if elapsedTimeSeconds >= 60.0 and elapsedTimeSeconds < 3600.0:
            minutes, seconds = divmod(elapsedTimeSeconds, 60)
            logger.debug  ( className + ': real time = {0:.2f} seconds - {1:.2f} minutes {2:.2f} seconds'.format(elapsedTimeSeconds, minutes, seconds) )
        else:
            minutes, seconds = divmod(elapsedTimeSeconds, 60)
            hours, minutes = divmod(minutes, 60)
            logger.debug  ( className + ': real time = {0:.2f} seconds - {1:.2f} hours {2:.2f} minutes {3:.2f} seconds'.format(elapsedTimeSeconds, hours, minutes, seconds) )
