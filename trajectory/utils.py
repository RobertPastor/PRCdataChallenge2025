'''
Created on 12 oct. 2025

@author: robert
'''
import pandas as pd


def keepOnlyColumns( df , listOfColumnNamesToKeep):
    #listOfColumnNamesToKeep = [ 'flight_id', 'takeoff']
    for columnName in list(df):
        if not columnName in listOfColumnNamesToKeep:
            df = df.drop(columnName, axis=1)
    return df
    
def dropUnusedColumns( df , listOfColumnsToDrop ):
    for columnName in list(df):
        if columnName in listOfColumnsToDrop:
            df = df.drop(columnName, axis=1)
    return df 

def oneHotEncodeSource( df , columnName):
        ''' source columns contains only two values
        INFO:root:source
        adsb     25453
        acars        8 '''
        
        # Get one hot encoding of columns B
        one_hot = pd.get_dummies(df[columnName])
        # Drop column B as it is now encoded
        df = df.drop(columnName , axis = 1)
        # Join the encoded df
        return df.join(one_hot)