'''
Created on 12 oct. 2025

@author: robert
'''


def keepOnlyColumns( df , listOfColumnNamesToKeep):
        #listOfColumnNamesToKeep = [ 'flight_id', 'takeoff']
        for columnName in list(df):
            if not columnName in listOfColumnNamesToKeep:
                df = df.drop(columnName, axis=1)

        return df