'''
Created on 12 oct. 2025

@author: robert
'''
import pandas as pd
from datetime import datetime

from sklearn.preprocessing import OneHotEncoder
oneHotEncoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

def getCurrentDateTimeAsStr( ):
    # Create a datetime object
    current_date = datetime.now()

    # Convert to string with a specific format
    date_string = current_date.strftime("%Y-%m-%d-%H-%M-%S")
    print("Formatted Date String:", date_string)
    return date_string

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
    
    ''' Get one hot encoding of columns B '''
    df[[columnName]] = df[[columnName]].fillna(value = 'unknown')
    one_hot = pd.get_dummies(df[columnName] , dtype=float )
    # Drop column B as it is now encoded
    df = df.drop(columnName , axis = 1)
    # Join the encoded df
    return df.join(one_hot)
    
def oneHotEncoderSklearn(df , columnName):
    
    print(columnName)
    df[columnName].fillna('missing', inplace=True)
    df[columnName] = df[columnName].astype(str)
    # Apply label encoding
    encoded_array = oneHotEncoder.fit_transform(df[[columnName]])
    # Convert the encoded array to a DataFrame
    encoded_df = pd.DataFrame( encoded_array, columns = oneHotEncoder.get_feature_names_out([columnName]))
    ''' suppress not used anymore column '''
    df = df.drop(columnName , axis = 1)
    # Join the encoded DataFrame with the original DataFrame
    df = pd.concat([df, encoded_df], axis=1)
    
    return df
    