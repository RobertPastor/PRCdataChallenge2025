'''
Created on 5 oct. 2025

@author: rober
'''



def convertDegreeMinuteSecondToDecimal(DegreeMinuteSecond='43-40-51.00-N'):
    '''
        convert from Decimal Degrees = Degrees + minutes/60 + seconds/3600
        to float
        mays start or end with NE SW
    '''
    DecimalValue = 0.0
    coeff = 0.0
    assert isinstance(DegreeMinuteSecond, str) 
        
    if ( str(DegreeMinuteSecond).endswith("N") or 
         str(DegreeMinuteSecond).endswith("E") or 
         str(DegreeMinuteSecond).startswith("N") or 
         str(DegreeMinuteSecond).startswith("E") ):
        ''' transform into decimal value '''
        coeff = 1.0
        
    elif ( str(DegreeMinuteSecond).endswith("S") or 
           str(DegreeMinuteSecond).endswith("W") or
           str(DegreeMinuteSecond).startswith("S") or 
           str(DegreeMinuteSecond).startswith("W") ):
        ''' transform into decimal value '''
        coeff = -1.0
    
    else :
        raise ValueError ('Degrees Minutes Seconds string should be starting or ending by N-E-S-W')
    
    if  ( str(DegreeMinuteSecond).endswith("N") or 
          str(DegreeMinuteSecond).endswith("E") or 
          str(DegreeMinuteSecond).endswith("S") or 
          str(DegreeMinuteSecond).endswith("W") ):
        ''' suppress last char and split '''
        strSplitList = str(DegreeMinuteSecond[:-1]).split('-')
    else:
        ''' suppress first char and split '''
        strSplitList = str(DegreeMinuteSecond[1:]).split('-')

    #print strSplitList[0]
    if str(strSplitList[0]).isdigit() and str(strSplitList[1]).isdigit():
        DecimalDegreeValue = int(strSplitList[0])
        DecimalMinutesValue = int(strSplitList[1])
        #print strSplitList[1]
        strSplitList2 = str(strSplitList[2]).split(".")
        #print strSplitList2[0]
        if (len(strSplitList2)==2 and str(strSplitList2[0]).isdigit() and str(strSplitList2[1]).isdigit()):
                
            DecimalSecondsValue = int(strSplitList2[0])
            TenthOfSecondsValue = int(strSplitList2[1])
            
            DecimalValue = DecimalDegreeValue + float(DecimalMinutesValue)/float(60.0)
            DecimalValue += float(DecimalSecondsValue)/float(3600.0)
            if TenthOfSecondsValue < 10.0:
                DecimalValue += (float(TenthOfSecondsValue)/float(3600.0)) / 10.0
            else:
                ''' two digits of milliseconds '''
                DecimalValue += (float(TenthOfSecondsValue)/float(3600.0)) / 100.0
                    
            DecimalValue = coeff * DecimalValue
        else:
            raise ValueError ('unexpected Degrees Minutes Seconds format')
    else:
        raise ValueError ('unexpected Degrees Minutes Seconds format')

    #print "DegreeMinuteSecond= ", DegreeMinuteSecond, " DecimalValue= ", DecimalValue
    return DecimalValue