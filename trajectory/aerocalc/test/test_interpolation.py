'''
Created on 9 juin 2025

@author: robert
'''


def interpolate(x , xArray, yArray):
    #x1 must be greater to x0
    x0 , x1 = xArray
    y0 , y1 = yArray
    
    y = ( ( (y1 - y0) / (x1 - x0) ) * ( x - x0 ) ) + y0
    
    return y

if __name__ == '__main__':
    pass
    x = 28000.0
    xArray = [ 15000 , 33000]
    yArray = [ 0.55 , 0.82 ]
    y = interpolate( x , xArray, yArray)
    print ( y )
    
    x = 31000
    y = interpolate( x , xArray, yArray)
    print ( y )
    
    
    x = 33000
    y = interpolate( x , xArray, yArray)
    print ( y )
    
    x = 39000
    y = interpolate( x , xArray, yArray)
    print ( y )