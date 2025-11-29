'''
Created on 15 nov. 2024

@author: rober

'''
import sys
sys.path.append("C:/Users/rober/git/openap/") #replace PATH with the path to Foo

from openap import Drag

drag = Drag(ac='A320' , wave_drag = True)

print('-'*70)

D = drag.clean(mass=60000, tas=200, alt=20000)
print("drag.clean(mass=60000, tas=200, alt=20000)")
print(D)
print('-'*70)

D = drag.clean(mass=60000, tas=250, alt=20000)
print("drag.clean(mass=60000, tas=250, alt=20000)")
print(D)
print('-'*70)

D = drag.clean(mass=60000, tas=2.50, alt=200)
print("drag.clean(mass=60000, tas=2.50, alt=200)")
print(D)
print('-'*70)

D = drag.nonclean(mass=60000, tas=150, alt=1000, flap_angle=20,  landing_gear=False)
print("drag.nonclean(mass=60000, tas=150, alt=1000, flap_angle=20, path_angle=10, landing_gear=False)")
print(D)
print('-'*70)

D = drag.nonclean(mass=60000, tas=1.50, alt=100, flap_angle=20,  landing_gear=True)
print("drag.nonclean(mass=60000, tas=1.50, alt=100, flap_angle=20, landing_gear=True)")
print(D)
print('-'*70)

D = drag.nonclean(mass=60000, tas=1.50, alt=100, flap_angle=20,  landing_gear=False)
print("drag.nonclean(mass=60000, tas=1.50, alt=100, flap_angle=20, landing_gear=False)")
print(D)
print('-'*70)

D = drag.nonclean(mass=60000, tas=150, alt=200, flap_angle=20,  landing_gear=True)
print("drag.nonclean(mass=60000, tas=150, alt=1000, flap_angle=20, path_angle=10, landing_gear=True)")
print(D)

print('-'*70)
print('-'*70)

for tas in range( 0 , 150 ):
    print("drag.nonclean(mass=60000, tas={0}, alt=100, flap_angle=20, landing_gear=True)".format( tas))
    D = drag.nonclean(mass=60000, tas=tas, alt=200, flap_angle=20,  landing_gear=True)

    print(D)
    print('-'*70)
    
D = drag.clean(mass=[60000], tas=[200], alt=[20000])
print("drag.clean(mass=[60000], tas=[200], alt=[20000])")
print(D)
print('-'*70)

D = drag.nonclean(mass=[60000], tas=[150], alt=[200], flap_angle=[20])
print("drag.nonclean(mass=[60000], tas=[150], alt=[200], flap_angle=[20]")
print(D)
print('-'*70)
