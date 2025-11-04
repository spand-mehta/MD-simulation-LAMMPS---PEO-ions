#Script to generate timesteps for outputting system paramteres

#Created:P.Vijayaraghavan 4/10/2014

import sys
Timestep=open('Timesteps.txt' , 'w')

e=20
for n in range(0,e):
    a=2
    b=3
    while b<=10000000:
        c=6291456*n+a
        d=6291456*n+b
        print ("%i\n" %(c)) 
        Timestep.write("%i\n" %(c));
        Timestep.write("%i\n" %(d));
        a=a*2
        b=b*2

Timestep.close()

    
    


 