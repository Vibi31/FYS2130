import numpy as np
from numpy import pi, sqrt
import matplotlib.pyplot as plt


#oppgave 2B
N = 3000  #cell total
length_tot = 0.03 #30mm in m
Lcell = length_tot/3000 #length per cell

height = np.linspace(0.0003, 0.0001, N)
width = np.linspace(0.0001, 0.0003, N)
density = np.linspace(1500, 2500, N)
volume = np.zeros(N)

for i in range (N):
    volume[i] = height[i]*width[i]*Lcell  

def masse(l):
    mass = density[l] * volume[l]
    return mass

print(masse(2))

#Oppgave 2C

cell = np.linspace(0,3000,3000) #cell number
k = np.linspace(10e-6,10e-1,3000)#fjærstivhet 

frekvens = np.zeros(3000)
for i in range (3000):
    frekvens[i] =   sqrt(k[i]/masse(i))/(2*pi)
"""
plt.plot(cell, frekvens)
plt.xlabel('cell nummer')
plt.ylabel('frekvens')
#plt.show()
"""
#oppgave 2D
F = 10
f1 =  np.zeros(3000)
f2 = np.zeros(3000)

b=10e-10
for i in range(3000):
    f1[i] = (F/masse(i)) / sqrt(frekvens[i]**2 - (261.63*2*pi)**2 + (b*261.63*2*pi/masse(i))**2)
    f2[i] = (F/masse(i)) / sqrt(frekvens[i]**2 - (277.18*2*pi)**2 + (b*277.18*2*pi/masse(i))**2)
x = np.linspace(0,0.003, 3000)
plt.plot(x, f1)
plt.plot(x, f2)
plt.legend('261', '277')
plt.ylabel('amplitude')
plt.xlabel('posisjon (m)')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, pi

N=3000
height = np.linspace(0.0003, 0.0001, N)
width = np.linspace(0.0001, 0.0003, N)
density = np.linspace(1500, 2500, N)
L=1e-5
m = height*width*density*L

k=np.linspace(10e-6,10e-1,3000)
f=(1/2*pi)*sqrt(k/m)

C4=261.63
C4s=277.18
C4w=2*pi*C4
C4sw=2*pi*C4s
F=1
b=10**(-7)

aC4 =(F/m)/(sqrt((f**2-C4w**2)**2+(b*C4w/m)**2))
aC4s=(F/m)/(sqrt((f**2-C4w**2)**2+(b*C4sw/m)**2))

x = np.linspace(0,0.03,3000)
plt.plot(x,aC4)
plt.plot(x,aC4s)

plt.xlabel("posisjon (m)")
plt.ylabel("Amplitude")
plt.show()

