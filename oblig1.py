import numpy as np 
from numpy import cos, sin
import matplotlib.pyplot as plt 
A = 1.076
w = -2
phi = 1.19

t = np.linspace(0,5,100)
xt = A*cos(w*t + phi) #position
vt = -A*w*sin(w*t + phi) #velocity

plt.plot(xt, vt)
plt.axis('equal')
plt.xlabel('posisjon')
plt.ylabel('hastighet')
plt.show()