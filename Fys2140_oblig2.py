import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, sqrt

x = np.linspace(-200,200,1001)
t = np.linspace(-200,200,1001)
k = np.linspace(-5,5,1001)
a=2
y = 0
for i in range(1001):
    k = k[i]
    A = a/(k**2 + a**2)
    w = sqrt(k**2 + 1)
    yx = A * sin(x*k - w*t)
    y += yx

plt.plot(x,y)
plt.show()

