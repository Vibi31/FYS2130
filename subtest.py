import numpy as np
import matplotlib.pyplot as plt

t0, t1 = 0, 1
dt = 0.001  #tidssteg
m = 1       #masse, 1kg
g = 9.82    #tyngdekraft
f = 0.3     #friksjons koeffisient 
h = 1       #høyde, 1 meter
K = 10000   #flærkonstant
v0x = 3     #start hastighet m/s
R = 0.15    #radius i meter
n = int(t1/dt)


x = np.zeros(n)
y = np.zeros(n)
t = np.linspace(t0, t1, n)
y[0] = h

vx = np.zeros(n+1)
vx[0] = v0x   
vy = np.zeros(n+1)

ax = np.zeros(n)
ay = np.zeros(n)

for i in range(n):
    if y[i] < R:
        ax[i] = f*K*((R-y[i])**(3/2))/m
        ay[i] = (K*(R-y[i])**(3/2))/m - g

    else:
        ax[i], ay[i] = 0, g

    vx[i+1] = vx[i] + ax[i]*dt
    vy[i+1] = vy[i] + ay[i]*dt

    x[i] = x[i-1] - vx[i-1]*dt + 0.5*ax[i]*dt**2 
    y[i] = y[i-1] - vy[i-1]*dt + 0.5*ay[i]*dt**2

plt.plot(x,y)
#plt.plot(t,x, label='x')
plt.show()
    








