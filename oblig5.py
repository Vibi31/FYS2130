
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin

#  signal parameters
omega = 2*pi

# sampling parameters
per = 13 #antal perioder
T = 2*pi/omega * per; # total samplingtid
N = 512; # total samplingspunkter
#N = 16*8; # bruke denne høyere samplingsfrekvens 
dt = T/N
f_samp = 1/dt # samplingsfrekvens 
t = np.linspace(0, T, N)
x_n = sin(omega*t) # sampled signal
#x_n = 1.0 * np.sin(omega_sig*t) + 1.0*np.sin(14*omega_sig*t) # legg til en høyere frekvens
#x_n = 1.0 * np.sin(omega_sig*t) + 1.0*np.sin(50*omega_sig*t) # sampled signal with one fsig > fs/2

# calc DFT via FFT
X_k = (1/N)*np.fft.fft(x_n)
#freq = 1/T*np.linspace(0, N-1, N) # bruk denne for å studere folding
#freq = 1/T*np.concatenate( [np.linspace(0, N//2, N//2+1), np.linspace(-N//2+1, -1, N//2-1)]) # what fftfreq does
freq = np.fft.fftfreq(N, dt)
#print('X', X_k)
#print('freq', freq)

#
fig, ax = plt.subplots(2,1, figsize = (13, 6))
ax[0].grid(1)
ax[0].plot(t, x_n, color='blue', linestyle='solid', marker='o')
ax[0].set_xlabel('t [s]')
ax[0].set_ylabel('x_n')

ax[1].grid(1)
ax[1].bar(freq, np.imag(X_k), color='black', width=0.2)
ax[1].bar(freq, np.abs(X_k), color='black', width=0.2) # ofte ønsker vi å se på absolutt-verdien
ax[1].set_xlabel('f [Hz]')
ax[1].set_ylabel('X_k')
#ax[1].set_xlim([45, 55])
plt.show()
