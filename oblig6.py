
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin

#oppgave 1
A = 1
T = 1 #total samplings tid
f_samp = 1000 # samplingsfrekvens  1KHz = 1000Hz
N = T*f_samp; # total samplingspunkter WHAT sampling number should we use? 16*8?
f = 100 #100Hz
dt = 1/f_samp
t = np.linspace(0, T, N) #tids array

for i in range (4):
    ti = ['100Hz', '400Hz', '700Hz', '1300Hz']
    f = [100, 400, 700, 1300]

    g = A*sin(2*pi*f[i]*t)

    # regner DFT via FFT
    g_k = (1/N)*np.fft.fft(g)
    freq = np.fft.fftfreq(N, dt)

    plt.plot(t, g, color='blue', linestyle='-', marker='.')
    plt.xlabel('t [s]')
    plt.ylabel('g_n')
    plt.show()

    plt.bar(freq, np.imag(g_k), color='black', width=0.5)
    plt.bar(freq, np.abs(g_k), color='black', width=0.5) # ofte ønsker vi å se på absolutt-verdien
    plt.xlabel('f [Hz]')
    plt.ylabel('g_k')
    plt.title(ti[i])
    plt.show()

"""
#oppgave 2
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, pi

N = 1000
X = np.linspace(-2*pi, 2*pi, N)
g = sin(X)**2
mean = sum(g/N)
g_k = (1/N)*np.fft.fft(g)

print('gjennomsnitt', mean)
print('første komponent', g_k[0])

plt.plot(X, g)
plt.axhline(y = mean, color = 'r', linestyle = '-')
plt.show()
"""
"""
#oppgave 3
import matplotlib.pyplot as plt 
import numpy as np 
from numpy import sin, exp, pi, imag
A1, A2 = 1, 1.7     #amplitude
f1, f2 = 100, 160   #freq in Hz
t1, t2 = 0.2, 0.6   #seconds
std1, std2 = 0.05, 0.1 #standard deviation
T = 1
N = 1000
t = np.linspace(0,1,N)
dt = T/N
f = A1*sin(2*pi*f1*t)*exp(-((t-t1)/std1)**2) + A2*sin(2*pi*f2*t)*exp(-((t-t2)/std2)**2)
#samplede tidsserien
plt.plot(t, f)
plt.title('sample')
plt.show()

#DFT via fft
g_k = (1/N)*np.fft.fft(f)
freq = np.fft.fftfreq(N, dt)

plt.plot(freq, imag(g_k))
plt.xlabel('Hz')
plt.ylabel('g_k')
plt.title('diskrete fourier transform')
plt.show()
"""


"""
#oppgave3b og mest av 3c
import matplotlib.pyplot as plt 
import numpy as np 
from numpy import sin, exp, pi, imag

A1, A2 = 1, 1.7     #amplitude
f1, f2 = 100, 160   #freq in Hz
t1, t2 = 0.2, 0.6   #seconds
std1, std2 = 0.05, 0.1 #standard deviation
T = 1
N = 1000
t = np.linspace(0,1,N)
dt = T/N
f = A1*sin(2*pi*f1*t)*exp(-((t-t1)/std1)**2) + A2*sin(2*pi*f2*t)*exp(-((t-t2)/std2)**2)

#DFT via fft
g_k = (1/N)*np.fft.fft(f)
freq = np.fft.fftfreq(N, dt)

def wavelet(K, w, tk, tn): #wavelet formula (14.8)
    cplx = np.complex(0, 1)
    fs = A1*sin(2*pi*f1*t)*exp(-((t-t1)/std1)**2) + A2*sin(2*pi*f2*t)*exp(-((t-t2)/std2)**2)
    C = 0.798*w/(fs*K1)  #formula (14.7)
    return C*(exp(-cplx*w*(tn-tk)) - exp(-K**2)) * exp(-w*2*(tn-tk)**2 / (2*K)**2)

def wavelet_transform(K, w, tk, tn, x, N): #wavelet transform (14.9)
    gamma = np.zeros(N, dtype=np.complex_)
    for i in range(N):
        print(w)
        gamma[i] = x* np.conjugate(wavelet(K, w, tk, tn))
    return gamma

def wavelet_diagram(K, w, tk, tn, x, N): #der w er en liste
    wav_list = np.zeros(N)
    for i in range(N):
        wav_list[i] = wavelet_transform(K, w[i], tk, tn, x[i], N)

N = 100 #analyse 
w = np.logspace(80,200, num = N)
K1 = 6
k2 = 60
fs = A1*sin(2*pi*f1*t)*exp(-((t-t1)/std1)**2) + A2*sin(2*pi*f2*t)*exp(-((t-t2)/std2)**2)
x = np.linspace(-2*pi, 2*pi, N)
wavelet_diagram(K1, w, t1, t2, x, N)
"""

"""
#oppgave 4c
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# time series
SAMPLE_RATE, DATA = wavfile.read('cuckoo.wav')
# N = DATA.shape[0]
F_S = SAMPLE_RATE
T = 1 # N / F_SAMP
N_SAMP = 20
T_1 = 0.4
T_2 = 1.1
X_N = DATA[np.int_(T_1*F_S):np.int_(T_2*F_S):N_SAMP, 0]
N = len(X_N)

DT = 1 / F_S
TIME = DT*np.linspace(T_1, T_2, N)

# calc DFT via FFT
X_K = (1/N)*np.fft.fft(X_N)
FREQ = np.fft.fftfreq(N, DT)

# short-time fourier transform
# F_STFT, T_STFT, FXX = signal.stft(X_N, F_SAMP, nperseg=2000)

FIG, AX = plt.subplots(2, 1)
AX[0].grid(1)
AX[1].grid(1)
AX[0].plot(TIME, X_N, color='blue', linestyle='solid', linewidth=0.2)
AX[0].set_xlabel('t [s]')
AX[0].set_ylabel('x$_n$')
#
AX[1].plot(FREQ, np.abs(X_K[:N]), color='black', linewidth=0.5)
AX[1].set_xlabel('f [Hz]')
AX[1].set_ylabel('X$_k$')
AX[1].set_xlim([0, 1000])
plt.show()
plt.close()


K1 = 60
AN_FREQ = 100
OM_A = np.logspace(np.log10(2*np.pi*80), np.log10(2*np.pi*200), AN_FREQ)
t_n = np.linspace(0.4, 1.1, N)

#wavelet_plot(K1)
# psi = wavelet(800, t_n, 0.3, 6)

# plt.plot(t_n, psi)
plt.show()
"""

"""
#oppgave 4a
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

f_s , data = wavfile.read('cuckoo.wav')
print('sample rate:',f_s)
N = data.shape[0]
T = N/f_s
x = data[:,0]
f_c = (1/N) *np.fft.fft(x)
freq = np.fft.fftfreq(N, T/f_s)

plt.plot(freq, f_c)
plt.title('DFT')
plt.xlabel('frekvens')
plt.ylabel('fourier koeffisient')
plt.show()
"""