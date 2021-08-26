import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq, irfft
from IPython.display import Audio

sr, data = wavfile.read('engine-8.wav')
N = data.shape[0]
T = N/sr   
engine = wavfile.read('engine-8.wav')

def DFT(audio):
    sr, data = audio #samplerate og data
    N = data.shape[0]
    T = 1/sr    
    time = np.linspace(0, T, N)
    x = data[:,0]
    #regner DFT via FFT
    f_c = (1/N)*np.fft.fft(x)
    freq = np.fft.fftfreq(N, T)
    return f_c, freq, time
xk, freq_eng, time = DFT(engine)

def find_freq(audio):
    sr, data = audio
    freq = rfftfreq(data.shape[0] , 1/sr) #beholder real verdiene ved bruke av rfft
    freq_tot = 0
    for i in range(0, int(sr/2)):
        if xk[i] > 0:
            freq_tot += 1 
    print('total frekvenser som lager lyden:', freq_tot)
    return freq

find_freq(engine)       
def noise_reduce(audio):
    sr, data = audio

    r_freq = find_freq(audio)         #bruker funksjonen find_req til å hente in hoved frekvenser som lager lyden
    orig_freq = rfftfreq(data.shape[0] , 1/sr) #alle frekvensene fra original lyden-filen
    orig_koef = xk                             #regnet fra første funksjon

    new_freq = np.zeros(len(orig_freq))
    print(type(r_freq))
    #new_freq = orig_freq - r_freq
    for i in range(len(orig_freq)):
        new_freq[i] = (abs(orig_freq[i]) - abs(r_freq[i]))
    
    noise_red_koef = np.zeros(len(new_freq))

    for i in range (len(new_freq)):
        if new_freq[i] > 0:
            noise_red_koef[i] = orig_koef[i]

    #henter in dft verdier vi kan bruke siden vi ikke har invers fftfreq()
    red_data = irfft(noise_red_koef)           #reduced data
    return red_data                            

noise_red = noise_reduce(wavfile.read('engine-8.wav'))

plt.plot(time, noise_red[:N])
plt.xlabel('tid [s]')
plt.ylabel('data')
plt.show()
#Audio(data= noise_red[:N], rate=sr)
def combine(audio_1, audio_2):
    sr1, data1 = audio_1
    sr2, data2 = audio_2
    N = data1.shape[1]
    
    x1 = data1[:,0]
    x2 = data2[:,0]

#sjekker om samplingsrate er like
    if sr1 == sr2:
        pass
    else:
        print('sampling rates are different, sampling rate needs to be checked',sr1, sr2)
        
#sjekker lengden av lyden er det samme og sørger for at de har samme lengden
    if len(x1) == len(x2):
        pass
    else:
        print('data sizes are different, changing size...')
        print('length of data 1:',len(x1))
        print('length of data 2:', len(x2))
        
#if-stement som sjekker hvilken data-array er kortest og utvider den med 0-verdier
        if len(x1) > len(x2):
            n = len(x1) - len(x2)
            list2 = x2.tolist()          #bytter den om til liste sånn at vi kan appende
            for i in range(n):
                list2.append(0)
                
            x2 = np.array(list2)         #bytter tilbake til array
            
        else:
            n = len(x2) - len(x1)
            list1 = x1.tolist()          #bytter den om til liste sånn at vi kan appende
            for i in range(n):
                list1.append(0)
                
            x1 = np.array(list1)         #bytter tilbake til array
        print('')    
        print('size has been matched...')
                 
    
    print('length of data 1:', len(x1))
    print('length of data 2:', len(x2))
    c = np.vstack((x1,x2))               #lydene er seperart i hver øre, lydfil_1 i venstre, og lyd_2 i høyre
    d = x1+x2                            #lydene legges sammen
    return c, d
        
    
        
engine  = wavfile.read('engine-8.wav')
talking = wavfile.read('talk(1).wav')
sr, data = engine

left_right, new_aud = combine(engine, talking)
#new_aud, er lyden som har begge kombinert
#left_right, lydene i seprate sider

print('')
print('kombinert lydfilen:')
def ANC(audio, loud): #funksjon som tar in kombinert lyden, og lyden vi skal dempe(høy-lyden)
    
    audio_freq    = rfftfreq(audio.shape[0] , 1/sr)
    unwanted_freq = rfftfreq(loud.shape[0] , 1/sr)
    
    #regner DFT via FFT av kombinert-lyden
    N = audio.shape[0]
    T = 1/sr    
    f_c = (1/N)*np.fft.fft(audio)

    
    audio_koef = f_c               #fourier koeffisientene til kombinert lyden
    filtered_audio = np.copy(f_c)  #array som skal ha koeffisientene etter å kanselere frekvensene fra høye-lyden
    
    if len(audio_freq) == len(unwanted_freq):
        print('comparing frequencies...') 
    else:
        print('error with data length')
        
    n = len(audio_freq)  
    print(n)  
#to check how the arrays compare
    print(audio_freq)
    print(unwanted_freq)
    for i in range(n):
        if audio_freq[i] == unwanted_freq[i]:
            audio_freq[i] = 0                 #fjerner frekvensen som finnes i høye lyden fra kombinert lyd
            filtered_audio[i] = 0             #erstater assosierte fourier-koeffisienten 
                
    print('done calculating filtered fourier coefficients')   
    filtered_data = irfft(filtered_audio)
    print('plotting the filtered data...')
    plt.plot(freq_eng, filtered_data[:N])
    plt.show()
    return filtered_data
    


sr, engine_data  = wavfile.read('engine-8.wav')

#sender kombinert lyd-dataen og motor-dataen som skal representer høy-lyden vi demper:
filtered = ANC(new_aud, engine_data) 
time = np.linspace(0, T, N)




