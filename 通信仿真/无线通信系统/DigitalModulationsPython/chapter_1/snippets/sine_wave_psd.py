import numpy as np
import matplotlib.pyplot as plt

A=1 #Amplitude of sine wave
fc=100 #Frequency of sine wave
fs=3000 # Sampling frequency - oversampled by the rate of 30
nCyl=3 # Number of cycles of the sinewave
t=np.arange(start = 0,stop = nCyl/fc,step = 1/fs) #Time base
x=-A*np.sin(2*np.pi*fc*t) # Sinusoidal function

fig, (ax1,ax2) = plt.subplots(nrows=1,ncols = 2)
ax1.plot(t,x)
ax1.set_title('Sinusoid of frequency $f_c=100 Hz$')
ax1.set_xlabel('Time(s)');ax1.set_ylabel('Amplitude')
fig.show()

from numpy.linalg import norm
L=len(x)
P=(norm(x)**2)/L; #norm from numpy linear algo package
print('Power of the Signal from Time domain {:0.4f}'.format(P))

from scipy.fftpack import fft,fftshift
NFFT=L
X=fftshift(fft(x,NFFT))
Px=X*np.conj(X)/(L**2) #Power of each freq components
fVals=fs*np.arange(start = -NFFT/2,stop = NFFT/2)/NFFT
ax2.stem(fVals,Px,'r')
ax2.set_title('Power Spectral Density');
ax2.set_xlabel('Frequency (Hz)');ax2.set_ylabel('Power')