



from scipy.fftpack import fft, ifft, fftshift, ifftshift
import numpy as np
import matplotlib.pyplot as plt

A = 0.5 # amplitude of the cosine wave
fc=10 # frequency of the cosine wave in Hz
phase=30 # desired phase shift of the cosine in degrees
fs=32*fc # sampling frequency with oversampling factor 32
t=np.arange(start = 0,stop = 2,step = 1/fs) # 2 seconds duration

phi = phase*np.pi/180; # convert phase shift in degrees in radians
x=A*np.cos(2*np.pi*fc*t+phi) # time domain signal with phase shift

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)
ax1.plot(t,x) # plot time domain representation
ax1.set_title(r'$x(t) = 0.5 cos (2 \pi 10 t + \pi/6)$')
ax1.set_xlabel('time (t seconds)');ax1.set_ylabel('x(t)')

N=256 # FFT size
X = 1/N*fftshift(fft(x,N)) # N-point complex DFT

df=fs/N # frequency resolution
sampleIndex = np.arange(start = -N//2,stop = N//2) # // for integer division
f=sampleIndex*df # x-axis index converted to ordered frequencies
ax2.stem(f, abs(X), ) # magnitudes vs frequencies
ax2.set_xlim(-30, 30)
ax2.set_title('Amplitude spectrum')
ax2.set_xlabel('f (Hz)');ax2.set_ylabel(r'$ \left| X(k) \right|$')

phase=np.arctan2(np.imag(X),np.real(X))*180/np.pi # phase information
ax3.plot(f,phase) # phase vs frequencies
ax3.set_title('Phase spectrum')
ax3.set_ylabel(r"$\angle$ X[k]");ax3.set_xlabel('f(Hz)')

X2=X #store the FFT results in another array
# detect noise (very small numbers (eps)) and ignore them
threshold = max(abs(X))/10000; # tolerance threshold
X2[abs(X)<threshold]=0 # maskout values below the threshold
phase=np.arctan2(np.imag(X2),np.real(X2))*180/np.pi # phase information
ax4.stem(f,phase, ) # phase vs frequencies
ax4.set_xlim(-30, 30); ax4.set_title('Phase spectrum')
ax4.set_ylabel(r"$\angle$ X[k]");ax4.set_xlabel('f(Hz)')
fig.show()

x_recon = N*ifft(ifftshift(X),N) # reconstructed signal
t = np.arange(start = 0,stop = len(x_recon))/fs # recompute time index
fig2, ax5 = plt.subplots()
ax5.plot(t,np.real(x_recon)) # reconstructed signal
ax5.set_title('reconstructed signal')
ax1.set_xlabel('time (t seconds)');ax1.set_ylabel('x(t)');
fig2.show()
