from scipy.fftpack import fft, ifft
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(formatter={"float_kind": lambda x: "%g" % x})

fc=10 # frequency of the carrier
fs=32*fc # sampling frequency with oversampling factor=32
t=np.arange(start = 0,stop = 2,step = 1/fs) # 2 seconds duration
x=np.cos(2*np.pi*fc*t) # time domain signal (real number)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
ax1.plot(t,x) #plot the signal
ax1.set_title('$x[n]= cos(2 \pi 10 t)$')
ax1.set_xlabel('$t=nT_s$')
ax1.set_ylabel('$x[n]$')

N=256 # FFT size
X = fft(x,N) # N-point complex DFT, output contains DC at index 0
# Nyquist frequency at N/2 th index positive frequencies from
# index 2 to N/2-1 and negative frequencies from index N/2 to N-1
print(X[0]) #output approximately equal to zero
print(abs(X[7:10])) #display samples 7 to 9

# calculate frequency bins with FFT
df=fs/N # frequency resolution
sampleIndex = np.arange(start = 0,stop = N) # raw index for FFT plot
f=sampleIndex*df # x-axis index converted to frequencies

ax2.stem(sampleIndex,abs(X),) # sample values on x-axis
ax2.set_title('X[k]');ax2.set_xlabel('k');ax2.set_ylabel('|X(k)|');
ax3.stem(f,abs(X),use_line_collection=True); # x-axis represent frequencies
ax3.set_title('X[f]');ax3.set_xlabel('frequencies (f)');ax3.set_ylabel('|X(f)|');
fig.show()

nyquistIndex=N//2 #// is for integer division
print(X[nyquistIndex-2:nyquistIndex+3, None]) #None argument to print array as column

from scipy.fftpack import fftshift
#re-order the index for emulating fftshift
sampleIndex = np.arange(start = -N//2,stop = N//2) # // for integer division
X1 = X[sampleIndex] #order frequencies without using fftShift
X2 = fftshift(X) # order frequencies by using fftshift
df=fs/N # frequency resolution
f=sampleIndex*df # x-axis index converted to frequencies

#plot ordered spectrum using the two methods
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)#subplots creation
ax1.stem(sampleIndex,abs(X1), use_line_collection=True)# result without fftshift
ax1.stem(sampleIndex,abs(X2),'r',use_line_collection=True) #result with fftshift
ax1.set_xlabel('k');ax1.set_ylabel('|X(k)|')

ax2.stem(f,abs(X1), )
ax2.stem(f,abs(X2),'r' , use_line_collection=True)
ax2.set_xlabel('frequencies (f)'),ax2.set_ylabel('|X(f)|'); fig.show()
