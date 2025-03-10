"""
CPFSK modulation (waveform simulation)

@author: Mathuranathan Viswanathan
Created on Jul 23, 2019
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

L = 50 # oversampling factor
Tb = 0.5 # bit period in seconds
fs = L/Tb # sampling frequency in Hertz
fc = 2/Tb # carrier frequency
N = 8 # number of bits to transmit
h = 1 # modulation index

b = 2*np.random.randint(2, size=N)-1 # random information sequence in +1/-1 format
b = np.tile(b, (L,1)).flatten('F')
b_integrated = lfilter([1.0],[1.0,-1.0],b)/fs #Integrate b using filter

theta= np.pi*h/Tb*b_integrated
t=np.arange(0, Tb*N, 1/fs) # time base

s = np.cos(2*np.pi*fc*t + theta) # CPFSK signal

fig, (ax1,ax2,ax3) = plt.subplots(3, 1)
ax1.plot(t,b);ax1.set_xlabel('t');ax1.set_ylabel('b(t)')
ax2.plot(t,theta);ax2.set_xlabel('t');ax2.set_ylabel('$\theta(t)$')
ax3.plot(t,s);ax3.set_xlabel('t');ax3.set_ylabel('s(t)')
fig.show()