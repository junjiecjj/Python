"""
Script: DigiCommPy.chapter_5.zf_equalizer_test.py
Simulation of Zero Forcing equalizer

@author: Mathuranathan Viswanathan
Created on Aug 23, 2019
"""
import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from numpy import pi,log,convolve
nSamp=5 #%Number of samples per symbol determines baud rate Tsym
Fs=100 # Sampling Frequency of the system
Ts=1/Fs # Sampling time
Tsym=nSamp*Ts # symbol time period

#Define transfer function of the channel
k=6 # define limits for computing channel response
N0 = 0.001 # Standard deviation of AWGN channel noise
t = np.arange(start=-k*Tsym,stop=k*Tsym,step=Ts) # time base defined till +/-kTsym
h_t = 1/(1+(t/Tsym)**2) # channel model, replace with your own model
h_t = h_t + N0*np.random.randn(len(h_t)) # add Noise to the channel response
h_k = h_t[0::nSamp] # downsampling to represent symbol rate sampler
t_inst=t[0::nSamp] # symbol sampling instants

fig, ax = plt.subplots(nrows=1,ncols = 1)
ax.plot(t,h_t,label='continuous-time model');#response at sampling instants
# channel response at symbol sampling instants
ax.stem(t_inst,h_k,'r',label='discrete-time model',use_line_collection=True)
ax.legend();ax.set_title('Channel impulse response');
ax.set_xlabel('Time (s)');ax.set_ylabel('Amplitude');fig.show()

# Equalizer Design Parameters
N = 14 # Desired number of taps for equalizer filter
delay = 11

# design zero-forcing equalizer for given channel and get tap weights and
# filter the input through the equalizer find equalizer co-effs for given CIR
from equalizers import zeroForcing
zf = zeroForcing(N) #initialize ZF equalizer (object) of length N
mse = zf.design(h=h_k,delay=delay) #design equalizer and get Mean Squared Error
w = zf.w # get the tap coeffs of the designed equalizer filter
# mse = zf.design(h=h_k) # Try this delay optimized equalizer

r_k=h_k # Test the equalizer with the sampled channel response as input
d_k=zf.equalize(r_k) # filter input through the eq
h_sys=zf.equalize(h_k) # overall effect of channel and equalizer
print('ZF equalizer design: N={} Delay={} error={}'.format(N,delay,mse))
print('ZF equalizer weights:{}'.format(w))

#Frequency response of channel,equalizer & overall system
from scipy.signal import freqz
Omega_1, H_F  = freqz(h_k) # frequency response of channel
Omega_2, W = freqz(w) # frequency response of equalizer
Omega_3, H_sys = freqz(h_sys) # frequency response of overall system

fig, ax = plt.subplots(nrows=1,ncols = 1)
ax.plot(Omega_1/pi,20*log(abs(H_F)/max(abs(H_F))),'g',label='channel')
ax.plot(Omega_2/pi,20*log(abs(W)/max(abs(W))),'r',label='ZF equalizer')
ax.plot(Omega_3/pi,20*log(abs(H_sys)/max(abs(H_sys))),'k',label='overall system')
ax.legend();ax.set_title('Frequency response');
ax.set_ylabel('Magnitude(dB)');
ax.set_xlabel('Normalized frequency(x $\pi$ rad/sample)');
fig.show()

#Plot equalizer input and output(time-domain response)
fig, (ax1,ax2) = plt.subplots(nrows=2,ncols = 1)
ax1.stem( np.arange(0,len(r_k)), r_k, use_line_collection=True)
ax1.set_title('Equalizer input');
ax1.set_xlabel('Samples');
ax1.set_ylabel('Amplitude');
ax2.stem( np.arange(0,len(d_k)), d_k, use_line_collection=True);
ax2.set_title('Equalizer output- N=={} Delay={} error={}'.format(N,delay,mse));
ax2.set_xlabel('Samples');
ax2.set_ylabel('Amplitude');
fig.show()
