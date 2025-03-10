"""
Script: DigiCommPy.chapter_5.mmse_equalizer_test.py
Simulation of MMSE equalizer

@author: Mathuranathan Viswanathan
Created on Aug 29, 2019
"""
import sys
sys.path.append("..")
import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from numpy import pi, log, log10, convolve


nSamp=5 #%Number of samples per symbol determines baud rate Tsym
Fs=100 # Sampling Frequency of the system
Ts=1/Fs # Sampling time
Tsym=nSamp*Ts # symbol time period

#Define transfer function of the channel
k=6 # define limits for computing channel response
N0 = 0.1 # Standard deviation of AWGN channel noise
t=np.arange(start=-k*Tsym,stop=k*Tsym,step=Ts)#time base defined till +/-kTsym
h_t = 1/(1+(t/Tsym)**2) # channel model, replace with your own model
h_t = h_t + N0*np.random.randn(len(h_t)) # add Noise to the channel response
h_k = h_t[0::nSamp] # downsampling to represent symbol rate sampler
t_inst=t[0::nSamp] # symbol sampling instants

fig, ax = plt.subplots(nrows=1,ncols = 1)
ax.plot(t,h_t,label='continuous-time model');#response at all sampling instants
# channel response at symbol sampling instants
ax.stem(t_inst,h_k,'r',label='discrete-time model',use_line_collection=True)
ax.legend();ax.set_title('Channel impulse response');
ax.set_xlabel('Time (s)');ax.set_ylabel('Amplitude');fig.show()

# Equalizer Design Parameters
N = 14 # Desired number of taps for equalizer filter

#design DELAY OPTIMIZED MMSE eq. for given channel, get tap weights and filter
#the input through the equalizer
from DigiCommPy.equalizers import MMSEEQ
noiseVariance = N0**2 # noise variance
snr = 10*log10(1/N0) # convert to SNR (assume var(signal) = 1)
mmse_eq = MMSEEQ(N) #initialize MMSE equalizer (object) of length N
mse = mmse_eq.design(h=h_k,snr=snr)#design equalizer and get Mean Squared Error
w = mmse_eq.w # get the tap coeffs of the designed equalizer filter
opt_delay = mmse_eq.opt_delay

r_k = h_k # Test the equalizer with the sampled channel response as input
d_k = mmse_eq.equalize(r_k) # filter input through the eq
h_sys = mmse_eq.equalize(h_k) # overall effect of channel and equalizer

print('MMSE equalizer design: N={} Delay={} error={}'.format(N,opt_delay,mse))
print('MMSE equalizer weights:{}'.format(w))

#Plot equalizer input and output(time-domain response)
fig, (ax1,ax2) = plt.subplots(nrows=2,ncols = 1)
ax1.stem( np.arange(0,len(r_k)), r_k, use_line_collection=True)
ax1.set_title('Equalizer input');
ax1.set_xlabel('Samples');ax1.set_ylabel('Amplitude');

ax2.stem( np.arange(0,len(d_k)), d_k, use_line_collection=True);
ax2.set_title('Equalizer output- N=={} Delay={} error={}'.format(N,opt_delay,mse));
ax2.set_xlabel('Samples');ax2.set_ylabel('Amplitude');fig.show()
