#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 17:39:10 2025

@author: jack
"""



#%% Program 69: DigiCommPy\chapter 5\zf equalizer test.py: Simulation of zero-forcing equalizer
import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from numpy import pi,log,convolve
from DigiCommPy.equalizers import zeroForcing
nSamp = 5 #%Number of samples per symbol determines baud rate Tsym
Fs = 100 # Sampling Frequency of the system
Ts = 1/Fs # Sampling time
Tsym = nSamp*Ts # symbol time period

#Define transfer function of the channel
k = 6 # define limits for computing channel response
N0 = 0.001 # Standard deviation of AWGN channel noise
t = np.arange(start = -k*Tsym, stop = k*Tsym, step = Ts) # time base defined till +/-kTsym
h_t = 1/(1+(t/Tsym)**2) # channel model, replace with your own model
h_t = h_t + N0*np.random.randn(len(h_t)) # add Noise to the channel response
h_k = h_t[0::nSamp] # downsampling to represent symbol rate sampler
t_inst=t[0::nSamp] # symbol sampling instants

fig, ax = plt.subplots(nrows=1, ncols = 1)
ax.plot(t, h_t, label = 'continuous-time model');#response at sampling instants
# channel response at symbol sampling instants
ax.stem(t_inst, h_k, 'r', label='discrete-time model', )
ax.legend()
ax.set_title('Channel impulse response')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
plt.show()
plt.close()

# Equalizer Design Parameters
N = 14 # Desired number of taps for equalizer filter
delay = 11

# design zero-forcing equalizer for given channel and get tap weights and
# filter the input through the equalizer find equalizer co-effs for given CIR

zf = zeroForcing(N) #initialize ZF equalizer (object) of length N
mse = zf.design(h = h_k, delay = delay) #design equalizer and get Mean Squared Error
w = zf.w # get the tap coeffs of the designed equalizer filter
# mse = zf.design(h=h_k) # Try this delay optimized equalizer

r_k = h_k # Test the equalizer with the sampled channel response as input
d_k = zf.equalize(r_k) # filter input through the eq
h_sys = zf.equalize(h_k) # overall effect of channel and equalizer
print('ZF equalizer design: N={} Delay={} error={}'.format(N, delay, mse))
print('ZF equalizer weights:{}'.format(w))

#Frequency response of channel,equalizer & overall system
from scipy.signal import freqz
Omega_1, H_F  = freqz(h_k) # frequency response of channel
Omega_2, W = freqz(w) # frequency response of equalizer
Omega_3, H_sys = freqz(h_sys) # frequency response of overall system

fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.plot(Omega_1/pi, 20*log(abs(H_F)/max(abs(H_F))),'g', label = 'channel')
ax.plot(Omega_2/pi, 20*log(abs(W)/max(abs(W))),'r', label = 'ZF equalizer')
ax.plot(Omega_3/pi, 20*log(abs(H_sys)/max(abs(H_sys))), 'k', label = 'overall system')
ax.legend()
ax.set_title('Frequency response');
ax.set_ylabel('Magnitude(dB)');
ax.set_xlabel('Normalized frequency(x $\pi$ rad/sample)');
plt.show()
plt.close()

#Plot equalizer input and output(time-domain response)
fig, (ax1,ax2) = plt.subplots(nrows = 2, ncols = 1, figsize = (6, 8), constrained_layout = True)
ax1.stem( np.arange(0,len(r_k)), r_k,  )
ax1.set_title('Equalizer input')
ax1.set_xlabel('Samples')
ax1.set_ylabel('Amplitude')
ax2.stem( np.arange(0,len(d_k)), d_k, )
ax2.set_title('Equalizer output- N=={} Delay={} error={}'.format(N, delay, mse))
ax2.set_xlabel('Samples')
ax2.set_ylabel('Amplitude')
plt.show()
plt.close()

#%% Program 71: DigiCommPy\chapter 5\mmse equalizer test.py: Simulation of MMSE equalizer

import sys
sys.path.append("..")
import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from DigiCommPy.equalizers import MMSEEQ

nSamp=5 #%Number of samples per symbol determines baud rate Tsym
Fs=100 # Sampling Frequency of the system
Ts=1/Fs # Sampling time
Tsym=nSamp*Ts # symbol time period

#Define transfer function of the channel
k=6 # define limits for computing channel response
N0 = 0.1 # Standard deviation of AWGN channel noise
t=np.arange(start = -k*Tsym, stop = k*Tsym, step = Ts)#time base defined till +/-kTsym
h_t = 1/(1+(t/Tsym)**2) # channel model, replace with your own model
h_t = h_t + N0*np.random.randn(len(h_t)) # add Noise to the channel response
h_k = h_t[0::nSamp] # downsampling to represent symbol rate sampler
t_inst=t[0::nSamp] # symbol sampling instants

fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.plot(t, h_t, label='continuous-time model');#response at all sampling instants
# channel response at symbol sampling instants
ax.stem(t_inst, h_k, 'r', label = 'discrete-time model', )
ax.legend()
ax.set_title('Channel impulse response');
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
plt.show()
plt.close()
# Equalizer Design Parameters
N = 14 # Desired number of taps for equalizer filter

#design DELAY OPTIMIZED MMSE eq. for given channel, get tap weights and filter
#the input through the equalizer

noiseVariance = N0**2 # noise variance
snr = 10*np.log10(1/N0) # convert to SNR (assume var(signal) = 1)
mmse_eq = MMSEEQ(N) #initialize MMSE equalizer (object) of length N
mse = mmse_eq.design(h = h_k, snr = snr)#design equalizer and get Mean Squared Error
w = mmse_eq.w # get the tap coeffs of the designed equalizer filter
opt_delay = mmse_eq.opt_delay

r_k = h_k # Test the equalizer with the sampled channel response as input
d_k = mmse_eq.equalize(r_k) # filter input through the eq
h_sys = mmse_eq.equalize(h_k) # overall effect of channel and equalizer

print('MMSE equalizer design: N={} Delay={} error={}'.format(N, opt_delay, mse))
print('MMSE equalizer weights:{}'.format(w))

#Plot equalizer input and output(time-domain response)
fig, (ax1,ax2) = plt.subplots(nrows=2, ncols = 1)
ax1.stem( np.arange(0, len(r_k)), r_k,  )
ax1.set_title('Equalizer input')
ax1.set_xlabel('Samples')
ax1.set_ylabel('Amplitude');

ax2.stem( np.arange(0,len(d_k)), d_k, );
ax2.set_title('Equalizer output- N=={} Delay={} error={}'.format(N,opt_delay,mse))
ax2.set_xlabel('Samples');ax2.set_ylabel('Amplitude')
plt.show()
plt.close()

#%% Program 72: DigiCommPy\chapter 5\mmse eq delay opti.py: Delay optimization of MMSE eq.

import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt #for plotting functions
from DigiCommPy.equalizers import MMSEEQ

h=np.array([-0.1, -0.3, 0.4, 1, 0.4, 0.3, -0.1]) # test channel
SNR=10 # Signal-to-noise ratio at the equalizer input in dB
Ns= np.arange(start=5, stop=35, step=5) # sweep number of equalizer taps from 5 to 30
maxDelay=Ns[-1]+len(h)-2 #max delay cannot exceed this value
optimalDelay=np.zeros(len(Ns));

fig, ax = plt.subplots(nrows=1,ncols = 1)
for i,N in enumerate(Ns): #sweep number of equalizer taps
    maxDelay = N+len(h)-2
    mse=np.zeros(maxDelay)
    for j, delay in enumerate(range(0,maxDelay)): # sweep delays
        # compute MSE and optimal delay for each combination
        mmse_eq = MMSEEQ(N) #initialize MMSE equalizer (object) of length N
        mse[j] = mmse_eq.design(h, SNR, delay)
        optimalDelay[i] = mmse_eq.opt_delay
    #plot mse in log scale
    ax.plot(np.arange(0, maxDelay), np.log10(mse), label = 'N='+str(N))
ax.set_title('MSE Vs eq. delay for given channel and equalizer lengths')
ax.set_xlabel('Equalizer delay');ax.set_ylabel('$log_{10}$[mse]');
ax.legend()
plt.show()
plt.close()
#display optimal delays for each selected filter length N. this will correspond
#with the bottom of the buckets displayed in the plot
print('Optimal Delays for each N value ->{}'.format(optimalDelay))

#%% Program 73: DigiCommPy\chapter 5\isi equalizers bpsk.py: Performance of linear equalizers

import sys
sys.path.append("..")
import numpy as np
import scipy
import matplotlib.pyplot as plt #for plotting functions
from DigiCommPy.modem import PSKModem #import PSKModem
from DigiCommPy.channels import awgn
from DigiCommPy.equalizers import zeroForcing, MMSEEQ #import MMSE equalizer class
from DigiCommPy.errorRates import ser_awgn #for theoretical BERs
from scipy.signal import freqz
#---------Input Fields------------------------
h_cA = np.array([0.04, -0.05, 0.07, -0.21, -0.5, 0.72, 0.36, 0.21, 0.03, 0.07]) #  Channel A
h_cB = np.array([0.407, 0.815, 0.407])               #  Channel B
h_cC = np.array([0.227, 0.460, 0.688, 0.460, 0.227]) #  Channel C

channelTypes = ["Channel A", "Channel B", "Channel C" ]
H_C = {}
H_C[0] = h_cA
H_C[1] = h_cB
H_C[2] = h_cC
markers = ['none', "o", 'v', ]
colors = ['k', 'r', 'b']

## compute and plot channel characteristics
for idx, channeltype in enumerate(channelTypes):
    h_c = H_C[idx]
    F, H = scipy.signal.freqz(h_c)
    ##### plot
    fig, axs = plt.subplots(1, 2, figsize = (12, 4), constrained_layout = True)
    axs[0].stem(h_c, linefmt = f'{colors[idx]}-', markerfmt = 'D', )
    axs[0].set_xlabel('Time(s)',)
    axs[0].set_ylabel('h(t)',)
    axs[0].set_title(f"Channel impulse response, {channeltype}" )
    axs[1].plot(F , 20*np.log10(np.abs(H)/np.max(np.abs(H))), color = colors[idx],  )
    axs[1].set_xlabel('Samples',)
    axs[1].set_ylabel('Amplitude',)
    axs[1].set_title( "Frequency response" )
    plt.show()
    plt.close()

N = 100000
EbN0dBs = np.arange(0, 32, 2)
nTaps = 31 # Desired number of taps for equalizer filter
# ntaps = 31
MOD_TYPE = "psk"
M = 2

u = np.random.randint(low = 0, high = 2, size = N) #uniform random symbols 0s & 1s
modem = PSKModem(M)
s = modem.modulate(u)

fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)
SER_theory = ser_awgn(EbN0dBs, MOD_TYPE, M)
for idx, channeltype in enumerate(channelTypes):
    SER_zf = np.zeros(EbN0dBs.size)
    SER_mmse = np.zeros(EbN0dBs.size)
    h_c = H_C[idx]
    x = scipy.signal.convolve(s, h_c)
    for i, EbN0dB in enumerate(EbN0dBs):
        ## channel
        r = awgn(x, EbN0dB)

        ## Receiver
        ## MMSE equalizer
        mmse_eq = MMSEEQ(nTaps)       #initialize MMSE equalizer (object) of length nTaps
        mmse_eq.design(h_c, EbN0dB)   #Design MMSE equalizer
        optDelay = mmse_eq.opt_delay  #get the optimum delay of the equalizer
        #filter received symbols through the designed equalizer
        equalizedSamples = mmse_eq.equalize(r)
        y_mmse = equalizedSamples[optDelay:optDelay+N] # samples from optDelay position
        u_mmse = modem.demodulate(y_mmse)
        #  ZF equalizer
        zf_eq = zeroForcing(nTaps) #initialize ZF equalizer (object) of length nTaps
        zf_eq.design(h_c)             # Design ZF equalizer
        optDelay = zf_eq.opt_delay    # get the optimum delay of the equalizer
        #filter received symbols through the designed equalizer
        equalizedSamples = zf_eq.equalize(r)
        y_zf = equalizedSamples[optDelay:optDelay+N] # samples from optDelay position
        u_zf = modem.demodulate(y_zf)

        # SER when filtered thro MMSE eq.
        SER_mmse[i] = np.sum((u != u_mmse))/N
        # SER when filtered thro ZF eq.
        SER_zf[i] = np.sum((u != u_zf))/N

    axs.semilogy(EbN0dBs, SER_zf, color = 'g', ls = '-', marker = markers[idx], ms = 12, label = f'{channeltype}, ZF eq.')
    axs.semilogy(EbN0dBs, SER_mmse, color = 'r', ls = '-', marker = markers[idx], ms = 12, label = f'{channeltype}, MMSE eq.')
axs.semilogy(EbN0dBs, SER_theory, color = 'k', ls = '-', label = f'{M}-{MOD_TYPE.upper()}' )

axs.set_ylim(1e-4, 1)
axs.set_xlabel( 'Eb/N0(dB)',)
axs.set_ylabel('SER',)
axs.set_title( "Probability of Symbol Error for BPSK signals")
axs.legend(fontsize = 20)

plt.show()
plt.close()

#%% Program 75: DigiCommPy\chapter 5\lms test.py: Verifying the LMS algorithm
import numpy as np
from numpy.random import randn
from numpy import convolve
from DigiCommPy.equalizers import LMSEQ

N = 5 # length of the desired filter
mu = 0.1 # step size for LMS algorithm
r = randn(10000) # random input sequence of length 10000
h = randn(N) + 1j*randn(N) # random complex system
a = convolve(h, r) # reference signal

lms_eq = LMSEQ(N) #initialize the LMS filter object
lms_eq.design(mu, r, a) # design using input and reference sequences
print('System impulse response (h): {}'.format(h))
print('LMS adapted filter (w): {}'.format(lms_eq.w))


# #%%
# N = 100000
# mu = 0.1
# EbN0dBs = np.arange(0, 32, 2)
# nTaps = 31 # Desired number of taps for equalizer filter
# h_c = np.array([0.227, 0.460, 0.688, 0.460, 0.227])

# MOD_TYPE = "psk"
# M = 2

# SER_zf = np.zeros(len(EbN0dBs))

# #-----------------Transmitter---------------------
# inputSymbols = np.random.randint(low = 0, high = 2, size = N) #uniform random symbols 0s & 1s
# modem = PSKModem(M)
# modulatedSyms = modem.modulate(inputSymbols)
# x = np.convolve(modulatedSyms, h_c) # apply channel effect on transmitted symbols

# for i,EbN0dB in enumerate(EbN0dBs):
#     receivedSyms = awgn(x, EbN0dB) #add awgn noise

#     # DELAY OPTIMIZED MMSE equalizer
#     lmsq_eq = LMSEQ(nTaps) #initialize MMSE equalizer (object) of length nTaps
#     lmsq_eq.design(mu, modulatedSyms, receivedSyms) #Design MMSE equalizer
#     # optDelay = lmsq_eq.opt_delay #get the optimum delay of the equalizer
#     #filter received symbols through the designed equalizer
#     equalizedSamples = lmsq_eq.equalize(receivedSyms)
#     y_mmse = equalizedSamples[optDelay:optDelay+N] # samples from optDelay position

#     # Optimum Detection in the receiver - Euclidean distance Method
#     estimatedSyms_mmse = modem.demodulate(y_mmse)
#     estimatedSyms_zf = modem.demodulate(y_zf)
#     # SER when filtered thro MMSE eq.
#     SER_mmse[i] = np.sum((inputSymbols != estimatedSyms_mmse))/N

# SER_theory = ser_awgn(EbN0dBs,'PSK',M=2) #theoretical SER

# fig1, ax1 = plt.subplots(nrows=1,ncols = 1)
# ax1.semilogy(EbN0dBs, SER_mmse, 'r', label = 'MMSE equalizer')
# ax1.semilogy(EbN0dBs, SER_theory, 'k', label = 'No interference')
# ax1.set_title('Probability of Symbol Error for BPSK signals');
# ax1.set_xlabel('$E_b/N_0$(dB)')
# ax1.set_ylabel('Probability of Symbol Error-$P_s$')
# ax1.legend()
# ax1.set_ylim(bottom = 10**-4, top = 1)
# plt.show()
# plt.close()
















































































































































































































































































