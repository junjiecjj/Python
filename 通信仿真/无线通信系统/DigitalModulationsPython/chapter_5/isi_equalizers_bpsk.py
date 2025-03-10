"""
Script: DigiCommPy.chapter_5.isi_equalizers_bpsk.py
Demonstration of Eb/N0 Vs SER for baseband BPSK modulation scheme over different ISI channels with MMSE and ZF equalizers

@author: Mathuranathan Viswanathan
Created on Aug 29, 2019
"""

import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt #for plotting functions
from DigiCommPy.modem import PSKModem #import PSKModem
from DigiCommPy.channels import awgn
from DigiCommPy.equalizers import zeroForcing, MMSEEQ #import MMSE equalizer class
from DigiCommPy.errorRates import ser_awgn #for theoretical BERs
from scipy.signal import freqz
#---------Input Fields------------------------
N=10**6 # Number of bits to transmit
EbN0dBs = np.arange(start=0,stop=30,step=2) #  Eb/N0 range in dB for simulation
M=2 # 2-PSK
#h_c=[0.04, -0.05, 0.07, -0.21, -0.5, 0.72, 0.36, 0.21, 0.03, 0.07] # Channel A
#h_c=[0.407, 0.815, 0.407] # uncomment this for Channel B
h_c=[0.227, 0.460, 0.688, 0.460, 0.227] # uncomment this for Channel C
nTaps = 31 # Desired number of taps for equalizer filter
SER_zf = np.zeros(len(EbN0dBs)); SER_mmse = np.zeros(len(EbN0dBs))
#-----------------Transmitter---------------------
inputSymbols=np.random.randint(low=0,high=2,size=N) #uniform random symbols 0s & 1s
modem = PSKModem(M)
modulatedSyms = modem.modulate(inputSymbols)
x = np.convolve(modulatedSyms,h_c) # apply channel effect on transmitted symbols

for i,EbN0dB in enumerate(EbN0dBs):
    receivedSyms = awgn(x,EbN0dB) #add awgn noise

    # DELAY OPTIMIZED MMSE equalizer
    mmse_eq = MMSEEQ(nTaps) #initialize MMSE equalizer (object) of length nTaps
    mmse_eq.design(h_c,EbN0dB) #Design MMSE equalizer
    optDelay = mmse_eq.opt_delay #get the optimum delay of the equalizer
    #filter received symbols through the designed equalizer
    equalizedSamples = mmse_eq.equalize(receivedSyms)
    y_mmse=equalizedSamples[optDelay:optDelay+N] # samples from optDelay position

    # DELAY OPTIMIZED ZF equalizer
    zf_eq = zeroForcing(nTaps) #initialize ZF equalizer (object) of length nTaps
    zf_eq.design(h_c) #Design ZF equalizer
    optDelay = zf_eq.opt_delay #get the optimum delay of the equalizer
    #filter received symbols through the designed equalizer
    equalizedSamples = zf_eq.equalize(receivedSyms)
    y_zf = equalizedSamples[optDelay:optDelay+N] # samples from optDelay position

    # Optimum Detection in the receiver - Euclidean distance Method
    estimatedSyms_mmse = modem.demodulate(y_mmse)
    estimatedSyms_zf = modem.demodulate(y_zf)
    # SER when filtered thro MMSE eq.
    SER_mmse[i]=sum((inputSymbols != estimatedSyms_mmse))/N
    # SER when filtered thro ZF eq.
    SER_zf[i]=sum((inputSymbols != estimatedSyms_zf))/N

SER_theory = ser_awgn(EbN0dBs,'PSK',M=2) #theoretical SER

fig1, ax1 = plt.subplots(nrows=1,ncols = 1)
ax1.semilogy(EbN0dBs,SER_zf,'g',label='ZF Equalizer');
ax1.semilogy(EbN0dBs,SER_mmse,'r',label='MMSE equalizer')
ax1.semilogy(EbN0dBs,SER_theory,'k',label='No interference')
ax1.set_title('Probability of Symbol Error for BPSK signals');
ax1.set_xlabel('$E_b/N_0$(dB)');ax1.set_ylabel('Probability of Symbol Error-$P_s$')
ax1.legend(); ax1.set_ylim(bottom=10**-4, top=1);fig1.show()

# compute and plot channel characteristics
Omega, H_c  = freqz(h_c) #frequency response of the channel
fig2, (ax2,ax3) = plt.subplots(nrows=1,ncols = 2)
ax2.stem(h_c,use_line_collection=True) # time domain
ax3.plot(Omega,20*np.log10(abs(H_c)/max(abs(H_c))));fig2.show()
