"""
Compare PSDs of bandpass MSK QPSK BPSK signals

@author: Mathuranathan Viswanathan
Created on Jul 29, 2019
"""
import numpy as np
import matplotlib.pyplot as plt

def bpsk_qpsk_msk_psd():
    # Usage:
    #    >> from chapter_2.psd_estimates import bpsk_qpsk_msk_psd
    #    >> bpsk_qpsk_msk_psd()
    from passband_modulations import bpsk_mod,qpsk_mod,msk_mod
    from essentials import plotWelchPSD    
    N=100000 # Number of symbols to transmit
    fc=800;OF =8 # carrier frequency and oversamping factor
    fs = fc*OF # sampling frequency
    
    a = np.random.randint(2, size=N) # uniform random symbols from 0's and 1's
    (s_bb,t) = bpsk_mod(a,OF) # BPSK modulation(waveform) - baseband
    s_bpsk = s_bb*np.cos(2*np.pi*fc*t/fs) # BPSK with carrier
    s_qpsk = qpsk_mod(a,fc,OF)['s(t)'] # conventional QPSK
    s_msk = msk_mod(a,fc,OF)['s(t)'] # MSK signal
    
    # Compute and plot PSDs for each of the modulated versions
    fig, ax = plt.subplots(1, 1)
    plotWelchPSD(s_bpsk,fs,fc,ax = ax,color = 'b',label='BPSK')
    plotWelchPSD(s_qpsk,fs,fc,ax = ax,color = 'r',label='QPSK')
    plotWelchPSD(s_msk,fs,fc, ax = ax,color = 'k',label='MSK')
    ax.set_xlabel('$f-f_c$');ax.set_ylabel('PSD (dB/Hz)');ax.legend();fig.show()

def gmsk_psd():
    from passband_modulations import gmsk_mod
    from essentials import plotWelchPSD
    
    N = 10000 # Number of symbols to transmit
    fc = 800 # carrier frequency in Hertz
    L = 16 # oversampling factor,use L= Fs/Fc, where Fs >> 2xFc
    fs = L*fc
    a = np.random.randint(2, size=N) # uniform random symbols from 0's and 1's
    
    #'_':unused output variable
    (s1 , _ ) = gmsk_mod(a,fc,L,BT=0.3, enable_plot=True) # BT_b=0.3
    (s2 , _ ) = gmsk_mod(a,fc,L,BT=0.5) # BT_b=0.5
    (s3 , _ ) = gmsk_mod(a,fc,L,BT=0.7) # BT_b=0.7
    (s4 , _ ) = gmsk_mod(a,fc,L,BT=10000) # BT_b=very value value (MSK)
    
    # Compute and plot PSDs for each of the modulated versions
    fig, ax = plt.subplots(1, 1)
    plotWelchPSD(s1,fs,fc, ax = ax , color = 'r', label = '$BT_b=0.3$')
    plotWelchPSD(s2,fs,fc, ax = ax , color = 'b', label = '$BT_b=0.5$')
    plotWelchPSD(s3,fs,fc, ax = ax , color = 'm', label = '$BT_b=0.7$')
    plotWelchPSD(s4,fs,fc, ax = ax , color = 'k', label = '$BT_b=\infty$')    
    ax.set_xlabel('$f-f_c$'); ax.set_ylabel('PSD (dB/Hz)')
    ax.legend(); fig.show()  