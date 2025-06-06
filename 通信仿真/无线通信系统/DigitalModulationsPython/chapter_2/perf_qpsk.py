"""
Performance of conventional QPSK (waveform simulation)

@author: Mathuranathan Viswanathan
Created on Jul 22, 2019
"""
#Execute in Python3 command line: exec(open("./perf_qpsk.py").read())
import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from passband_modulations import qpsk_mod,add_awgn_noise,qpsk_demod
from scipy.special import erfc

N=100000 # Number of symbols to transmit
EbN0dB = np.arange(start=-4,stop = 11,step = 2) # Eb/N0 range in dB for simulation
fc=100 # carrier frequency in Hertz
OF =8 # oversampling factor, sampling frequency will be fs=OF*fc

BER = np.zeros(len(EbN0dB)) # For BER values for each Eb/N0

a = np.random.randint(2, size=N) # uniform random symbols from 0's and 1's
result = qpsk_mod(a,fc,OF,enable_plot=False) #QPSK modulation
s = result['s(t)'] # get values from returned dictionary
for i,EbN0 in enumerate(EbN0dB):
    # Compute and add AWGN noise
    (r,n,N0) = add_awgn_noise(s,EbN0,OF) # refer Chapter section 4.1
    a_hat = qpsk_demod(r,fc,OF) # QPSK demodulation
    BER[i] = np.sum(a!=a_hat)/N # Bit Error Rate Computation
#------Theoretical Bit Error Rate-------------
theoreticalBER = 0.5*erfc(np.sqrt(10**(EbN0dB/10)))
#-------------Plot performance curve------------------------
fig, axs = plt.subplots(nrows=1,ncols = 1)
axs.semilogy(EbN0dB,BER,'k*',label='Simulated')
axs.semilogy(EbN0dB,theoreticalBER,'r-',label='Theoretical')
axs.set_title('Probability of Bit Error for QPSK modulation');
axs.set_xlabel(r'$E_b/N_0$ (dB)')
axs.set_ylabel(r'Probability of Bit Error - $P_b$');
axs.legend();fig.show()