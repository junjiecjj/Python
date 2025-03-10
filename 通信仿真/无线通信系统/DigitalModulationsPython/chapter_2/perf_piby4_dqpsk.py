"""
Performance of Pi/4 DQPSK (waveform simulation)

@author: Mathuranathan Viswanathan
Created on Jul 23, 2019
"""
#Execute in Python3 command line: exec(open("./perf_piby4_dqpsk.py").read())
import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from passband_modulations import piBy4_dqpsk_mod,add_awgn_noise,piBy4_dqpsk_demod
from scipy.special import erfc

N=1000000 # Number of symbols to transmit
EbN0dB = np.arange(start=-4,stop = 11,step = 2) # Eb/N0 range in dB for simulation
fc=100 # carrier frequency in Hertz
OF =8 # oversampling factor, sampling frequency will be fs=OF*fc

BER = np.zeros(len(EbN0dB)) # For BER values for each Eb/N0

a = np.random.randint(2, size=N) # uniform random symbols from 0's and 1's
result = piBy4_dqpsk_mod(a,fc,OF,enable_plot=False)# dqpsk modulation
s = result['s(t)'] # get values from returned dictionary

for i,EbN0 in enumerate(EbN0dB):
    # Compute and add AWGN noise
    (r,n,N0) = add_awgn_noise(s,EbN0,OF) # refer Chapter section 4.1
    a_hat = piBy4_dqpsk_demod(r,fc,OF,enable_plot=False)
    BER[i] = np.sum(a!=a_hat)/N # Bit Error Rate Computation
#------Theoretical Bit Error Rate-------------
x = np.sqrt(4*10**(EbN0dB/10))*np.sin(np.pi/(4*np.sqrt(2)))
theoreticalBER = 0.5*erfc(x/np.sqrt(2))
#-------------Plot performance curve------------------------
fig, axs = plt.subplots(nrows=1,ncols = 1)
axs.semilogy(EbN0dB,BER,'k*',label='Simulated')
axs.semilogy(EbN0dB,theoreticalBER,'r-',label='Theoretical')
axs.set_title('Probability of Bit Error for $\pi/4$-DQPSK');
axs.set_xlabel(r'$E_b/N_0$ (dB)')
axs.set_ylabel(r'Probability of Bit Error - $P_b$');
axs.legend();fig.show()