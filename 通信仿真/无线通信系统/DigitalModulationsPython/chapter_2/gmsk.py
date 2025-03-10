"""
Performance of baseband GMSK modulation scheme

@author: Mathuranathan Viswanathan
Created on Jul 30, 2019
"""
#Execute in Python3: exec(open("chapter_2/gmsk.py").read())
import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from passband_modulations import gmsk_mod,gmsk_demod
from channels import awgn 

N=100000 # Number of symbols to transmit
EbN0dB = np.arange(start=0,stop = 19, step = 2) # Eb/N0 range in dB for simulation
BTs = [0.1, 0.3 ,0.5, 1] # Gaussian LPF's BT products
fc = 800 # Carrier frequency in Hz (must be < fs/2 and > fg)
L = 16 # oversampling factor

fig, axs = plt.subplots(nrows=1,ncols = 1)
lineColors = ['g','b','k','r']

for i,BT in enumerate(BTs): 
    a = np.random.randint(2, size=N) # uniform random symbols from 0's and 1's
    (s_t,s_complex) = gmsk_mod(a,fc,L,BT) # GMSK modulation
    BER = np.zeros(len(EbN0dB)) # For BER values for each Eb/N0
    
    for j,EbN0 in enumerate(EbN0dB):
        r_complex = awgn(s_complex,EbN0) # refer Chapter section 4.1
        a_hat = gmsk_demod(r_complex,L) # Baseband GMSK demodulation
        BER[j] = np.sum(a!=a_hat)/N # Bit Error Rate Computation
    
    axs.semilogy(EbN0dB,BER,lineColors[i]+'*-',label='$BT_b=$'+str(BT))

axs.set_title('Probability of Bit Error for GMSK modulation')
axs.set_xlabel('E_b/N_0 (dB)');axs.set_ylabel('Probability of Bit Error - $P_b$')
axs.legend();fig.show()