"""
Coherent detection of D-BPSK with phase ambiguity in local oscillator

@author: Mathuranathan Viswanathan
Created on Jul 18, 2019
"""
#Execute in Python3 command line: exec(open("./perf_dbpsk_coherent.py").read())
import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from passband_modulations import bpsk_mod,add_awgn_noise,bpsk_demod 
from scipy.signal import lfilter
from scipy.special import erfc

N=1000000 # Number of symbols to transmit
EbN0dB = np.arange(start=-4,stop = 11,step = 2) # Eb/N0 range in dB for simulation
L=16 # oversampling factor,L=Tb/Ts(Tb=bit period,Ts=sampling period)
# if a carrier is used, use L = Fs/Fc, where Fs >> 2xFc
Fc=800 # carrier frequency
Fs=L*Fc # sampling frequency

SER = np.zeros(len(EbN0dB)) # for SER values for each Eb/N0

ak = np.random.randint(2, size=N) # uniform random symbols from 0's and 1's
bk = lfilter([1.0],[1.0,-1.0],ak) # filter for differential encoding
bk = bk%2 #XOR operation is equivalent to modulo-2

[s_bb,t]= bpsk_mod(bk,L) # BPSK modulation(waveform) - baseband
s = s_bb*np.cos(2*np.pi*Fc*t/Fs) # DPSK with carrier

for i,EbN0 in enumerate(EbN0dB):
    # Compute and add AWGN noise
    (r,n,N0) = add_awgn_noise(s,EbN0,L) # refer Chapter section 4.1
    
    phaseAmbiguity=np.pi # 180* phase ambiguity of Costas loop
    r_bb=r*np.cos(2*np.pi*Fc*t/Fs+phaseAmbiguity) # recoveredsignal
    b_hat=bpsk_demod(r_bb,L) # baseband correlation type demodulator
    a_hat=lfilter([1.0,1.0],[1.0],b_hat) # FIR for differential decoding
    a_hat= a_hat % 2 # binary messages, therefore modulo-2
    SER[i] = np.sum(ak !=a_hat)/N #Symbol Error Rate Computation    
#------Theoretical Bit/Symbol Error Rates-------------
EbN0lins = 10**(EbN0dB/10) # converting dB values to linear scale
theorySER_DPSK = erfc(np.sqrt(EbN0lins))*(1-0.5*erfc(np.sqrt(EbN0lins)))
theorySER_BPSK = 0.5*erfc(np.sqrt(EbN0lins))
#-------------Plots---------------------------
fig, ax = plt.subplots(nrows=1,ncols = 1)
ax.semilogy(EbN0dB,SER,'k*',label='Coherent D-BPSK(sim)')
ax.semilogy(EbN0dB,theorySER_DPSK,'r-',label='Coherent D-BPSK(theory)')
ax.semilogy(EbN0dB,theorySER_BPSK,'b-',label='Conventional BPSK')
ax.set_title('Probability of Bit Error for BPSK over AWGN');
ax.set_xlabel(r'$E_b/N_0$ (dB)');ax.set_ylabel(r'Probability of Bit Error - $P_b$');
ax.legend();fig.show()