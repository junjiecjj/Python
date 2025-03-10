"""
Performance of BFSK modulation

@author: Mathuranathan Viswanathan
Created on Aug 1, 2019
"""
#Execute in Python3: exec(open("chapter_2/bfsk.py").read())
import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from passband_modulations import bfsk_mod, bfsk_coherent_demod, bfsk_noncoherent_demod
from channels import awgn
from scipy.special import erfc

N=100000 # Number of bits to transmit
EbN0dB = np.arange(start=-4,stop = 11, step = 2) # Eb/N0 range in dB for simulation
fc = 400 # center carrier frequency f_c- integral multiple of 1/Tb
fsk_type = 'coherent' # coherent/noncoherent FSK generation at Tx
h = 1 # modulation index
# h should be minimum 0.5 for coherent FSK or multiples of 0.5
# h should be minimum 1 for non-coherent FSK or multiples of 1
L = 40 # oversampling factor
fs = 8*fc # sampling frequency for discrete-time simulation
fd = h/(L/fs) # Frequency separation

BER_coherent = np.zeros(len(EbN0dB)) # BER for coherent BFSK
BER_noncoherent = np.zeros(len(EbN0dB)) # BER for non-coherent BFSK

a = np.random.randint(2, size=N) # uniform random symbols from 0's and 1's
[s_t,phase]=bfsk_mod(a,fc,fd,L,fs,fsk_type) # BFSK modulation

for i, EbN0 in enumerate(EbN0dB):
    r_t = awgn(s_t,EbN0,L) # refer Chapter section 4.1
    
    if fsk_type.lower() == 'coherent':
        # coherent FSK could be demodulated coherently or non-coherently
        a_hat_coherent = bfsk_coherent_demod(r_t,phase,fc,fd,L,fs) # coherent demod
        a_hat_noncoherent = bfsk_noncoherent_demod(r_t,fc,fd,L,fs)#noncoherent demod
        
        BER_coherent[i] = np.sum(a!=a_hat_coherent)/N # BER for coherent case
        BER_noncoherent[i] = np.sum(a!=a_hat_noncoherent)/N # BER for non-coherent
    
    if fsk_type.lower() == 'noncoherent':
        #non-coherent FSK can only non-coherently demodulated
        a_hat_noncoherent = bfsk_noncoherent_demod(r_t,fc,fd,L,fs)#noncoherent demod
        BER_noncoherent[i] = np.sum(a!=a_hat_noncoherent)/N # BER for non-coherent

#Theoretical BERs
theory_coherent = 0.5*erfc(np.sqrt(10**(EbN0dB/10)/2)) # Theory BER - coherent
theory_noncoherent = 0.5*np.exp(-10**(EbN0dB/10)/2) # Theory BER - non-coherent

fig, axs = plt.subplots(1, 1)
if fsk_type.lower() == 'coherent':
    axs.semilogy(EbN0dB,BER_coherent,'k*',label='sim-coherent demod')
    axs.semilogy(EbN0dB,BER_noncoherent,'m*',label='sim-noncoherent demod')
    axs.semilogy(EbN0dB,theory_coherent,'r-',label='theory-coherent demod')
    axs.semilogy(EbN0dB,theory_noncoherent,'b-',label='theory-noncoherent demod')
    axs.set_title('Performance of coherent BFSK modulation')
    
if fsk_type.lower() == 'noncoherent':
    axs.semilogy(EbN0dB,BER_noncoherent,'m*',label='sim-noncoherent demod')
    axs.semilogy(EbN0dB,theory_noncoherent,'b-',label='theory-noncoherent demod')
    axs.set_title('Performance of noncoherent BFSK modulation')
    
axs.set_xlabel('$E_b/N_0$ (dB)');axs.set_ylabel('Probability of Bit Error - $P_b$')
axs.legend();fig.show()