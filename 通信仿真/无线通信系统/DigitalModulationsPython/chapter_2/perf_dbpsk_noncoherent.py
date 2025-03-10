"""
Non-coherent detection of D-BPSK with phase ambiguity in local oscillator

@author: Mathuranathan Viswanathan
Created on Jul 18, 2019
"""
#Execute in Python3 command line: exec(open("./perf_dbpsk_noncoherent.py").read())
import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from passband_modulations import bpsk_mod,add_awgn_noise
from scipy.signal import lfilter
from scipy.special import erfc

N=100000 # Number of symbols to transmit
EbN0dB = np.arange(start=-4,stop = 11,step = 2) # Eb/N0 range in dB for simulation
L=8 # oversampling factor,L=Tb/Ts(Tb=bit period,Ts=sampling period)
# if a carrier is used, use L = Fs/Fc, where Fs >> 2xFc
Fc=800 # carrier frequency
Fs=L*Fc # sampling frequency

BER_suboptimum = np.zeros(len(EbN0dB)) # BER measures
BER_optimum = np.zeros(len(EbN0dB))

#-----------------Transmitter---------------------
ak = np.random.randint(2, size=N) # uniform random symbols from 0's and 1's
bk = lfilter([1.0],[1.0,-1.0],ak) # IIR filter for differential encoding
bk = bk%2 #XOR operation is equivalent to modulo-2
[s_bb,t]= bpsk_mod(bk,L) # BPSK modulation(waveform) - baseband
s = s_bb*np.cos(2*np.pi*Fc*t/Fs).astype(complex) # DPSK with carrier

for i,EbN0 in enumerate(EbN0dB):
    # Compute and add AWGN noise
    (r,n,N0) = add_awgn_noise(s,EbN0,L) # refer Chapter section 4.1
    
    #----------suboptimum receiver---------------
    p=np.real(r)*np.cos(2*np.pi*Fc*t/Fs) # demodulate to baseband using BPF
    w0= np.hstack((p,np.zeros(L))) # append L samples on one arm for equal lengths
    w1= np.hstack((np.zeros(L),p)) # delay the other arm by Tb (L samples)
    w = w0*w1 # multiplier
    z = np.convolve(w,np.ones(L)) #integrator from kTb to (K+1)Tb (L samples)    
    u =  z[L-1:-1-L:L] # sampler t=kTb
    ak_hat = (u<0) #decision    
    BER_suboptimum[i] = np.sum(ak!=ak_hat)/N #BER for suboptimum receiver

    #-----------optimum receiver--------------
    p=np.real(r)*np.cos(2*np.pi*Fc*t/Fs); # multiply I arm by cos
    q=np.imag(r)*np.sin(2*np.pi*Fc*t/Fs) # multiply Q arm by sin
    x = np.convolve(p,np.ones(L)) # integrate I-arm by Tb duration (L samples)
    y = np.convolve(q,np.ones(L)) # integrate Q-arm by Tb duration (L samples)
    xk = x[L-1:-1:L] # Sample every Lth sample
    yk = y[L-1:-1:L] # Sample every Lth sample
    w0 = xk[0:-2] # non delayed version on I-arm
    w1 = xk[1:-1] # 1 bit delay on I-arm
    z0 = yk[0:-2] # non delayed version on Q-arm
    z1 = yk[1:-1] # 1 bit delay on Q-arm
    u =w0*w1 + z0*z1 # decision statistic
    ak_hat=(u<0) # threshold detection
    BER_optimum[i] = np.sum(ak[1:-1]!=ak_hat)/N # BER for optimum receiver
#------Theoretical Bit/Symbol Error Rates-------------
EbN0lins = 10**(EbN0dB/10) # converting dB values to linear scale
theory_DBPSK_optimum = 0.5*np.exp(-EbN0lins)
theory_DBPSK_suboptimum = 0.5*np.exp(-0.76*EbN0lins)
theory_DBPSK_coherent=erfc(np.sqrt(EbN0lins))*(1-0.5*erfc(np.sqrt(EbN0lins)))
theory_BPSK_conventional = 0.5*erfc(np.sqrt(EbN0lins))
#-------------Plotting---------------------------
fig, ax = plt.subplots(nrows=1,ncols = 1)
ax.semilogy(EbN0dB,BER_suboptimum,'k*',label='DBPSK subopt (sim)')
ax.semilogy(EbN0dB,BER_optimum,'b*',label='DBPSK opt (sim)')
ax.semilogy(EbN0dB,theory_DBPSK_suboptimum,'m-',label='DBPSK subopt (theory)')
ax.semilogy(EbN0dB,theory_DBPSK_optimum,'r-',label='DBPSK opt (theory)')
ax.semilogy(EbN0dB,theory_DBPSK_coherent,'k-',label='coherent DEBPSK')
ax.semilogy(EbN0dB,theory_BPSK_conventional,'b-',label='coherent BPSK')
#set(gca,'XLim',[-4 12]);set(gca,'YLim',[1E-6 1E0]);
#set(gca,'XTick',-4:2:12)
ax.set_title('Probability of D-BPSK over AWGN')
ax.set_xlabel('$E_b/N_0 (dB)$');ax.set_ylabel('$Probability of Bit Error - P_b$')
ax.legend();fig.show()