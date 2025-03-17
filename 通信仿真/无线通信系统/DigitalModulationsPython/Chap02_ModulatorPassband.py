#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 17:37:07 2025

@author: Junjie Chen

关于模拟通信，数字基带传输，数字通带传输，均衡，OFDM的一些思考与总结：
(一) 模拟通信：主要是将信息承载信号(调制信号)加载到载波上，可以分为：调幅(AM)、调频(FM)，调相(PM)，后两者统称为角度调制。调幅就是用调制信号乘以载波；调频就是用调制信号的积分作为载波的相位；调相就是用调制信号作为载波的相位；三种模拟调制都可以用相干解调和非相干解调。相干解调主要是需要使用锁相环实现载波同步进而相干解调，而飞相干解调主要是通过Hilbert信号得到包络或者相位；
(二)  数字传输系统的数字信息既可以是来自计算机等终端的数字信号，也可以是模拟信号数字化处理化处理后的脉冲编码信号，这些信号所占据的频率是从零开始或很低频率开始，成为数字基带信号。数字基带传输的含义非常丰富：
     (1) 教科书上主要是从无码间串扰的角度阐述的，也就是奈奎斯特第一准则，使用脉冲成型和匹配滤波实现无码间串扰ISI。.
     (2) 而现在的很多通信书籍主要是将通信一步步模型抽象为 y = hx + n 的角度阐述的。
(三) 数字带通传输: 实际中大多数信道(如无线)因具有带通特性而不能直接传送基带信号，这是因为数字基带信号含有丰富的低频分量。为了使数字信号在带通信道中传输，必须使用数字基带信号对载波进行调制，以使得信号与信道匹配。这种用数字基带信号控制载波把数字基带信号变为数字带通信号的过程成为数字调制。把包括调制和解调过程的数字传输系统叫做数字带通传输系统。包括 幅移键控ASK、频移键控FSK、相移键控PSK已经新型数字带通调制技术：最小频移键控MSK和高斯最小频移键控GMSK。

(四) 均衡:首先理清楚什么时候需要均衡，均衡用在哪里。虽然从奈奎斯特第一准则中理论上找到了实现无ISI的方法，但是实际实现时难免存在滤波器的设计误差和信道特性的变化，无法实现理想的传输特性，故在抽样时刻总会存在一定的码间串扰，从而导致性能的下降。为了减少码间串扰的影响，通常在系统中插入一种可调滤波器来矫正或者补偿系统特性，这种起补偿作用的滤波器成为均衡器。通常在均衡系统中，已经把发送端滤波器，信道和接收方滤波器看成一个整体h(t)，y[n] = x[t]*h[t] + n[t]，注意，这时候其实是基带的模型，已经把模拟调制载波等细节都隐含在了h[t]中，信道其实已经是考虑了衰落等的。发送符号 -> 与综合信道卷积 -> +AWGN -> 与均衡系数卷积 -> 解调。

(五) OFDM: 在多径衰落的无线信道上，也会产生码间串扰，这时候为了解决这个问题，除了均衡器之外，还可以采用OFDM。注意，OFDM也是基带模型，发送符号 -> IFFT -> Add CP -> 信道卷积 (计算频域信道) -> AWGN -> Remove CP -> FFT -> 除以频域信道 -> 解调。

"""

#%% Program 19: DigiCommPy\chapter 2\bpsk.py: Performance of BPSK using waveform simulation
import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from scipy.special import erfc
from DigiCommPy.passband_modulations import bpsk_mod, bpsk_demod
from DigiCommPy.channels import awgn

N = 100000 # Number of symbols to transmit
EbN0dB = np.arange(start = -4, stop = 11, step = 2) # Eb/N0 range in dB for simulation
L = 16 # oversampling factor, L=Tb/Ts(Tb=bit period,Ts=sampling period)

# if a carrier is used, use L = Fs/Fc, where Fs >> 2xFc
Fc = 800 # carrier frequency
Fs = L*Fc # sampling frequency
BER = np.zeros(len(EbN0dB)) # for BER values for each Eb/N0
ak = np.random.randint(2, size = N) # uniform random symbols from 0's and 1's
(s_bb, t) = bpsk_mod(ak, L) # BPSK modulation(waveform) - baseband
s = s_bb*np.cos(2*np.pi*Fc*t/Fs) # with carrier # Waveforms at the transmitter
fig1, axs = plt.subplots(2, 2, figsize = (8, 6), constrained_layout = True)
axs[0, 0].plot(t, s_bb) # baseband wfm zoomed to first 10 bits
axs[0, 0].set_xlabel('t(s)')
axs[0, 1].set_ylabel(r'$s_{bb}(t)$-baseband')
axs[0, 1].plot(t, s) # transmitted wfm zoomed to first 10 bits
axs[0, 1].set_xlabel('t(s)')
axs[0, 1].set_ylabel('s(t)-with carrier')
axs[0, 0].set_xlim(0, 10*L)
axs[0, 1].set_xlim(0, 10*L) #signal constellation at transmitter
axs[1, 0].plot(np.real(s_bb), np.imag(s_bb), 'o')
axs[1, 0].set_xlim(-1.5, 1.5)
axs[1, 0].set_ylim(-1.5, 1.5)

for i,EbN0 in enumerate(EbN0dB):
    # Compute and add AWGN noise
    r = awgn(s, EbN0, L)             # refer Chapter section 4.1
    r_bb = r*np.cos(2*np.pi*Fc*t/Fs) # recovered baseband signal
    ak_hat = bpsk_demod(r_bb, L)     # baseband correlation demodulator
    BER[i] = np.sum(ak != ak_hat)/N  # Bit Error Rate Computation
    # Received signal waveform zoomed to first 10 bits
    axs[1, 1].plot(t, r) # received signal (with noise)
    axs[1, 1].set_xlabel('t(s)')
    axs[1, 1].set_ylabel('r(t)')
    axs[1, 1].set_xlim(0, 10*L)
    plt.show()
    plt.close()
    #------Theoretical Bit/Symbol Error Rates-------------
theoreticalBER = 0.5*erfc(np.sqrt(10**(EbN0dB/10))) # Theoretical bit error rate
#-------------Plots---------------------------
fig2, ax1 = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 6), constrained_layout = True)
ax1.semilogy(EbN0dB, BER, 'k*', label = 'Simulated') # simulated BER
ax1.semilogy(EbN0dB, theoreticalBER, 'r-', label = 'Theoretical')
ax1.set_xlabel(r'$E_b/N_0$ (dB)')
ax1.set_ylabel(r'Probability of Bit Error - $P_b$')
ax1.set_title(['Probability of Bit Error for BPSK modulation'])
ax1.legend()
plt.show()
plt.close()

#%% Program 20: DigiCommPy\chapter 2\debpsk coherent.py: Coherent detection of DEBPSK

#Execute in Python3: exec(open("chanter_2/debpsk_coherent.py").read())
import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from scipy.signal import lfilter
from scipy.special import erfc

from DigiCommPy.passband_modulations import bpsk_mod, bpsk_demod
from DigiCommPy.channels import awgn

N=1000000 # Number of symbols to transmit
EbN0dB = np.arange(start = -4, stop = 11, step = 2) # Eb/N0 range in dB for simulation
L = 16 # oversampling factor,L=Tb/Ts(Tb=bit period,Ts=sampling period)
# if a carrier is used, use L = Fs/Fc, where Fs >> 2 * Fc
Fc = 800 # carrier frequency
Fs = L*Fc # sampling frequency

SER = np.zeros(len(EbN0dB)) # for SER values for each Eb/N0
ak = np.random.randint(2, size = N) # uniform random symbols from 0's and 1's
bk = lfilter([1.0], [1.0, -1.0], ak) #IIR filter for differential encoding
bk = bk % 2 #XOR operation is equivalent to modulo-2

[s_bb, t] = bpsk_mod(bk, L) # BPSK modulation(waveform) - baseband
s = s_bb*np.cos(2*np.pi*Fc*t/Fs) # DEBPSK with carrier

for i,EbN0 in enumerate(EbN0dB):
    # Compute and add AWGN noise
    r = awgn(s, EbN0, L) # refer Chapter section 4.1

    phaseAmbiguity = np.pi # 180* phase ambiguity of Costas loop
    r_bb = r*np.cos(2*np.pi*Fc*t/Fs + phaseAmbiguity) # recovered signal
    b_hat = bpsk_demod(r_bb,L)                        # baseband correlation type demodulator
    a_hat = lfilter([1.0,1.0],[1.0],b_hat)            # FIR for differential decoding
    a_hat = a_hat % 2                                 # binary messages, therefore modulo-2
    SER[i] = np.sum(ak !=a_hat)/N                     # Symbol Error Rate Computation
#------Theoretical Bit/Symbol Error Rates-------------
EbN0lins = 10**(EbN0dB/10)                            # converting dB values to linear scale
theorySER_DPSK = erfc(np.sqrt(EbN0lins))*(1-0.5*erfc(np.sqrt(EbN0lins)))
theorySER_BPSK = 0.5*erfc(np.sqrt(EbN0lins))
#-------------Plots---------------------------
fig, ax = plt.subplots(nrows=1,ncols = 1)
ax.semilogy(EbN0dB,SER,'k*',label='Coherent DEBPSK(sim)')
ax.semilogy(EbN0dB,theorySER_DPSK,'r-',label='Coherent DEBPSK(theory)')
ax.semilogy(EbN0dB,theorySER_BPSK,'b-',label='Conventional BPSK')
ax.set_title('Probability of Bit Error for BPSK over AWGN');
ax.set_xlabel(r'$E_b/N_0$ (dB)');ax.set_ylabel(r'Probability of Bit Error - $P_b$');
ax.legend()
plt.show()
plt.close()

#%% Program 21: DigiCommPy\chapter 2\dbpsk noncoherent.py: DBPSK non-coherent detection
"""
Non-coherent detection of D-BPSK with phase ambiguity in local oscillator
"""
#Execute in Python3: exec(open("chapter_2/dbpsk_noncoherent.py").read())
import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from scipy.signal import lfilter
import scipy
from DigiCommPy.passband_modulations import bpsk_mod
from DigiCommPy.channels import awgn

N=100000 # Number of symbols to transmit
EbN0dB = np.arange(start=-4,stop = 11,step = 2) # Eb/N0 range in dB for simulation
L=8 # oversampling factor,L=Tb/Ts(Tb=bit period,Ts=sampling period)
# if a carrier is used, use L = Fs/Fc, where Fs >> 2xFc
Fc=800 # carrier frequency
Fs=L*Fc # sampling frequency

BER_suboptimum = np.zeros(len(EbN0dB)) # BER measures
BER_optimum = np.zeros(len(EbN0dB))

#-----------------Transmitter---------------------
ak = np.random.randint(2, size = N) # uniform random symbols from 0's and 1's
bk = lfilter([1.0], [1.0, -1.0], ak) # IIR filter for differential encoding
bk = bk%2 #XOR operation is equivalent to modulo-2
[s_bb,t]= bpsk_mod(bk, L) # BPSK modulation(waveform) - baseband
s = s_bb*np.cos(2*np.pi*Fc*t/Fs).astype(complex) # DBPSK with carrier

for i,EbN0 in enumerate(EbN0dB):
    # Compute and add AWGN noise
    r = awgn(s,EbN0,L) # refer Chapter section 4.1

    #----------suboptimum receiver---------------
    p=np.real(r)*np.cos(2*np.pi*Fc*t/Fs) # demodulate to baseband using BPF
    w0= np.hstack((p, np.zeros(L))) # append L samples on one arm for equal lengths
    w1= np.hstack((np.zeros(L), p)) # delay the other arm by Tb (L samples)
    w = w0*w1 # multiplier
    z = np.convolve(w, np.ones(L)) #integrator from kTb to (K+1)Tb (L samples)
    u =  z[L-1:-1-L:L] # sampler t=kTb
    ak_hat = (u<0) #decision
    BER_suboptimum[i] = np.sum(ak != ak_hat)/N #BER for suboptimum receiver

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
theory_DBPSK_coherent = scipy.special.erfc(np.sqrt(EbN0lins))*(1-0.5*scipy.special.erfc(np.sqrt(EbN0lins)))
theory_BPSK_conventional = 0.5*scipy.special.erfc(np.sqrt(EbN0lins))

#-------------Plotting---------------------------
fig, ax = plt.subplots(nrows=1,ncols = 1)
ax.semilogy(EbN0dB,BER_suboptimum,'k*',label='DBPSK subopt (sim)')
ax.semilogy(EbN0dB,BER_optimum,'b*',label='DBPSK opt (sim)')
ax.semilogy(EbN0dB,theory_DBPSK_suboptimum,'m-',label='DBPSK subopt (theory)')
ax.semilogy(EbN0dB,theory_DBPSK_optimum,'r-',label='DBPSK opt (theory)')
ax.semilogy(EbN0dB,theory_DBPSK_coherent,'k-',label='coherent DEBPSK')
ax.semilogy(EbN0dB,theory_BPSK_conventional,'b-',label='coherent BPSK')
ax.set_title('Probability of D-BPSK over AWGN')
ax.set_xlabel('$E_b/N_0 (dB)$')
ax.set_ylabel('$Probability of Bit Error - P_b$')
ax.legend()
plt.show()
plt.close()


#%% Program 24: DigiCommPy\chapter 2\qpsk.py: Waveform simulation of performance of QPSK
import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from DigiCommPy.passband_modulations import qpsk_mod,qpsk_demod
from DigiCommPy.channels import awgn
from scipy.special import erfc

N = 100000 # Number of symbols to transmit
EbN0dB = np.arange(start = -4, stop = 11, step = 2) # Eb/N0 range in dB for simulation
fc = 100 # carrier frequency in Hertz
OF = 8 # oversampling factor, sampling frequency will be fs=OF*fc

BER = np.zeros(len(EbN0dB)) # For BER values for each Eb/N0

a = np.random.randint(2, size = N) # uniform random symbols from 0's and 1's
result = qpsk_mod(a, fc, OF, enable_plot = 1) #QPSK modulation
s = result['s(t)'] # get values from returned dictionary

for i, EbN0 in enumerate(EbN0dB):
    # Compute and add AWGN noise
    r = awgn(s, EbN0, OF)         # refer Chapter section 4.1
    a_hat = qpsk_demod(r, fc, OF) # QPSK demodulation
    BER[i] = np.sum(a != a_hat)/N # Bit Error Rate Computation

#------Theoretical Bit Error Rate-------------
theoreticalBER = 0.5*scipy.special.erfc(np.sqrt(10**(EbN0dB/10)))
#-------------Plot performance curve------------------------
fig, axs = plt.subplots(nrows=1,ncols = 1)
axs.semilogy(EbN0dB,BER,'k*',label='Simulated')
axs.semilogy(EbN0dB,theoreticalBER,'r-',label='Theoretical')
axs.set_title('Probability of Bit Error for QPSK modulation');
axs.set_xlabel(r'$E_b/N_0$ (dB)')
axs.set_ylabel(r'Probability of Bit Error - $P_b$');
axs.legend()
plt.show()
plt.close()


#%% Program 27: DigiCommPy\chapter 2\oqpsk.py: Waveform simulation of performance of OQPSK

import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from DigiCommPy.passband_modulations import oqpsk_mod,oqpsk_demod
from DigiCommPy.channels import awgn
from scipy.special import erfc

N = 100000 # Number of symbols to transmit
EbN0dB = np.arange(start = -4,stop = 11,step = 2) # Eb/N0 range in dB for simulation
fc = 100 # carrier frequency in Hertz
OF = 8 # oversampling factor, sampling frequency will be fs=OF*fc
BER = np.zeros(len(EbN0dB)) # For BER values for each Eb/N0

a = np.random.randint(2, size = N) # uniform random symbols from 0's and 1's
result = oqpsk_mod(a, fc, OF, enable_plot = False) # QPSK modulation
s = result['s(t)'] # get values from returned dictionary
for i,EbN0 in enumerate(EbN0dB):
    # Compute and add AWGN noise
    r = awgn(s, EbN0, OF) # refer Chapter section 4.1
    a_hat = oqpsk_demod(r, N, fc, OF, enable_plot = False) # QPSK demodulation
    BER[i] = np.sum(a != a_hat)/N # Bit Error Rate Computation
#------Theoretical Bit Error Rate-------------
theoreticalBER = 0.5*erfc(np.sqrt(10**(EbN0dB/10)))
#-------------Plot performance curve------------------------
fig, axs = plt.subplots(nrows=1,ncols = 1)
axs.semilogy(EbN0dB,BER,'k*',label='Simulated')
axs.semilogy(EbN0dB,theoreticalBER,'r-',label='Theoretical')
axs.set_title('Probability of Bit Error for OQPSK');
axs.set_xlabel(r'$E_b/N_0$ (dB)')
axs.set_ylabel(r'Probability of Bit Error - $P_b$');
axs.legend();
plt.show()
plt.close()

#%% Program 32: DigiCommPy\chapter 2\piby4 dqpsk.py: π/4 − DQPSK performance simulation
#Execute in Python3: exec(open("chapter_2/piby4_dqpsk.py").read())
import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from DigiCommPy.passband_modulations import piBy4_dqpsk_mod,piBy4_dqpsk_demod
from DigiCommPy.channels import awgn
from scipy.special import erfc

N = 1000000 # Number of symbols to transmit
EbN0dB = np.arange(start=-4,stop = 11,step = 2) # Eb/N0 range in dB for simulation
fc = 100 # carrier frequency in Hertz
OF = 8 # oversampling factor, sampling frequency will be fs=OF*fc

BER = np.zeros(len(EbN0dB)) # For BER values for each Eb/N0

a = np.random.randint(2, size=N) # uniform random symbols from 0's and 1's
result = piBy4_dqpsk_mod(a,fc,OF,enable_plot=1)# dqpsk modulation
s = result['s(t)'] # get values from returned dictionary

for i,EbN0 in enumerate(EbN0dB):
    # Compute and add AWGN noise
    r = awgn(s,EbN0,OF) # refer Chapter section 4.1
    a_hat = piBy4_dqpsk_demod(r,fc,OF,enable_plot=False)
    BER[i] = np.sum(a!=a_hat)/N # Bit Error Rate Computation

#------Theoretical Bit Error Rate-------------
x = np.sqrt(4*10**(EbN0dB/10))*np.sin(np.pi/(4*np.sqrt(2)))
theoreticalBER = 0.5*erfc(x/np.sqrt(2))

#-------------Plot performance curve------------------------
fig, axs = plt.subplots(nrows=1,ncols = 1)
axs.semilogy(EbN0dB,BER,'k*',label='Simulated')
axs.semilogy(EbN0dB,theoreticalBER,'r-',label='Theoretical')
axs.set_title(r'Probability of Bit Error for $\pi/4$-DQPSK');
axs.set_xlabel(r'$E_b/N_0$ (dB)')
axs.set_ylabel(r'Probability of Bit Error - $P_b$');
axs.legend();
plt.show()
plt.close()


#%% Program 33: DigiCommPy\chapter 2\cpfsk.py: Binary CPFSK modulation

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

L = 50 # oversampling factor
Tb = 0.5 # bit period in seconds
fs = L/Tb # sampling frequency in Hertz
fc = 2/Tb # carrier frequency
N = 8 # number of bits to transmit
h = 1 # modulation index

np.random.seed(42)
b = 2*np.random.randint(2, size=N)-1 # random information sequence in +1/-1 format
b = np.tile(b, (L, 1)).flatten('F')
# b_integrated = lfilter([1.0], [1.0, -1.0], b)/fs #Integrate b using filter
b_integrated = np.cumsum(b)/fs


theta= np.pi*h/Tb*b_integrated
t=np.arange(0, Tb*N, 1/fs) # time base

s = np.cos(2*np.pi*fc*t + theta) # CPFSK signal

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(t, b)
ax1.set_xlabel('t')
ax1.set_ylabel('b(t)')
ax2.plot(t, theta)
ax2.set_xlabel('t')
ax2.set_ylabel('$\theta(t)$')
ax3.plot(t, s)
ax3.set_xlabel('t')
ax3.set_ylabel('s(t)')
plt.show()
plt.close()

#%% Program 36: DigiCommPy\chapter 2\msk.py: Performance of MSK over AWGN

import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from scipy.special import erfc
from DigiCommPy.passband_modulations import msk_mod,msk_demod
from DigiCommPy.channels import awgn

N = 100000 # Number of symbols to transmit
EbN0dB = np.arange(start=-4,stop = 11,step = 2) # Eb/N0 range in dB for simulation
fc = 800 # carrier frequency in Hertz
OF = 32 # oversampling factor, sampling frequency will be fs=OF*fc

BER = np.zeros(len(EbN0dB)) # For BER values for each Eb/N0

a = np.random.randint(2, size = N) # uniform random symbols from 0's and 1's
result = msk_mod(a, fc, OF, enable_plot=1) # MSK modulation
s = result['s(t)']

for i,EbN0 in enumerate(EbN0dB):
    # Compute and add AWGN noise
    r = awgn(s, EbN0, OF) # refer Chapter section 4.1
    a_hat = msk_demod(r, N, fc, OF) #receiver
    BER[i] = np.sum(a!=a_hat)/N # Bit Error Rate Computation
theoreticalBER = 0.5*erfc(np.sqrt(10**(EbN0dB/10))) # Theoretical bit error rate

#-------------Plots---------------------------
fig, ax = plt.subplots(nrows=1,ncols = 1)
ax.semilogy(EbN0dB,BER,'k*',label='Simulated') # simulated BER
ax.semilogy(EbN0dB,theoreticalBER,'r-',label='Theoretical')
ax.set_xlabel(r'$E_b/N_0$ (dB)')
ax.set_ylabel(r'Probability of Bit Error - $P_b$')
ax.set_title(['Probability of Bit Error for MSK modulation'])
ax.legend();
plt.show()
plt.close()

#%% Program 39: DigiCommPy\chapter 2\constellations.py:Constellations of RC filtered QPSK & MSK

import numpy as np
import matplotlib.pyplot as plt
from DigiCommPy.passband_modulations import qpsk_mod, oqpsk_mod, piBy4_dqpsk_mod, msk_mod
from DigiCommPy.pulseshapers import raisedCosineDesign

N = 1000 # Number of symbols to transmit, keep it small and adequate
fc = 10
L = 8 # carrier frequency and oversampling factor
a = np.random.randint(2, size=N) # uniform random symbols from 0's and 1's

#modulate the source symbols using QPSK,QPSK,pi/4-DQPSK and MSK
qpsk_result = qpsk_mod(a, fc, L)
oqpsk_result = oqpsk_mod(a, fc, L)
piby4qpsk_result = piBy4_dqpsk_mod(a, fc, L)
msk_result = msk_mod(a, fc, L);

#Pulse shape the modulated waveforms by convolving with RC filter
alpha = 0.3
span = 10 # RC filter alpha and filter span in symbols
b = raisedCosineDesign(alpha,span, L) # RC pulse shaper
iRC_qpsk = np.convolve(qpsk_result['I(t)'],b, mode= 'valid') # RC shaped QPSK I channel
qRC_qpsk = np.convolve(qpsk_result['Q(t)'],b, mode= 'valid') # RC shaped QPSK Q channel
iRC_oqpsk = np.convolve(oqpsk_result['I(t)'],b, mode= 'valid') #RC shaped OQPSK I channel
qRC_oqpsk = np.convolve(oqpsk_result['Q(t)'],b, mode= 'valid') #RC shaped OQPSK Q channel
iRC_piby4qpsk = np.convolve(piby4qpsk_result['U(t)'],b, mode= 'valid') #RC shaped pi/4-QPSK I channel
qRC_piby4qpsk = np.convolve(piby4qpsk_result['V(t)'],b, mode= 'valid') #RC shaped pi/4-QPSK Q channel
i_msk = msk_result['sI(t)'] # MSK sI(t)
q_msk = msk_result['sQ(t)'] # MSK sQ(t)

fig, axs = plt.subplots(2, 2)

axs[0,0].plot(iRC_qpsk, qRC_qpsk)# RC shaped QPSK
axs[0,1].plot(iRC_oqpsk, qRC_oqpsk)# RC shaped OQPSK
axs[1,0].plot(iRC_piby4qpsk, qRC_piby4qpsk)# RC shaped pi/4-QPSK
axs[1,1].plot(i_msk[20:-20], q_msk[20:-20])# RC shaped OQPSK

axs[0,0].set_title(r'QPSK, RC $\alpha$='+str(alpha))
axs[0,0].set_xlabel('I(t)');
axs[0,0].set_ylabel('Q(t)');
axs[0,1].set_title(r'OQPSK, RC $\alpha$='+str(alpha))
axs[0,1].set_xlabel('I(t)');
axs[0,1].set_ylabel('Q(t)');
axs[1,0].set_title(r'$\pi$/4 - QPSK, RC $\alpha$='+str(alpha))
axs[1,0].set_xlabel('I(t)');
axs[1,0].set_ylabel('Q(t)');
axs[1,1].set_title('MSK')
axs[1,1].set_xlabel('I(t)')
axs[1,1].set_ylabel('Q(t)')
plt.show()
plt.close()



#%% Program 40: DigiCommPy\chapter 2\psd estimates.py: PSD estimates of BPSK QPSK and MSK

import numpy as np
import matplotlib.pyplot as plt

def bpsk_qpsk_msk_psd():
    # Usage: >> from chapter_2.psd_estimates import bpsk_qpsk_msk_psd
    #        >> bpsk_qpsk_msk_psd()
    from DigiCommPy.passband_modulations import bpsk_mod, qpsk_mod, msk_mod
    from DigiCommPy.essentials import plotWelchPSD
    N = 100000   # Number of symbols to transmit
    fc = 800
    OF = 8       # carrier frequency and oversamping factor
    fs = fc*OF   # sampling frequency

    a = np.random.randint(2, size = N) # uniform random symbols from 0's and 1's
    (s_bb,t) = bpsk_mod(a, OF) # BPSK modulation(waveform) - baseband
    s_bpsk = s_bb*np.cos(2*np.pi*fc*t/fs) # BPSK with carrier
    s_qpsk = qpsk_mod(a, fc, OF)['s(t)'] # conventional QPSK
    s_msk = msk_mod(a, fc, OF)['s(t)'] # MSK signal

    # Compute and plot PSDs for each of the modulated versions
    fig, ax = plt.subplots(1, 1)
    plotWelchPSD(s_bpsk, fs, fc, ax = ax, color = 'b', label = 'BPSK')
    plotWelchPSD(s_qpsk, fs, fc, ax = ax, color = 'r', label = 'QPSK')
    plotWelchPSD(s_msk, fs, fc, ax = ax, color = 'k', label = 'MSK')
    ax.set_xlabel('$f-f_c$')
    ax.set_ylabel('PSD (dB/Hz)')
    ax.legend()
    plt.show()
    plt.close()
    return


def gmsk_psd():
    from DigiCommPy.passband_modulations import gmsk_mod
    from DigiCommPy.essentials import plotWelchPSD

    N = 10000 # Number of symbols to transmit
    fc = 800 # carrier frequency in Hertz
    L = 16 # oversampling factor,use L= Fs/Fc, where Fs >> 2xFc
    fs = L*fc
    a = np.random.randint(2, size=N) # uniform random symbols from 0's and 1's

    #'_':unused output variable
    (s1 , _ ) = gmsk_mod(a, fc, L, BT = 0.3, enable_plot = True) # BT_b=0.3
    (s2 , _ ) = gmsk_mod(a, fc, L, BT = 0.5) # BT_b=0.5
    (s3 , _ ) = gmsk_mod(a, fc, L, BT = 0.7) # BT_b=0.7
    (s4 , _ ) = gmsk_mod(a, fc, L, BT = 10000) # BT_b=very value value (MSK)

    # Compute and plot PSDs for each of the modulated versions
    fig, ax = plt.subplots(1, 1)
    plotWelchPSD(s1,fs,fc, ax = ax , color = 'r', label = '$BT_b=0.3$')
    plotWelchPSD(s2,fs,fc, ax = ax , color = 'b', label = '$BT_b=0.5$')
    plotWelchPSD(s3,fs,fc, ax = ax , color = 'm', label = '$BT_b=0.7$')
    plotWelchPSD(s4,fs,fc, ax = ax , color = 'k', label = '$BT_b=\infty$')
    ax.set_xlabel('$f-f_c$'); ax.set_ylabel('PSD (dB/Hz)')
    ax.legend()
    plt.show()
    plt.close()
    return

bpsk_qpsk_msk_psd()
gmsk_psd()


#%% Program 45: DigiCommPy\chapter 2\gmsk.py: Performance simulation of baseband GMSK

import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from DigiCommPy.passband_modulations import gmsk_mod,gmsk_demod
from DigiCommPy.channels import awgn

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
axs.legend();
plt.show()
plt.close()


#%% Program 49: DigiCommPy\chapter 2\bfsk.py: Performance of coherent and non-coherent BFSK

#Execute in Python3: exec(open("chapter_2/bfsk.py").read())
import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from DigiCommPy.passband_modulations import bfsk_mod, bfsk_coherent_demod, bfsk_noncoherent_demod
from DigiCommPy.channels import awgn
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
axs.legend()
plt.show()
plt.close()




















































































































































































