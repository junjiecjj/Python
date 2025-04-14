#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 14:18:55 2025

@author: jack
"""

import scipy
import numpy as np
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import commpy

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14        # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22   # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22   # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300      # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'        # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'          # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22

#%% Program 13.1: sequence correlation.m: Computing sequence correlation between two sequences
def sequence_correlation(x, y, k1, k2):
    x = x[:]
    y = y[:]
    if x.size != y.size:
        print("Sequences x and y should be of same length")
    L = x.size
    rangeOFKs = np.arange(k1, k2+1)
    Rxy = np.zeros(rangeOFKs.size)

    if k1 != 0:
        start = (L+k1) % L
        final = (L+k1-1) % L
        x = np.hstack(( x[start:], x[:final+1] ))
    q = int(np.floor(x.size/y.size))
    r = int(x.size % y.size)
    y = np.hstack((np.tile(y,(1, q)).flatten() , y[:r] ))

    for i in range(rangeOFKs.size):
        aggre = np.sum(x == y)
        disaggre = np.sum(x!=y)
        x = np.hstack((x[1:], x[0] ))
        Rxy[i] = aggre - disaggre

    return Rxy

#%% Program 13.2: lfsr.m: Implementation of matrix form of LFSR for m-sequence generation
def lfsr(G, X):
    g = G[:]
    x = X[:]
    if g.size - x.size != 1:
        print('Length of initial seed X0 should be equal to the number of delay elements (length(g)-1)')
    # LFSR state-transistion matrix construction
    L = g.size - 1                                      # order of polynomial
    A0 = np.vstack((np.zeros((1, L-1)), np.eye(L-1)))   # A-matrix construction
    g = g[:-1]
    A = np.hstack((A0, g.reshape(-1,1)))                # LFSR state-transistion matrix

    N = 2**L-1                                          # period of maximal length sequence
    y = np.zeros(N)                                     # array to store output
    states = np.zeros(N)                                # LFSR states(useful for Frequeny Hopping)
    for i in range(N):                      # repeate for each clock period
        states[i] = commpy.utilities.bitarray2dec(x)  # convert LFSR states to a number
        y[i] = x[-1]                       # output the last bit
        x =  A@x % 2                      # LFSR equation

    return y, states

G = np.array([1, 0, 0, 1, 0, 1])
X = np.array([0, 0, 0, 0, 1])
y, states = lfsr(G, X)
N = 31
Ryy = 1/N * sequence_correlation(y, y, -35, 35)

fig, axs = plt.subplots(1, 1, figsize = (12, 4), constrained_layout = True)
axs.plot(np.arange(-35, 36), Ryy, color = 'r', ls = '-', lw = 2, )
plt.show()
plt.close()

x, _ = lfsr(np.array([1, 1, 0, 1, 1, 1,]) , np.array([0, 0, 0, 0, 1]) )
y, _ = lfsr(np.array([1, 1, 1, 0, 1, 1,]) , np.array([0, 0, 0, 0, 1]) )
N = 31
Rxy = 1 / N * sequence_correlation(x, y, 0, 31)
fig, axs = plt.subplots(1, 1, figsize = (12, 4), constrained_layout = True)
axs.plot(np.arange(N+1), Rxy, color = 'r', ls = '-', lw = 2,  )
plt.show()
plt.close()

#%% Program 13.5: gold code generator.m: Software implementation of Gold code generator
def gold_code_generator(G1, G2, X1, X2):
    # Implementation of Gold code generator
    # G1-preferred polynomial 1 for LFSR1 arranged as [g0 g1 g2 ... gL-1]
    # G2-preferred polynomial 2 for LFSR2 arranged as [g0 g1 g2 ... gL-1]
    # X1-initial seed for LFSR1 [x0 x1 x2 ... xL-1]
    # X2-initial seed for LFSR2 [x0 x1 x2 ... xL-1]
    # y-Gold code
    # The function outputs the m-sequence for a single period
    # Sample call:
    # 7th order preferred polynomials [1,2,3,7] and [3,7] (polynomials : 1+x+x^2+x^3+x^7 and 1+x^3+x^7)
    # i.e, G1 = [1,1,1,1,0,0,0,1] and G2=[1,0,0,1,0,0,0,1]
    # with intial states X1=[0,0,0,0,0,0,0,1], X2=[0,0,0,0,0,0,0,1]:
    # gold_code_generator(G1,G2,X1,X2)
    g1 = G1[:]
    x1 = X1[:]     # serialize G1 and X1 matrices
    g2 = G2[:]
    x2 = X2[:]     # serialize G2 and X2 matrices
    if g1.size != g2.size and x1.size != x2.size:
        print('Length mismatch between G1 & G2 or X1 & X2')

    # LFSR state-transistion matrix construction
    L = g1.size - 1               # order of polynomial
    A0 = np.vstack((np.zeros((1, L-1)), np.eye(L-1)))  # A-matrix construction
    g1 = g1[:-1]
    g2 = g2[:-1]
    A1 = np.hstack((A0, g1.reshape(-1, 1)))  # LFSR1 state-transistion matrix
    A2 = np.hstack((A0, g2.reshape(-1, 1)))  # LFSR2 state-transistion matrix

    N = 2**L-1                               # period of maximal length sequence
    y = np.zeros(N)                          # array to store output
    for i in range(N):                        # repeate for each clock period
        y[i] =  (x1[-1] + x2[-1]) % 2     # XOR of outputs of LFSR1 & LFSR2
        x1 = (A1@x1) % 2                 # LFSR equation
        x2 = (A2@x2) % 2                 # LFSR equation
    return y

# Gold codes auto-correlation
G1 = np.array([1, 1, 1, 1, 0, 1])
G2 = np.array([1, 0, 0, 1, 0, 1])     # feedback connections
X1 = np.array([0, 0, 0, 0, 1])
X2 = np.array([0, 0, 0, 0, 1])        # initial states of LFSRs
y1, _ = lfsr(G1,X1)
y2, _ = lfsr(G2, X2)
N = 31    # m-sequence 1 and 2
Ry1y2 = 1/N*sequence_correlation(y1, y2, 0, 31)   # cross-correlation
# plot(0:1:31,Ry1y2)%plot correlation
fig, axs = plt.subplots(1, 1, figsize = (12, 4), constrained_layout = True)
axs.plot(np.arange(N+1), Ry1y2, color = 'r', ls = '-', lw = 2, )
plt.show()
plt.close()

# Gold codes cross-correlation
N = 31                              # period of Gold code
G1 = np.array([1, 1, 1, 1, 0, 1])
G2 = np.array([1, 0, 0, 1, 0, 1])   # feedback connections
X1 = np.array([0, 0, 0, 0, 1])
X2 = np.array([0, 0, 0, 0, 1])      # initial states of LFSRs
y = gold_code_generator(G1, G2, X1, X2)       # Generate Gold code
Ryy = 1/N*sequence_correlation(y, y, 0, 31)   # auto-correlation

fig, axs = plt.subplots(1, 1, figsize = (12, 4), constrained_layout = True)
axs.plot(np.arange(N+1), Ryy, color = 'r', ls = '-', lw = 2, )
plt.show()
plt.close()

#%% Program 13.8: generatePRBS.m: Generating PRBS sequence - from msequence or gold code
def generatePRBS(prbsType, G1, G2, X1, X2):
    # Generate PRBS sequence - choose from either msequence or gold code
    # prbsType - type of PRBS generator - 'MSEQUENCE' or 'GOLD'
    #   If prbsType == 'MSEQUENCE' G1 is the generator poly for LFSR
    #    and X1 its seed. G2 and X2 are not used
    #   If prbsType == 'GOLD' G1,G2 are the generator polynomials
    #    for LFSR1/LFSR2 and X1,X2 are their initial seeds.
    # G1,G2 - Generator polynomials for PRBS generation
    # X1,X2 - Initial states of LFSRs

    # The PRBS generators results in 1 period PRBS, need to repeat it to suit the data length
    if prbsType == 'MSEQUENCE':
       prbs, _ = lfsr(G1, X1)  # use only one poly and initial state vector
    elif prbsType == 'GOLD':
       prbs = gold_code_generator(G1, G2, X1, X2)  # full length Gold sequence
    else:  # Gold codes as default
       G1 = np.array([1, 1, 1, 1, 0, 1])
       G2 = np.array([1, 0, 0, 1, 0, 1])
       X1 = np.array([ 0, 0, 0, 0, 1])
       X2 = np.array([ 0, 0, 0, 0, 1])
       prbs = gold_code_generator(G1, G2, X1, X2)
    return prbs

def repeatSequence(x, N):
    # Repeat a given sequence x of arbitrary length to match the given
    # length N. This function is useful to repeat any sequence
    # say PRBS) to match the length of another sequence
    x = x[:] # serialize
    xLen = x.size  # length of sequence x truncate or extend sequence x to suite the given length N
    if xLen >= N: # Truncate x when sequencelength less than N
       y = x[:N]
    else:
       temp = np.tile(x, (1, int(N//xLen))).flatten()  # repeat sequence integer times
       residue =  int(N % xLen)             # reminder when dividing N by xLen
       # append reminder times
       if residue != 0:
           temp = np.hstack((temp, x[:residue]))  # [temp; x(1:residue)]
       y = temp                                   # repeating sequence matching length N
    return y.flatten()
repeatSequence(np.array([1, 1, 1, 1, 0, 1]), 45)

#%% Program 13.10: dsss transmitter.m: Function implementing a DSSS transmitter
def dsss_transmitter(d, prbsType, G1, G2, X1, X2, Rb, Rc, L):
    # Direct Sequence Spread Spectrum (DSSS) transmitter - returns the DSSS  waveform (s), reference carrier, the prbs reference waveform for use in synchronization in the receiver
    # d - input binary data stream
    # prbsType - type of PRBS generator - 'MSEQUENCE' or 'GOLD'
    #   If prbsType == 'MSEQUENCE' G1 is the generator poly for LFSR
    #       and X1 its seed. G2 and X2 are not used
    #   If prbsType == 'GOLD' G1,G2 are the generator polynomials
    #       for LFSR1/LFSR2 and X1,X2 are their initial seeds.
    # G1,G2 - Generator polynomials for PRBS generation
    # X1,X2 - Initial states of LFSRs
    # Rb - data rate (bps) for the data d
    # Rc - chip-rate (Rc >> Rb AND Rc is integral multiple of Rb)
    # L - oversampling factor for waveform generation

    prbs = generatePRBS(prbsType, G1, G2, X1, X2).astype(np.int64)
    prbs = prbs[:]
    d = d[:]                                     # serialize
    dataLen = int(d.size * (Rc/Rb) )             # required PRBS length to cover the data
    prbs_ref = repeatSequence(prbs, dataLen)     # repeat PRBS to match data

    d_t = np.kron(d, np.ones((int(L*Rc/Rb)))).flatten().astype(np.int64) # data waveform
    prbs_t = np.kron(prbs_ref, np.ones((L))).flatten().astype(np.int64)   # spreading sequence waveform
    sbb_t = 2 * (d_t ^ prbs_t) - 1           # XOR data and PRBS, convert to bipolar
    n = np.arange(sbb_t.size)
    carrier_ref = np.cos(2 * np.pi * 2 * n/L)
    s_t = sbb_t * carrier_ref                 # modulation,2 cycles per chip

    # fig, axs = plt.subplots(3, 1, figsize = (12, 12), constrained_layout = True)
    # axs[0].plot(d_t, color = 'b', ls = '-', lw = 2, label = '')
    # axs[0].set_title("data sequence")
    # axs[1].plot(prbs_t, color = 'r', ls = '-', lw = 2, label = '')
    # axs[1].set_title("PRBS sequence")
    # axs[2].plot(s_t, color = 'r', ls = '-', lw = 2, label = '')
    # axs[2].set_title("DS-SS signal (baseband)")
    # plt.show()
    # plt.close()
    return s_t, carrier_ref, prbs_ref

Rb = 100
Rc = 1000
L = 32  # data rate, chip rate and oversampling factor
prbsType= 'GOLD' # PRBS type is set to Gold code
G1 = np.array([1, 1, 1, 1, 0, 0, 0, 1])
G2 = np.array([1, 0, 0, 1, 0, 0, 0, 1])  # LFSR polynomials
X1 = np.array([0, 0, 0, 0, 0, 0, 1])
X2 = np.array([0, 0, 0, 0, 0, 0, 1])    # initial state of LFSRs
d = np.random.randint(0, 2, size = (2,)) #  10 bits of random data
s_t, carrier_ref, prbs_ref = dsss_transmitter(d, prbsType, G1, G2, X1, X2, Rb, Rc, L);

#%% Program 13.12: dsss receiver.m: Function implementing a DSSS receiver
def dsss_receiver(r_t, carrier_ref, prbs_ref, Rb, Rc, L):
    # Direct Sequence Spread Spectrum (DSSS) Receiver (Rx)
    # r_t         - received DS-SS signal from the transmitter (Tx)
    # carrier_ref - reference carrier (synchronized with transmitter)
    # prbs_ref    - reference PRBS signal(synchronized with transmitter)
    # Rb - data rate (bps) for the data d
    # Rc - chip-rate ((Rc >> Rb AND Rc is integral multiple of Rb)
    # L - versampling factor used for waveform generation at the Tx
    # The function first demodulates the receiver signal using the
    # reference carrier and then XORs the result with the reference
    # PRBS. Finally returns the recovered data.
    # BPSK Demodulation----------
    v_t = r_t*carrier_ref
    x_t = np.convolve(v_t, np.ones(L), mode = 'full') # integrate for Tc duration
    y = x_t[L-1:x_t.size+1:L]   # sample at every Lth sample (i.e, every Tc instant)
    z =  y > 0     # Hard decision (gives demodulated bits)
    # -----------De-Spreading----------------
    y = z ^ prbs_ref       # reverse the spreading process using PRBS ref.
    d_cap = y[int(Rc/Rb)-1 : y.size+1: int(Rc/Rb)]  #  sample at every Rc/Rb th symbol

    return d_cap

#%% Program 13.13: dsss tx rx chain.m : Performance of DSSS system over AWGN channel
# from DigiCommPy.errorRates import ser_awgn
prbsType = 'MSEQUENCE'                # PRBS type
G1 = np.array([1, 0, 0, 1, 0, 1])     # LFSR polynomial
X1 = np.array([0, 0, 0, 0, 1])        # initial seed for LFSR
G2 = 0
X2 = 0                          #  G2, X2 are zero for m-sequence (only one LFSR used)
#  Input data, data rate, chip rate------
N = int(10e5)                        # number of data bits to transmit
d = np.random.randint(0, 2, N)       # random data
Rb = int(2e3)                        # data rate (bps) for the data d
Rc = int(6e3)                        # chip-rate(Rc >> Rb AND Rc is integral multiple of Rb)
L = 8                                # oversampling factor for waveform generation
SNR_dB = np.arange(-4, 17, 4)        # signal to noise ratios (dB)
BER = np.zeros(SNR_dB.size)          # place holder for BER values
for i in range(SNR_dB.size):
    s_t, carrier_ref, prbs_ref = dsss_transmitter(d, prbsType, G1, G2, X1, X2, Rb, Rc, L)  # Tx
    # -----Compute and add AWGN noise to the transmitted signal---
    Esym = L*np.sum(np.abs(s_t)**2)/(s_t.size)         # Calculate symbol energy
    N0 = Esym/(10**(SNR_dB[i]/10));                    # Find the noise spectral density
    n_t = np.sqrt(N0/2)*np.random.randn(*s_t.shape);   # computed noise
    r_t = s_t + n_t;                                   # received signal

    dCap = dsss_receiver(r_t, carrier_ref,prbs_ref,Rb,Rc,L);       # Receiver
    BER[i] = np.sum(dCap!=d)/d.size                                # Bit Error Rate
SER_theory = 0.5 * scipy.special.erfc(np.sqrt(10**(SNR_dB/10)))

colors = plt.cm.jet(np.linspace(0, 1, 4)) # colormap
fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.semilogy(SNR_dB, BER, color = colors[0], marker = 'o', linestyle = '--', lw = 1, label = 'Sim' )
ax.semilogy(SNR_dB, SER_theory, color = colors[1], linestyle = '-',lw = 2, label = 'Theory' )
ax.set_ylim(1e-6, 1)
ax.set_xlabel('Eb/N0(dB)')
ax.set_ylabel('SER ($P_s$)')
ax.set_title('Probability of Symbol Error for BPSK over AWGN')
ax.legend()
plt.show()
plt.close()

#%% Program 13.14: tone jammer.m: Single tone jammer
def tone_jammer(JSR_dB, Fj, Thetaj, Esig, L):
    # Generates a single tone jammer (J) for the following inputs
    # JSR_dB - required Jammer to Signal Ratio for generating the jammer
    # Fj- Jammer frequency offset from center frequency (-0.5 < Fj < 0.5)
    # Thetaj - phase of the jammer tone (0 to 2*pi)
    # Esig -transmitted signal power to which jammer needs to be added
    # L    - length of the transmitter signal vector
    # The output JSRmeas is the measured JSR from the generated samples

    JSR = 10**(JSR_dB/10)    # Jammer-to-Signal ratio in linear scale
    Pj= JSR*Esig             # required Jammer power for the given signal power
    n = np.arange(L)       #  indices for generating jammer samples
    J = np.sqrt(2*Pj) * np.sin(2*np.pi*Fj*n + Thetaj); # Single Tone Jammer

    Ej = np.sum(np.abs(J)**22)/L; # computed jammer energy from generated samples
    JSRmeas = 10 * np.log10(Ej/Esig)     # measured JSR
    return J, JSRmeas

from tqdm import tqdm
Gp = 31                                    # processing gain
N = int(10e5)                              # number of data bits to transmit
EbN0dBs = np.arange(0, 16, 2)              # Eb/N0 ratios (dB)
JSR_dBs = np.array([-100, -10, -5, 0, 2])  # Jammer to Signal ratios (dB)
Fj = 0.0001                                # Jamming tone - normalized frequency (-0.5 < F < 0.5)
Thetaj = 2 * np.pi * np.random.rand()      # random jammer phase(0 to 2*pi radians)

# ----PRBS definition (refer previous sections of this chapter)---
prbsType = 'MSEQUENCE'         # PRBS type,period of the PRBS should match Gp
G1 = np.array([1, 0, 0, 1, 1, 1])     # LFSR polynomial
X1 = np.array([0, 0, 0, 0, 1])        # initial seed for LFSR
G2 = 0
X2 = 0
# ---------------Transmitter------------------
d = np.random.randint(0, 2, N)  # Random binary information source
dp = 2*d-1    # converted to polar format (+/- 1)
dr = np.kron(dp, np.ones(Gp))   # repeat d[n] - Gp times
L = N*Gp      # length of the data after the repetition

prbs = generatePRBS(prbsType,G1,G2,X1,X2);  #  period of PRBS
prbs = 2*prbs-1;                            # convert PRBS to +/- 1 values
# prbs = prbs(:);                              # serialize
c = repeatSequence(prbs, L)                    # repeat PRBS sequence to match data length
s = dr*c                                       # multiply information source and PRBS sequence
# Calculate signal power (used to generate jamming signal)
Eb = Gp * np.sum(np.abs(s)**2)/L

plotColors = ['r','b','k','g','m']
colors = plt.cm.jet(np.linspace(0, 1, len(JSR_dBs) + 1)) # colormap
fig, ax = plt.subplots(nrows = 1, ncols = 1)
# for k in range(JSR_dB.size):                      # loop for each given JSR
for k, JSR_dB in tqdm(enumerate(JSR_dBs)):
    # Generate single tone jammer for each given JSR
    J, _ = tone_jammer(JSR_dB, Fj, Thetaj, Eb, L)   # generate a tone jammer
    BER = np.zeros(EbN0dBs.size)                    # place holder for BER values
    for i, EbN0dB in enumerate(EbN0dBs):            # loop for each given SNR
        # -----------AWGN noise addition---------
        N0=Eb/(10**(EbN0dB/10))                           # Find the noise spectral density
        w = np.sqrt(N0/2)*np.random.randn(*s.shape)       # computed whitenoise
        r = s + J + w                                     # received signal
        # ------------Receiver--------------
        yr = r*c;                                        # multiply received signal with PRBS reference
        y = np.convolve(yr, np.ones(Gp), mode = 'full')  # correlation type receiver

        dCap = y[Gp-1:y.size+1:Gp] > 0       # threshold detection
        BER[i] = np.sum(dCap!=d)/N           # Bit Error Rate
    ax.semilogy(EbN0dBs, BER, color = colors[k], marker = 'o',linestyle = '--',lw = 2, label = f"JSR = {JSR_dB}(dB)")

#  theoretical BER (when no Jammer)
SER_theory = 0.5 * scipy.special.erfc(np.sqrt(10**(EbN0dBs/10)))
ax.semilogy(EbN0dBs, SER_theory,color = colors[-1], linestyle = '-', lw = 2, label = "Theory BPSK AWGN")
ax.set_ylim(1e-5, 1)
ax.set_xlabel('Eb/N0(dB)');
ax.set_ylabel('SER ($P_s$)')
ax.set_title('Performance of BPSK DSSS(Gp = 31) in presence of a tone jammer and AWGN noise')
ax.legend()
plt.show()
plt.close()

#%% Program 13.16: bfsk mod.m: Generating coherent and non-coherent discrete-time BFSK signals
colors = plt.cm.jet(np.linspace(0, 1, 4))
def bfsk_mod(a, Fc, Fd, L, Fs, fsk_type):
    # Function to modulate an incoming binary stream using BFSK
    # a - input binary data stream (0's and 1's) to modulate
    # Fc - center frequency of the carrier in Hertz
    # Fd - frequency separation measured from Fc
    # L - number of samples in 1-bit period
    # Fs - Sampling frequency for discrete-time simulation
    # fsk_type - 'COHERENT' (default) or 'NONCOHERENT' FSK generation at each bit period when generating the carriers
    # s - BFSK modulated signal
    # t - generated time base for the modulated signal phase - initial phase generated by modulator, applicable only for coherent FSK.
    # It can be used when using coherent detection at Rx at - data waveform for the input data
    phase = 0
    at = np.kron(a, np.ones(int(L)))             # data to waveform
    t = np.arange(at.size)/Fs                    # time base
    if fsk_type == 'NONCOHERENT':
        c1 = np.cos(2 * np.pi * (Fc + Fd/2) * t + 2 * np.pi * np.random.rand())  # carrier 1 with random phase
        c2 = np.cos(2 * np.pi * (Fc - Fd/2) * t + 2 * np.pi * np.random.rand())  # carrier 2 with random phase
    else:
        phase = 2 * np.pi * np.random.rand()     # random phase from uniform distribution [0, 2*pi]
        c1 = np.cos(2 * np.pi * (Fc + Fd/2) * t + phase)   # carrier 1 with random phase
        c2 = np.cos(2 * np.pi * (Fc - Fd/2) * t + phase)   # carrier 2 with random phase
    s = at * c1 + (-at + 1) * c2  # BFSK signal (MUX selection)
    doPlot = 0
    if doPlot:
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        ax.plot(t, at, color = colors[0],  )
        ax.plot(t, s, color = colors[1], )
        ax.set_xlim(0, 0.0002)
        plt.show()
        plt.close()

    return s, t, phase, at

#%% Program 13.17: bfsk noncoherent demod.m: Square-law based non-coherent demodulator
def bfsk_noncoherent_demod(r, Fc, Fd, L, Fs):
    # Non-coherent demodulation of BFSK modulated signal
    # r - BFSK modulated signal at the receiver;
    # Fc - center frequency of the carrier in Hertz;
    # Fd - frequency separation measured from Fc;
    # L - number of samples in 1-bit period;
    # Fs - Sampling frequency for discrete-time simulation;
    # a_cap - data bits after demodulation;
    t = np.arange(r.size)/Fs # time base;
    F1 = (Fc + Fd/2)
    F2 = (Fc - Fd/2)

    ## define four basis functions
    p1c =  np.cos(2 * np.pi * F1 * t)
    p2c =  np.cos(2 * np.pi * F2 * t)
    p1s = -np.sin(2 * np.pi * F1 * t)
    p2s = -np.sin(2 * np.pi * F2 * t)

    ## multiply and integrate from 0 to Tb
    r1c = np.convolve(r*p1c, np.ones(L))
    r2c = np.convolve(r*p2c, np.ones(L))
    r1s = np.convolve(r*p1s, np.ones(L))
    r2s = np.convolve(r*p2s, np.ones(L))

    ## sample at every sampling instant
    r1c = r1c[L-1: r1c.size+1: L]
    r2c = r2c[L-1: r2c.size+1: L]
    r1s = r1s[L-1: r1s.size+1: L]
    r2s = r2s[L-1: r2s.size+1: L]

    x = r1c**2 + r1s**2
    y = r2c**2 + r2s**2   # square and add
    a_cap = (x - y) > 0   # compare and decide
    return a_cap

#%% Program 13.18: gen FH code table.m: Lookup table generation for frequency synthesizer
def gen_FH_code_table(Fbase, Fspace, Fs, Lh, N):
    # Generate frequency translation table for Frequency Hopping (FH)
    # Fbase - base frequency (Hz) of the hop
    # Fspace - channel spacing (Hz) of the hop
    # Fs - sampling frequency (Hz)
    # Lh - num of discrete time samples in each hop period
    # N - num of total hopping frequencies required(full period of LFSR)
    # Return the frequency translation table
    N = int(N)
    Lh = int(Lh)
    t = np.arange(Lh) / Fs           # time base for each hopping period
    freqTable = np.zeros((N, Lh))    # Table to store N different freq waveforms
    for i in range(N):               # generate frequency translation table for all N states
        Fi = Fbase + (i-1) * Fspace
        freqTable[i,:] = np.cos(2*np.pi*Fi*t)
    return freqTable

#%% Program 13.19: hopping chip waveform.m: Hopping frequency synthesizer
def hopping_chip_waveform(G, X, nHops, Lh, Fbase, Fspace, Fs):
    # Generate Frequency Hopping chip sequence based on m-sequence LFSR
    # G, X - Generator poly. and initial seed for underlying m-sequence
    # nHops - total number of hops needed
    # Lh - number of discrete time samples in each hop period
    # Fbase - base frequency (Hz) of the hop
    # Fspace - channel spacing (Hz) of the hop
    # Fs - sampling frequency (Hz)

    prbs, STATES = lfsr(G, X)                    # initialize LFSR
    STATES = (STATES - 1).astype(np.int64)
    N = prbs.size                                # PRBS period
    LFSRStates = repeatSequence(STATES , nHops)  # repeat LFSR states depending on nHops
    nHops = int(nHops)
    Lh = int(Lh)
    freqTable = gen_FH_code_table(Fbase, Fspace, Fs, Lh, N)  # freq translation
    c = np.zeros((nHops, Lh))                                # place holder for the hopping sequence waveform
    for i in range(nHops):                               # for each hop choose one freq wave based on LFSR state
        LFSRState = int(LFSRStates[i])                   # given LFSR state
        c[i, : ] = freqTable[LFSRState, :]               # choose corresponding freq. wave
    return c.flatten()

#%% Program 13.20: FHSS BFSK.m: Simulation of BFSK-FHSS system
# *************** Source *********************
nBits = 60           # number of source bits to transmit
Rb = int(20e3)       # bit rate of source information in bps

# ********** BFSK definitions *****************
fsk_type = 'NONCOHERENT'     # BFSK generation type at the transmitter
h = 1                        # modulation index (0.5 = coherent BFSK / 1 = non-coherent BFSK)
Fc = int(50e3)               # center frequency of BFSK

# ********** Frequency Allocation *************
Fbase = int(200e3)           # The center frequency of the first channel
Fspace = int(100e3)          # freq. separation between adjacent hopping channels
Fs = int(10e6)               # sufficiently high sampling frequency for discretization

# ********* Frequency Hopper definition *******
G = np.array([1, 0, 0, 1, 1])
X = np.array([0, 0, 0, 1])             # LFSR generator poly and seed
hopType = 'FAST_HOP'                   # FAST_HOP or SLOW_HOP for frequency hopping

# -------- Derived Parameters -------
Tb = 1/Rb                  # bit duration of each information bit.
L = int(Tb*Fs)             # num of discrete-time samples in each bit
Fd = int(h/Tb)             # frequency separation of BFSK frequencies

# Adjust num of samples in a hop duration based on hop type selected
if hopType == 'FAST_HOP':               # hop duration less than bit duration
    Lh = int(L/4)                       # 4 hops in a bit period
    nHops = 4*nBits                     # total number of Hops during the transmission
else:                                   # default set to SLOW_HOP: hop duration more than bit duration
    Lh = int(L*4)                       # 4 bit worth of samples in a hop period
    nHops = nBits/4                     # total number of Hops during the transmission

# ----- Simulate the individual blocks ----------------------
d = np.random.randint(0, 2, size = (nBits, ))                   # random information bits
s_m, t, phase, dt = bfsk_mod(d, Fc, Fd, L, Fs, fsk_type)        # BFSK modulation
c = hopping_chip_waveform(G, X, nHops, Lh, Fbase, Fspace, Fs)   # Hopping wfm
s = s_m * c                                                     # mix BFSK waveform with hopping frequency waveform

n = 0                 # Left to the reader -modify for AWGN noise(see prev chapters)
r = s + n             # received signal with noise
v = r * c             # mix received signal with synchronized hopping freq wave

d_cap = bfsk_noncoherent_demod(v, Fc, Fd, L, Fs)      # BFSK demod
bie = np.sum(d != d_cap)                              # Calculate bits in error
print(f"BER = {bie}")
# --------Plot waveforms at various stages of tx/rx-------
fig, axs = plt.subplots(2, 1, figsize = (10, 8), constrained_layout = True)
axs[0].plot(t, dt, color = 'r', ls = '-', lw = 2,  )
axs[0].set_title("Source bits-d(t)")
axs[1].plot(t, s_m, color = 'r', ls = '-', lw = 2,  )
axs[1].set_xlim(0, 2e-4)
axs[1].set_title("BFSK modulated-s_m(t)")
plt.show()
plt.close()

fig, axs = plt.subplots(3, 1, figsize = (10, 10), constrained_layout = True)
axs[0].plot(t, s_m, color = 'b', ls = '-', lw = 2, )
axs[0].set_title("BFSK modulated - s_m(t)")
axs[0].set_xlim(0, 2e-4)
axs[1].plot(t, c, color = 'r', ls = '-', lw = 2, )
axs[1].set_xlim(0, 2e-4)
axs[1].set_title("Hopping waveform at Tx - c(t)")
axs[2].plot(t, s, color = 'g', ls = '-', lw = 2, )
axs[2].set_xlim(0, 2e-4)
axs[2].set_title("FHSS signal - s(t)")
plt.show()
plt.close()

fig, axs = plt.subplots(3, 1, figsize = (10, 10), constrained_layout = True)
axs[0].plot(t, r, color = 'k', ls = '-', lw = 2, )
axs[0].set_xlim(0, 2e-4)
axs[0].set_title("Received signal - r(t)")
axs[1].plot(t, c, color = 'r', ls = '-', lw = 2, )
axs[1].set_xlim(0, 2e-4)
axs[1].set_title("Synced hopping waveform at Rx-c(t)")
axs[2].plot(t, v, color = 'b', ls = '-', lw = 2, )
axs[2].set_xlim(0, 2e-4)
axs[2].set_title("Signal after mixing with hop pattern-v(t)")
plt.show()
plt.close("all")


#%% Performance of Frequency hopping spread spectrum system over AWGN channel

from DigiCommPy.channels import awgn
from DigiCommPy.errorRates import ser_awgn
from tqdm import tqdm
# *************** Source *********************

nBits = 100000               # number of source bits to transmit
Rb = int(20e3)               # bit rate of source information in bps
# ********** BFSK definitions *****************
fsk_type = 'NONCOHERENT'     # BFSK generation type at the transmitter
h = 1                        # modulation index (0.5 = coherent BFSK / 1 = non-coherent BFSK)
Fc = int(50e3)               # center frequency of BFSK
# ********** Frequency Allocation *************
Fbase = int(200e3)           # The center frequency of the first channel
Fspace = int(100e3)          # freq. separation between adjacent hopping channels
Fs = int(10e6)               # sufficiently high sampling frequency for discretization
# ********* Frequency Hopper definition *******
G = np.array([1, 0, 0, 1, 1])
X = np.array([0, 0, 0, 1])      # LFSR generator poly and seed
hopType = 'FAST_HOP'            # FAST_HOP or SLOW_HOP for frequency hopping
# -------- Derived Parameters -------
Tb = 1/Rb                    # bit duration of each information bit.
L = int(Tb*Fs)               # num of discrete-time samples in each bit
Fd = int(h/Tb)               # frequency separation of BFSK frequencies
# Adjust num of samples in a hop duration based on hop type selected
if hopType == 'FAST_HOP':               # hop duration less than bit duration
    Lh = int(L/4)                       # 4 hops in a bit period
    nHops = 4*nBits                     # total number of Hops during the transmission
else:                                   # default set to SLOW_HOP: hop duration more than bit duration
    Lh = int(L*4)                       # 4 bit worth of samples in a hop period
    nHops = nBits/4                     # total number of Hops during the transmission
d = np.random.randint(0, 2, size = (nBits, ))                   # random information bits
s_m, t, phase, dt = bfsk_mod(d, Fc, Fd, L, Fs, fsk_type)        # BFSK modulation
c = hopping_chip_waveform(G, X, nHops, Lh, Fbase, Fspace, Fs)   # Hopping wfm
s = s_m * c                                                     # mix BFSK waveform with hopping frequency waveform

SNR_dBs = np.arange(-4, 17, 2)        # signal to noise ratios (dB)
BER = np.zeros(SNR_dBs.size)          # place holder for BER values

for i, SNR_dB in tqdm(enumerate(SNR_dBs)):
    # r = awgn(s, SNR_dB, L)

    Esym = L*np.sum(np.abs(s)**2)/(s.size)         # Calculate symbol energy
    N0 = Esym/(10**(SNR_dB/10));                    # Find the noise spectral density
    n_t = np.sqrt(N0/2)*np.random.randn(*s.shape);   # computed noise
    r = s + n_t;                                   # received signal

    v = r * c             # mix received signal with synchronized hopping freq wave
    d_cap = bfsk_noncoherent_demod(v, Fc, Fd, L, Fs).astype(np.int32)      # BFSK demod
    BER[i] = np.sum(d != d_cap)/d.size                                # Bit Error Rate
# SER_theory = 0.5 * scipy.special.erfc(np.sqrt(10**(SNR_dBs/10)))
SER_theory = ser_awgn(SNR_dBs, 'fsk', 2, "noncoherent")  # theory SER

colors = plt.cm.jet(np.linspace(0, 1, 4)) # colormap
fig, ax = plt.subplots(1, 1, figsize = (6, 5), constrained_layout = True)
ax.semilogy(SNR_dBs, BER, color = colors[0], marker = 'o', linestyle = '--', lw = 1, label = 'Sim' )
ax.semilogy(SNR_dBs, SER_theory, color = colors[1], linestyle = '-',lw = 2, label = 'Theory' )
ax.set_ylim(1e-4, 1)
ax.set_xlabel('Eb/N0(dB)')
ax.set_ylabel(r'SER ($P_s$)')
# ax.set_title('Probability of Symbol Error for BPSK over AWGN')
ax.legend()
plt.show()
plt.close()















































































































































































