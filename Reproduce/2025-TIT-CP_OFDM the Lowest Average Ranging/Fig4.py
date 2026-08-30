#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 4 reproduction: 16-QAM Without CP, A-ACF."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from Modulations import modulator

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['legend.fontsize'] = 18
np.random.seed(42)

#%%
def FFTmatrix(L):
    mat = np.zeros((L, L), dtype=complex)
    ll = np.arange(L)
    for i in range(L):
        mat[i,:] = np.exp(-1j*2.0*np.pi*i*ll/L)/np.sqrt(L)
    return mat

def hadamard_matrix_sylvester(n):
    if n == 1:
        return np.array([[1]])
    else:
        H_prev = hadamard_matrix_sylvester(n//2)
        H = np.kron(H_prev, np.array([[1, 1], [1, -1]]))
        return H

def IDAFT(c1, c2, N):
    F = np.fft.fft(np.eye(N))
    F = F/np.linalg.norm(F, ord=2)
    n = np.arange(N)
    L1 = np.diag(np.exp(-1j*2*np.pi*c1*(n**2)))
    L2 = np.diag(np.exp(-1j*2*np.pi*c2*(n**2)))
    A = L2 @ F @ L1
    return A.conj().T

def generateAperiodicJk(N, k):
    if k == 0:
        Jk = np.eye(N)
    else:
        Jk = np.zeros((N, N))
        Jk[:N-k,k:N] = np.eye(N-k)
    return Jk

#%% 参数设置
pi = np.pi
N = 128
kappa = 1.32
Iter = 1000
FN = FFTmatrix(N)

MOD_TYPE = "qam"
Order = 16
modem, Es, bps = modulator(MOD_TYPE, Order)
Constellation = modem.constellation/np.sqrt(Es)
AvgEnergy = np.mean(np.abs(Constellation)**2)

#%% SC，公式(40)，理论结果
U = np.eye(N)
TheoAveAACF_SC = np.zeros(N)

for k in range(N):
    Jk = generateAperiodicJk(N, k)
    unH_Jk_un = np.diag(U.conj().T @ Jk @ U)
    r1 = N**2 if k == 0 else 0
    r2 = N-k
    r3 = (kappa-2)*np.sum(np.abs(unH_Jk_un)**2)
    TheoAveAACF_SC[k] = r1+r2+r3

TheoAveAACF_SC = np.abs(TheoAveAACF_SC)
TheoAveAACF_SC = TheoAveAACF_SC/TheoAveAACF_SC.max()+1e-10
TheoAveAACF_SC = np.concatenate((TheoAveAACF_SC[:0:-1], TheoAveAACF_SC))

#%% SC，公式(17)，Monte Carlo仿真
U = np.eye(N)
SimAveAACF_SC = np.zeros((Iter, N))

for k in range(N):
    Jk = generateAperiodicJk(N, k)
    for it in range(Iter):
        d = np.random.randint(Order, size=N)
        s = Constellation[d]
        x = U @ s
        rk = x.conj().T @ Jk @ x
        SimAveAACF_SC[it,k] = np.abs(rk)**2

Sim_AACF_SC_avg = SimAveAACF_SC.mean(axis=0)
Sim_AACF_SC_avg = Sim_AACF_SC_avg/Sim_AACF_SC_avg.max()+1e-10
Sim_AACF_SC_avg = np.concatenate((Sim_AACF_SC_avg[:0:-1], Sim_AACF_SC_avg))

#%% CDMA，公式(40)，理论结果
U = hadamard_matrix_sylvester(N)/np.sqrt(N)
TheoAveAACF_CDMA = np.zeros(N)

for k in range(N):
    Jk = generateAperiodicJk(N, k)
    unH_Jk_un = np.diag(U.conj().T @ Jk @ U)
    r1 = N**2 if k == 0 else 0
    r2 = N-k
    r3 = (kappa-2)*np.sum(np.abs(unH_Jk_un)**2)
    TheoAveAACF_CDMA[k] = r1+r2+r3

TheoAveAACF_CDMA = np.abs(TheoAveAACF_CDMA)
TheoAveAACF_CDMA = TheoAveAACF_CDMA/TheoAveAACF_CDMA.max()+1e-10
TheoAveAACF_CDMA = np.concatenate((TheoAveAACF_CDMA[:0:-1], TheoAveAACF_CDMA))

#%% CDMA，公式(17)，Monte Carlo仿真
U = hadamard_matrix_sylvester(N)/np.sqrt(N)
SimAveAACF_CDMA = np.zeros((Iter, N))

for k in range(N):
    Jk = generateAperiodicJk(N, k)
    for it in range(Iter):
        d = np.random.randint(Order, size=N)
        s = Constellation[d]
        x = U @ s
        rk = x.conj().T @ Jk @ x
        SimAveAACF_CDMA[it,k] = np.abs(rk)**2

Sim_AACF_CDMA_avg = SimAveAACF_CDMA.mean(axis=0)
Sim_AACF_CDMA_avg = Sim_AACF_CDMA_avg/Sim_AACF_CDMA_avg.max()+1e-10
Sim_AACF_CDMA_avg = np.concatenate((Sim_AACF_CDMA_avg[:0:-1], Sim_AACF_CDMA_avg))

#%% OTFS，公式(40)，理论结果
FFTN = 32
Neye = int(N/FFTN)
FFTM = FFTmatrix(FFTN)
eyeM = np.eye(Neye)
U = np.kron(FFTM, eyeM)
TheoAveAACF_OTFS = np.zeros(N)

for k in range(N):
    Jk = generateAperiodicJk(N, k)
    unH_Jk_un = np.diag(U.conj().T @ Jk @ U)
    r1 = N**2 if k == 0 else 0
    r2 = N-k
    r3 = (kappa-2)*np.sum(np.abs(unH_Jk_un)**2)
    TheoAveAACF_OTFS[k] = r1+r2+r3

TheoAveAACF_OTFS = np.abs(TheoAveAACF_OTFS)
TheoAveAACF_OTFS = TheoAveAACF_OTFS/TheoAveAACF_OTFS.max()+1e-10
TheoAveAACF_OTFS = np.concatenate((TheoAveAACF_OTFS[:0:-1], TheoAveAACF_OTFS))

#%% OTFS，公式(17)，Monte Carlo仿真
FFTN = 32
Neye = int(N/FFTN)
FFTM = FFTmatrix(FFTN)
eyeM = np.eye(Neye)
U = np.kron(FFTM, eyeM)
SimAveAACF_OTFS = np.zeros((Iter, N))

for k in range(N):
    Jk = generateAperiodicJk(N, k)
    for it in range(Iter):
        d = np.random.randint(Order, size=N)
        s = Constellation[d]
        x = U @ s
        rk = x.conj().T @ Jk @ x
        SimAveAACF_OTFS[it,k] = np.abs(rk)**2

Sim_AACF_OTFS_avg = SimAveAACF_OTFS.mean(axis=0)
Sim_AACF_OTFS_avg = Sim_AACF_OTFS_avg/Sim_AACF_OTFS_avg.max()+1e-10
Sim_AACF_OTFS_avg = np.concatenate((Sim_AACF_OTFS_avg[:0:-1], Sim_AACF_OTFS_avg))

#%% AFDM，公式(40)，理论结果
c1 = 1/128
c2 = 4/(3*pi)
U = IDAFT(c1, c2, N)
TheoAveAACF_AFDM = np.zeros(N)

for k in range(N):
    Jk = generateAperiodicJk(N, k)
    unH_Jk_un = np.diag(U.conj().T @ Jk @ U)
    r1 = N**2 if k == 0 else 0
    r2 = N-k
    r3 = (kappa-2)*np.sum(np.abs(unH_Jk_un)**2)
    TheoAveAACF_AFDM[k] = r1+r2+r3

TheoAveAACF_AFDM = np.abs(TheoAveAACF_AFDM)
TheoAveAACF_AFDM = TheoAveAACF_AFDM/TheoAveAACF_AFDM.max()+1e-10
TheoAveAACF_AFDM = np.concatenate((TheoAveAACF_AFDM[:0:-1], TheoAveAACF_AFDM))

#%% AFDM，公式(17)，Monte Carlo仿真
c1 = 1/128
c2 = 4/(3*pi)
U = IDAFT(c1, c2, N)
SimAveAACF_AFDM = np.zeros((Iter, N))

for k in range(N):
    Jk = generateAperiodicJk(N, k)
    for it in range(Iter):
        d = np.random.randint(Order, size=N)
        s = Constellation[d]
        x = U @ s
        rk = x.conj().T @ Jk @ x
        SimAveAACF_AFDM[it,k] = np.abs(rk)**2

Sim_AACF_AFDM_avg = SimAveAACF_AFDM.mean(axis=0)
Sim_AACF_AFDM_avg = Sim_AACF_AFDM_avg/Sim_AACF_AFDM_avg.max()+1e-10
Sim_AACF_AFDM_avg = np.concatenate((Sim_AACF_AFDM_avg[:0:-1], Sim_AACF_AFDM_avg))

#%% OFDM，公式(40)，理论结果
U = FN.conj().T
TheoAveAACF_OFDM = np.zeros(N)

for k in range(N):
    Jk = generateAperiodicJk(N, k)
    unH_Jk_un = np.diag(U.conj().T @ Jk @ U)
    r1 = N**2 if k == 0 else 0
    r2 = N-k
    r3 = (kappa-2)*np.sum(np.abs(unH_Jk_un)**2)
    TheoAveAACF_OFDM[k] = r1+r2+r3

TheoAveAACF_OFDM = np.abs(TheoAveAACF_OFDM)
TheoAveAACF_OFDM = TheoAveAACF_OFDM/TheoAveAACF_OFDM.max()+1e-10
TheoAveAACF_OFDM = np.concatenate((TheoAveAACF_OFDM[:0:-1], TheoAveAACF_OFDM))

#%% OFDM，公式(17)，Monte Carlo仿真
U = FN.conj().T
SimAveAACF_OFDM = np.zeros((Iter, N))

for k in range(N):
    Jk = generateAperiodicJk(N, k)
    for it in range(Iter):
        d = np.random.randint(Order, size=N)
        s = Constellation[d]
        x = U @ s
        rk = x.conj().T @ Jk @ x
        SimAveAACF_OFDM[it,k] = np.abs(rk)**2

Sim_AACF_OFDM_avg = SimAveAACF_OFDM.mean(axis=0)
Sim_AACF_OFDM_avg = Sim_AACF_OFDM_avg/Sim_AACF_OFDM_avg.max()+1e-10
Sim_AACF_OFDM_avg = np.concatenate((Sim_AACF_OFDM_avg[:0:-1], Sim_AACF_OFDM_avg))

#%% Fig. 4画图
x = np.arange(-N+1, N)
fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
axs.plot(x, 10*np.log10(TheoAveAACF_SC), color='#F65314', linestyle='--', linewidth=2, label='SC')
# axs.plot(x, 10*np.log10(Sim_AACF_SC_avg), color='#F65314', linestyle='none', marker='x', ms=8, markevery=8, label='SC, Simulation')
axs.plot(x, 10*np.log10(TheoAveAACF_CDMA), color='#00A1F1', linestyle='--', linewidth=2, label='CDMA')
# axs.plot(x, 10*np.log10(Sim_AACF_CDMA_avg), color='#00A1F1', linestyle='none', marker='1', ms=10, markevery=8, label='CDMA, Simulation')
axs.plot(x, 10*np.log10(TheoAveAACF_OTFS), color='#05C349', linestyle='-.', linewidth=2, label='OTFS')
# axs.plot(x, 10*np.log10(Sim_AACF_OTFS_avg), color='#05C349', linestyle='none', marker='o', ms=7, markevery=8, markerfacecolor='none', label='OTFS, Simulation')
axs.plot(x, 10*np.log10(TheoAveAACF_AFDM), color='#000000', linestyle=':', linewidth=2, label='AFDM')
# axs.plot(x, 10*np.log10(Sim_AACF_AFDM_avg), color='#B22222', linestyle='none', marker='s', ms=6, markevery=8, markerfacecolor='none', label='AFDM, Simulation')
axs.plot(x, 10*np.log10(TheoAveAACF_OFDM), color='#8A2BE2', linestyle='-', linewidth=2, label='OFDM')
# axs.plot(x, 10*np.log10(Sim_AACF_OFDM_avg), color='#8A2BE2', linestyle='none', marker='+', ms=12, markevery=8, label='OFDM, Simulation')
font1 = {'family':'Times New Roman', 'style':'normal', 'size':16}
# font1 = FontProperties(family='Times New Roman', style='normal', size=20)
legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', fontsize=16, labelspacing=0.2, prop=font1)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')

bw = 2
axs.spines['bottom'].set_linewidth(bw)
axs.spines['left'].set_linewidth(bw)
axs.spines['right'].set_linewidth(bw)
axs.spines['top'].set_linewidth(bw)

axs.set_xlabel(r'Delay Index')
axs.set_ylabel(r'Ambiguity Level (dB)')
# axs.set_title(r'16-QAM Without CP')
axs.set_xlim([-128, 128])
axs.set_ylim([-45, 0])
axs.set_xticks(np.arange(-128, 129, 32))
axs.set_yticks(np.arange(-45, 1, 5))
axs.tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=bw)
labels = axs.get_xticklabels()+axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(22) for label in labels]
axs.grid(linestyle=(0, (5, 10)), linewidth=0.5)
plt.savefig("Fig4_TIT.pdf")
plt.show()
plt.close()
