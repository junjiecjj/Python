#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 16:48:15 2026

@author: jack
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 3 reproduction: SG-64-APSK With CP, P-ACF."""

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

def generatePeriodicJk(N, k):
    if k == 0:
        Jk = np.eye(N)
    else:
        tmp1 = np.zeros((N-k, k))
        tmp2 = np.eye(N-k)
        tmp3 = np.eye(k)
        tmp4 = np.zeros((k, N-k))
        Jk = np.block([[tmp1, tmp2], [tmp3, tmp4]])
    return Jk

#%% 参数设置
pi = np.pi
N = 128
kappa = 3.9867
Iter = 1000
FN = FFTmatrix(N)

Order = 64
radius = np.array([4.54e-5, 0.0067, 0.0815, 1.9983])
phase = 2*pi*np.arange(16)/16
Constellation = np.concatenate((radius[0]*np.exp(1j*phase), radius[1]*np.exp(1j*phase), radius[2]*np.exp(1j*phase), radius[3]*np.exp(1j*phase)))
Constellation = Constellation/np.sqrt(np.mean(np.abs(Constellation)**2))
AvgEnergy = np.mean(np.abs(Constellation)**2)
kappa = np.mean(np.abs(Constellation)**4)
print(f"Average energy = {AvgEnergy}")
print(f"SG-64-APSK kurtosis = {kappa}")

#%% CP-SC，公式(23)，理论结果
U = np.eye(N)
V = U.conj().T @ FN.conj().T
tilde_V = V * V.conj()
TheoAvePACF_SC = np.zeros(N)

for k in range(N):
    fk = np.exp(-1j*2*pi*k*np.arange(N)/N)
    bk = tilde_V @ fk
    r1 = N**2 if k == 0 else 0
    r2 = N
    r3 = (kappa-2)*np.linalg.norm(bk)**2
    TheoAvePACF_SC[k] = r1+r2+r3

TheoAvePACF_SC = np.abs(TheoAvePACF_SC)
TheoAvePACF_SC = TheoAvePACF_SC/TheoAvePACF_SC.max()+1e-10
TheoAvePACF_SC = np.fft.fftshift(TheoAvePACF_SC)

#%% CP-SC，公式(17)，Monte Carlo仿真
U = np.eye(N)
SimAvePACF_SC = np.zeros((Iter, N))

for k in range(N):
    Jk = generatePeriodicJk(N, k)
    for it in range(Iter):
        d = np.random.randint(Order, size=N)
        s = Constellation[d]
        x = U @ s
        rk = x.conj().T @ Jk @ x
        SimAvePACF_SC[it,k] = np.abs(rk)**2

Sim_PACF_SC_avg = SimAvePACF_SC.mean(axis=0)
Sim_PACF_SC_avg = Sim_PACF_SC_avg/Sim_PACF_SC_avg.max()+1e-10
Sim_PACF_SC_avg = np.fft.fftshift(Sim_PACF_SC_avg)

#%% CP-CDMA，公式(23)，理论结果
U = hadamard_matrix_sylvester(N)/np.sqrt(N)
V = U.conj().T @ FN.conj().T
tilde_V = V * V.conj()
TheoAvePACF_CDMA = np.zeros(N)

for k in range(N):
    fk = np.exp(-1j*2*pi*k*np.arange(N)/N)
    bk = tilde_V @ fk
    r1 = N**2 if k == 0 else 0
    r2 = N
    r3 = (kappa-2)*np.linalg.norm(bk)**2
    TheoAvePACF_CDMA[k] = r1+r2+r3

TheoAvePACF_CDMA = np.abs(TheoAvePACF_CDMA)
TheoAvePACF_CDMA = TheoAvePACF_CDMA/TheoAvePACF_CDMA.max()+1e-10
TheoAvePACF_CDMA = np.fft.fftshift(TheoAvePACF_CDMA)

#%% CP-CDMA，公式(17)，Monte Carlo仿真
U = hadamard_matrix_sylvester(N)/np.sqrt(N)
SimAvePACF_CDMA = np.zeros((Iter, N))

for k in range(N):
    Jk = generatePeriodicJk(N, k)
    for it in range(Iter):
        d = np.random.randint(Order, size=N)
        s = Constellation[d]
        x = U @ s
        rk = x.conj().T @ Jk @ x
        SimAvePACF_CDMA[it,k] = np.abs(rk)**2

Sim_PACF_CDMA_avg = SimAvePACF_CDMA.mean(axis=0)
Sim_PACF_CDMA_avg = Sim_PACF_CDMA_avg/Sim_PACF_CDMA_avg.max()+1e-10
Sim_PACF_CDMA_avg = np.fft.fftshift(Sim_PACF_CDMA_avg)

#%% CP-OTFS，公式(23)，理论结果
FFTN = 32
Neye = int(N/FFTN)
FFTM = FFTmatrix(FFTN)
eyeM = np.eye(Neye)
U = np.kron(FFTM, eyeM)
V = U.conj().T @ FN.conj().T
tilde_V = V * V.conj()
TheoAvePACF_OTFS = np.zeros(N)

for k in range(N):
    fk = np.exp(-1j*2*pi*k*np.arange(N)/N)
    bk = tilde_V @ fk
    r1 = N**2 if k == 0 else 0
    r2 = N
    r3 = (kappa-2)*np.linalg.norm(bk)**2
    TheoAvePACF_OTFS[k] = r1+r2+r3

TheoAvePACF_OTFS = np.abs(TheoAvePACF_OTFS)
TheoAvePACF_OTFS = TheoAvePACF_OTFS/TheoAvePACF_OTFS.max()+1e-10
TheoAvePACF_OTFS = np.fft.fftshift(TheoAvePACF_OTFS)

#%% CP-OTFS，公式(17)，Monte Carlo仿真
FFTN = 32
Neye = int(N/FFTN)
FFTM = FFTmatrix(FFTN)
eyeM = np.eye(Neye)
U = np.kron(FFTM, eyeM)
SimAvePACF_OTFS = np.zeros((Iter, N))

for k in range(N):
    Jk = generatePeriodicJk(N, k)
    for it in range(Iter):
        d = np.random.randint(Order, size=N)
        s = Constellation[d]
        x = U @ s
        rk = x.conj().T @ Jk @ x
        SimAvePACF_OTFS[it,k] = np.abs(rk)**2

Sim_PACF_OTFS_avg = SimAvePACF_OTFS.mean(axis=0)
Sim_PACF_OTFS_avg = Sim_PACF_OTFS_avg/Sim_PACF_OTFS_avg.max()+1e-10
Sim_PACF_OTFS_avg = np.fft.fftshift(Sim_PACF_OTFS_avg)

#%% CP-AFDM，公式(23)，理论结果
c1 = 1/128
c2 = 4/(3*pi)
U = IDAFT(c1, c2, N)
V = U.conj().T @ FN.conj().T
tilde_V = V * V.conj()
TheoAvePACF_AFDM = np.zeros(N)

for k in range(N):
    fk = np.exp(-1j*2*pi*k*np.arange(N)/N)
    bk = tilde_V @ fk
    r1 = N**2 if k == 0 else 0
    r2 = N
    r3 = (kappa-2)*np.linalg.norm(bk)**2
    TheoAvePACF_AFDM[k] = r1+r2+r3

TheoAvePACF_AFDM = np.abs(TheoAvePACF_AFDM)
TheoAvePACF_AFDM = TheoAvePACF_AFDM/TheoAvePACF_AFDM.max()+1e-10
TheoAvePACF_AFDM = np.fft.fftshift(TheoAvePACF_AFDM)

#%% CP-AFDM，公式(17)，Monte Carlo仿真
c1 = 1/128
c2 = 4/(3*pi)
U = IDAFT(c1, c2, N)
SimAvePACF_AFDM = np.zeros((Iter, N))

for k in range(N):
    Jk = generatePeriodicJk(N, k)
    for it in range(Iter):
        d = np.random.randint(Order, size=N)
        s = Constellation[d]
        x = U @ s
        rk = x.conj().T @ Jk @ x
        SimAvePACF_AFDM[it,k] = np.abs(rk)**2

Sim_PACF_AFDM_avg = SimAvePACF_AFDM.mean(axis=0)
Sim_PACF_AFDM_avg = Sim_PACF_AFDM_avg/Sim_PACF_AFDM_avg.max()+1e-10
Sim_PACF_AFDM_avg = np.fft.fftshift(Sim_PACF_AFDM_avg)

#%% CP-OFDM，公式(23)，理论结果
U = FN.conj().T
V = U.conj().T @ FN.conj().T
tilde_V = V * V.conj()
TheoAvePACF_OFDM = np.zeros(N)

for k in range(N):
    fk = np.exp(-1j*2*pi*k*np.arange(N)/N)
    bk = tilde_V @ fk
    r1 = N**2 if k == 0 else 0
    r2 = N
    r3 = (kappa-2)*np.linalg.norm(bk)**2
    TheoAvePACF_OFDM[k] = r1+r2+r3

TheoAvePACF_OFDM = np.abs(TheoAvePACF_OFDM)
TheoAvePACF_OFDM = TheoAvePACF_OFDM/TheoAvePACF_OFDM.max()+1e-10
TheoAvePACF_OFDM = np.fft.fftshift(TheoAvePACF_OFDM)

#%% CP-OFDM，公式(17)，Monte Carlo仿真
U = FN.conj().T
SimAvePACF_OFDM = np.zeros((Iter, N))

for k in range(N):
    Jk = generatePeriodicJk(N, k)
    for it in range(Iter):
        d = np.random.randint(Order, size=N)
        s = Constellation[d]
        x = U @ s
        rk = x.conj().T @ Jk @ x
        SimAvePACF_OFDM[it,k] = np.abs(rk)**2

Sim_PACF_OFDM_avg = SimAvePACF_OFDM.mean(axis=0)
Sim_PACF_OFDM_avg = Sim_PACF_OFDM_avg/Sim_PACF_OFDM_avg.max()+1e-10
Sim_PACF_OFDM_avg = np.fft.fftshift(Sim_PACF_OFDM_avg)

#%% Fig. 3画图
x = np.arange(-N//2, N//2)
fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
axs.plot(x, 10*np.log10(TheoAvePACF_SC), color='#F65314', linestyle='--', linewidth=2, label='CP-SC, Theoretical')
# axs.plot(x, 10*np.log10(Sim_PACF_SC_avg), color='#F65314', linestyle='none', marker='x', ms=8, markevery=4, label='CP-SC, Simulation')
axs.plot(x, 10*np.log10(TheoAvePACF_CDMA), color='#00A1F1', linestyle='--', linewidth=2, label='CP-CDMA, Theoretical')
# axs.plot(x, 10*np.log10(Sim_PACF_CDMA_avg), color='#00A1F1', linestyle='none', marker='1', ms=10, markevery=4, label='CP-CDMA, Simulation')
axs.plot(x, 10*np.log10(TheoAvePACF_OTFS), color='#05C349', linestyle='-.', linewidth=2, label='CP-OTFS, Theoretical')
# axs.plot(x, 10*np.log10(Sim_PACF_OTFS_avg), color='#05C349', linestyle='none', marker='o', ms=7, markevery=4, markerfacecolor='none', label='CP-OTFS, Simulation')
axs.plot(x, 10*np.log10(TheoAvePACF_AFDM), color='#B22222', linestyle=':', linewidth=2, label='CP-AFDM, Theoretical')
# axs.plot(x, 10*np.log10(Sim_PACF_AFDM_avg), color='#B22222', linestyle='none', marker='s', ms=6, markevery=4, markerfacecolor='none', label='CP-AFDM, Simulation')
axs.plot(x, 10*np.log10(TheoAvePACF_OFDM), color='#8A2BE2', linestyle='-', linewidth=2, label='CP-OFDM, Theoretical')
# axs.plot(x, 10*np.log10(Sim_PACF_OFDM_avg), color='#8A2BE2', linestyle='none', marker='+', ms=12, markevery=4, label='CP-OFDM, Simulation')
font1 = {'family':'Times New Roman', 'style':'normal', 'size':12}
font1 = FontProperties(family='Times New Roman', style='normal', size=20)
legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', fontsize=12, labelspacing=0.2, prop=font1)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')

bw = 2
axs.spines['bottom'].set_linewidth(bw)
axs.spines['left'].set_linewidth(bw)
axs.spines['right'].set_linewidth(bw)
axs.spines['top'].set_linewidth(bw)

axs.set_xlabel(r'Delay Index')
axs.set_ylabel(r'Normalized Sidelobe Level (dB)')
axs.set_title(r'SG-64-APSK With CP')
axs.set_xlim([-64, 64])
axs.set_ylim([-25, 0])
axs.set_xticks(np.arange(-64, 65, 16))
axs.set_yticks(np.arange(-25, 1, 5))
axs.tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=bw)
labels = axs.get_xticklabels()+axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(22) for label in labels]
axs.grid(linestyle=(0, (5, 10)), linewidth=0.5)
# plt.savefig("Fig3_TIT_With_CP.pdf", )
plt.show()
plt.close()






