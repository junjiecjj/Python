#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 21:49:27 2025

@author: jack
"""
import scipy
import numpy as np
# import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
import commpy
from DigiCommPy.modem import PSKModem, QAMModem, PAMModem, FSKModem
from DigiCommPy.channels import awgn
from DigiCommPy.errorRates import ser_awgn

#%%
def generateJk( N, k):
    if k < 0:
        k = N+k
    if k == 0:
        Jk = np.eye(N)
    elif k > 0:
        tmp1 = np.zeros((k, N-k))
        tmp2 = np.eye(k)
        tmp3 = np.eye(N-k)
        tmp4 = np.zeros((N - k, k))
        Jk = np.block([[tmp1, tmp2], [tmp3, tmp4]])
    return Jk

def srrcFunction(beta, L, span, Tsym = 1):
    # Function for generating rectangular pulse for the given inputs
    # L - oversampling factor (number of samples per symbol)
    # span - filter span in symbol durations
    # Returns the output pulse p(t) that spans the discrete-time base -span:1/L:span. Also returns the filter delay.
    t = np.arange(-span*Tsym/2, span*Tsym/2 + 0.5/L, Tsym/L)
    A = np.sin(np.pi*t*(1-beta)/Tsym) + 4*beta*t/Tsym * np.cos(np.pi*t*(1+beta)/Tsym)
    B = np.pi*t/Tsym * (1-(4*beta*t/Tsym)**2)
    p = 1/np.sqrt(Tsym) * A/B
    p[np.argwhere(np.isnan(p))] = 1
    p[np.argwhere(np.isinf(p))] = beta/(np.sqrt(2*Tsym)) * ((1+2/np.pi)*np.sin(np.pi/(4*beta)) + (1-2/np.pi)*np.cos(np.pi/(4*beta)))
    filtDelay = (len(p)-1)/2
    p = p / np.sqrt(np.sum(np.power(p, 2))) # both Add and Delete this line is OK.
    return p, t, filtDelay

## CDMA, U
def hadamard_matrix_sylvester(n):
    """Sylvester 构造法生成 Hadamard 矩阵（n 必须是 2 的幂）"""
    if n == 1:
        return np.array([[1]])
    else:
        H_prev = hadamard_matrix_sylvester(n // 2)
        H = np.kron(H_prev, np.array([[1, 1], [1, -1]]))
        return H

# AFDM, U
def IDAFT(c1, c2, N):
    """
    AFDM调制函数
    参数:
        x : 输入信号 (Nx1 列向量)
        c1 : 第一个调频参数
        c2 : 第二个调频参数
    返回:
        out : 调制输出信号
    """
    # N = x.shape[0]

    # 创建DFT矩阵并归一化
    F = np.fft.fft(np.eye(N))
    F = F / np.linalg.norm(F, ord = 2)

    # 创建L1和L2对角矩阵
    n = np.arange(N)
    L1 = np.diag(np.exp(-1j * 2 * np.pi * c1 * (n**2)))
    L2 = np.diag(np.exp(-1j * 2 * np.pi * c2 * (n**2)))

    # 构建AFDM矩阵
    A = L2 @ F @ L1
    # 计算调制输出 (注意MATLAB的'是共轭转置，Python用.conj().T)
    return A.conj().T

#%%
Tsym = 1
pi = np.pi
N = 128       # 符号数
L = 6        # 过采样率
alpha = 0.3  # 滚降因子

t, p = commpy.filters.rrcosfilter(L*N , alpha, Tsym, L/Tsym)
p = p / np.sqrt(np.mean(np.abs(p)**2))
P = scipy.linalg.circulant(p)

MOD_TYPE = "psk"
coherence = 'coherent' #'coherent'/'noncoherent'-only for FSK
modem_dict = {'psk': PSKModem,'qam':QAMModem,'pam':PAMModem,'fsk':FSKModem}
M = 16

if MOD_TYPE.lower() == 'fsk':
    modem = modem_dict[MOD_TYPE.lower()](M, coherence)   # choose modem from dictionary
else: # for all other modulations
    modem = modem_dict[MOD_TYPE.lower()](M)              #  choose modem from dictionary

Tarnum = 3
Tar_del_lst = (np.linspace(0, 1, Tarnum + 1, endpoint = 0 )[1:] * L*N).astype(int)
beta_lst = np.ones(len(Tar_del_lst))   # np.random.rand(len(Tar_del_lst)) + 0.1

#%% gene U-matrix for different modulation system.
# OFDM
U = scipy.linalg.dft(N)/np.sqrt(N)

# # AFDM
# c1 = 1/128
# c2 = 4/(3*np.pi)
# U = IDAFT(c1, c2, N)

# # OTFS
# FFTN = 32
# Neye = int(N/FFTN)
# FFTM = scipy.linalg.dft(FFTN)/np.sqrt(FFTN)
# eyeM = np.eye(Neye)
# U = np.kron(FFTM, eyeM)

# # CDMA
# U = hadamard_matrix_sylvester(N)/np.sqrt(N)

#%%
Iter = 10
# ACF = np.zeros((len(Tar_del_lst), N))
ACF_profile = np.zeros((Iter, L*N), dtype = complex)

for it in range(Iter):
    d = np.random.randint(low = 0, high = M, size = N)
    s = modem.modulate(d)
    x = U @ s
    x_up = np.vstack((x, np.zeros((L-1, x.size)))).T.flatten()
    x_tilde = P @ x_up


    y = np.zeros(L*N, dtype = complex)
    for k, delay in enumerate(Tar_del_lst):
        beta = beta_lst[k]
        Jk = generateJk(L*N, delay)
        y += beta * Jk @ x_tilde
        # ACF[k] = x_tilde.conj().T @ Jk @ x_tilde
    for lag in range(L*N):
        JkT = generateJk(L*N, lag).T
        ACF_profile[it, lag] = x_tilde.conj().T @ JkT @ y

Amplitude = np.abs(ACF_profile).mean(axis = 0)

Amplitude = Amplitude/Amplitude.max() + 1e-10


#%% plot together
colors = plt.cm.jet(np.linspace(0, 1, 5))

lags = np.arange(L*N)
fig, axs = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
axs.plot(lags, 10 * np.log10(Amplitude), color='b', linestyle='-', label='Range Profile',)

legend1 = axs.legend(loc='best', borderaxespad=0,  edgecolor='black', fontsize = 18)

axs.set_xlabel(r'Delay Index', )
axs.set_ylabel(r'Ambiguity Level (dB)', )
# axs.set_xlim([-200, 200])

out_fig = plt.gcf()
# filepath2 = '/home/jack/snap/'
out_fig.savefig('Fig3.png', )
out_fig.savefig('Fig3.pdf', )
plt.show()
plt.close()


























































































































































































































































