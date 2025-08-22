#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 21:34:56 2025

@author: jack
"""


import scipy
import numpy as np
import matplotlib.pyplot as plt
# from scipy import signal
from scipy.constants import speed_of_light as c0
# from scipy.signal import fftconvolve
# import commpy
from Modulations import modulator

def Qfun(x):
    return 0.5 * scipy.special.erfc(x / np.sqrt(2))

def ser_awgn(EbN0dB, MOD_TYPE, M, COHERENCE = None):
    EbN0 = 10**(EbN0dB/10)
    EsN0 = np.log2(M) * EbN0
    SER = np.zeros(EbN0dB.size)
    if MOD_TYPE.lower() == "bpsk":
        SER = Qfun(np.sqrt(2 * EbN0))
    elif MOD_TYPE == "psk":
        if M == 2:
            SER = Qfun(np.sqrt(2 * EbN0))
        else:
            if M == 4:
                SER = 2 * Qfun(np.sqrt(2* EbN0)) - Qfun(np.sqrt(2 * EbN0))**2
            else:
                SER = 2 * Qfun(np.sin(np.pi/M) * np.sqrt(2 * EsN0))
    elif MOD_TYPE.lower() == "qam":
        SER = 1 - (1 - 2*(1 - 1/np.sqrt(M)) * Qfun(np.sqrt(3 * EsN0/(M - 1))))**2
    elif MOD_TYPE.lower() == "pam":
        SER = 2*(1-1/M) * Qfun(np.sqrt(6*EsN0/(M**2-1)))
    return SER

def ser_rayleigh(EbN0dB, MOD_TYPE, M):
    EbN0 = 10**(EbN0dB/10)
    EsN0 = np.log2(M) * EbN0
    SER = np.zeros(EbN0dB.size)
    if MOD_TYPE.lower() == "bpsk":
        SER = 1/2 * (1 - np.sqrt(EsN0/(1 + EsN0)))
    elif MOD_TYPE.lower() == "psk":
        SER = np.zeros(EsN0.size)
        for i in range(len(EsN0)):
            g = np.sin(np.pi/M)**2
            fun = lambda x: 1.0 / (1.0 + g * EsN0[i]/(np.sin(x)**2))
            SER[i] = 1/np.pi * scipy.integrate.quad(fun, 0, np.pi*(M-1)/M)[0]
    elif MOD_TYPE.lower() == "qam":
        SER = np.zeros(EsN0.size)
        for i in range(len(EsN0)):
            g = 1.5 / (M-1)
            fun = lambda x: 1.0 / (1.0 + g * EsN0[i]/np.sin(x)**2)
            SER[i] = 4/np.pi * (1 - 1/np.sqrt(M)) * scipy.integrate.quad(fun, 0, np.pi/2)[0] - 4/np.pi * (1 - 1/np.sqrt(M))**2 * scipy.integrate.quad(fun, 0, np.pi/4)[0]
    elif MOD_TYPE.lower() == "pam":
        SER = np.zeros(EsN0.size)
        for i in range(len(EsN0)):
            g = 3/(M**2 - 1)
            fun = lambda x: 1.0 / (1.0 + g * EsN0[i]/np.sin(x)**2)
            SER[i] = 2*(M-1)/(M*np.pi) * scipy.integrate.quad(fun, 0, np.pi/2)[0]
    return SER

def awgn(signal, snr_db, measured=True):
    """添加AWGN噪声"""
    if measured:
        signal_power = np.mean(np.abs(signal)**2)
    else:
        signal_power = 1.0  # 假设单位功率

    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
    return signal + noise

def range2time(range_val, c):
    """距离转换为时间"""
    return 2 * range_val / c

def speed2dop(speed, lambda_val):
    """速度转换为多普勒频率"""
    return 2 * speed / lambda_val

# %% ISAC Transmitter
# System parameters
c0 = 3e8  # speed of light
fc = 30e9  # carrier frequency
lambd = c0 / fc  # wavelength
M = 1024  # number of subcarriers
N = 15  # number of symbols
delta_f = 120e3  # subcarrier spacing
T = 1 / delta_f  # symbol duration
Tcp = T / 4  # cyclic prefix duration
Ts = T + Tcp  # total symbol duration
L = 10
CPsize = int(N / 4)  # cyclic prefix length
# bitsPerSymbol = 2  # bits per symbol
# qam = 2 ** bitsPerSymbol  # 4-QAM modulation

# Transmit data
QAM_mod = 4
bps = int(np.log2(QAM_mod))
MOD_TYPE = "qam"
modem, Es, bps = modulator(MOD_TYPE, QAM_mod)
map_table, demap_table = modem.getMappTable()


def sensingSignalGen(TxSignal_cp, range_val, velocity, SNR):
    """第一个函数：生成传感信号"""
    global c0, lambd, M, delta_f

    delay = 2 * np.array(range_val) / c0
    h_gain = np.exp(1j * 2 * np.pi * np.random.rand(len(delay)))
    doppler = 2 * np.array(velocity) / lambd

    RxSignal = np.zeros(len(TxSignal_cp), dtype=complex)

    for p in range(len(delay)):
        ii = np.arange(len(TxSignal_cp))
        d = np.exp(1j * 2 * np.pi * doppler[p] * ii / (delta_f * M))
        tau = np.exp(-1j * 2 * np.pi * delay[p] * delta_f * M / len(TxSignal_cp) * ii)

        # 频域处理
        RxSignal += h_gain[p] * np.fft.ifft(np.fft.fft(TxSignal_cp * d) * tau)

    # 添加噪声
    noise_power = 10**(-SNR/10)
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(RxSignal)) + 1j * np.random.randn(len(RxSignal)))
    RxSignal += noise

    return RxSignal

def MUSICforOFDMsensing(CIM, k):
    """第二个函数：MUSIC算法"""
    global M, N, c0, delta_f, lambd, Ts

    # 范围估计
    R_range = CIM @ CIM.conj().T / N
    D, V = np.linalg.eig(R_range)
    ind_D = np.argsort(D)[::-1]
    U_n = V[:, ind_D[k:]]

    delay = np.linspace(0, 2 * 100 / c0 * delta_f, M)
    ii = np.arange(M)
    A = np.exp(-1j * 2 * np.pi * np.outer(ii, delay))

    P_music_range = np.zeros(len(delay))
    for jj in range(len(delay)):
        a = A[:, jj]
        P_music_range[jj] = 1 / np.abs(a.conj().T @ U_n @ U_n.conj().T @ a)

    # 速度估计
    R_dop = CIM.conj().T @ CIM / M
    D, V = np.linalg.eig(R_dop)
    ind_D = np.argsort(D)[::-1]
    U_n = V[:, ind_D[k:]]

    doppler = np.linspace(0, 2 * 100 / lambd * Ts, M)
    ii = np.arange(N)
    A = np.exp(1j * 2 * np.pi * np.outer(ii, doppler))

    P_music_velo = np.zeros(len(doppler))
    for jj in range(len(doppler)):
        a = A[:, jj]
        P_music_velo[jj] = 1 / np.abs(a.conj().T @ U_n @ U_n.conj().T @ a)

    return P_music_range, P_music_velo

def ESPRITforOFDMsensing(CIM, k):
    """第三个函数：ESPRIT算法"""
    global lambd, delta_f, c0, Ts

    M_dim, N_dim = CIM.shape

    # 范围估计
    z = np.vstack([CIM[:M_dim-1, :], CIM[1:M_dim, :]])
    R_zz = z @ z.conj().T / N_dim
    U, S, Vh = scipy.linalg.svd(R_zz)
    Es = U[:, :k]

    Esx = Es[:M_dim-1, :]
    Esy = Es[M_dim-1:, :]

    EE = np.hstack([Esx, Esy])
    F, _, _ = scipy.linalg.svd(EE.conj().T @ EE)
    F = F[:, -k:]
    F1 = F[:k, :]
    F2 = F[k:, :]
    psi = -F1 @ np.linalg.inv(F2)
    D, _ = np.linalg.eig(psi)

    phi = np.angle(D)
    phi[phi > 0] = phi[phi > 0] - 2 * np.pi
    tau = -phi / (2 * np.pi * delta_f)
    range_est = tau * c0 / 2

    # 多普勒估计
    z = np.hstack([CIM[:, :N_dim-1], CIM[:, 1:N_dim]])
    R_zz = z.conj().T @ z / M_dim
    U, S, Vh = scipy.linalg.svd(R_zz)
    Es = U[:, :k]

    Esx = Es[:N_dim-1, :]
    Esy = Es[N_dim-1:, :]

    EE = np.hstack([Esx, Esy])
    F, _, _ = scipy.linalg.svd(EE.conj().T @ EE)
    F = F[:, -k:]
    F1 = F[:k, :]
    F2 = F[k:, :]
    psi = -F1 @ np.linalg.inv(F2)
    D, _ = np.linalg.eig(psi)

    phi = np.angle(D)
    phi[phi < 0] = phi[phi < 0] + 2 * np.pi
    doppler = phi / (2 * np.pi * Ts)
    velocity_est = doppler * lambd / 2

    return np.mean(range_est), np.mean(velocity_est)

def cccSensing(RxSignal, TxSignal_cp, mildM, Qbar, mildQ):
    """第四个函数：CCC传感"""
    # 重新分组原始接收信号
    mildN = int((len(TxSignal_cp) - Qbar - mildQ) / (mildM - Qbar))
    Rx_sub = np.zeros((mildM, mildN), dtype=complex)
    Tx_sub = np.zeros((mildM, mildN), dtype=complex)

    for ii in range(mildN):
        start_idx = ii * (mildM - Qbar)
        Rx_sub[:, ii] = RxSignal[start_idx:start_idx + mildM]
        Tx_sub[:, ii] = TxSignal_cp[start_idx:start_idx + mildM]

    # 添加VCP
    for ii in range(mildN):
        start_idx = ii * (mildM - Qbar) + mildM
        Rx_sub[:mildQ, ii] += RxSignal[start_idx:start_idx + mildQ]

    # 互相关
    r_cc = np.fft.ifft(np.fft.fft(Rx_sub, axis=0) * np.conj(np.fft.fft(Tx_sub, axis=0)), axis=0)
    RDM = np.fft.fft(r_cc.conj().T, 10 * mildN, axis=0)

    return np.argmax(np.max(np.abs(r_cc), axis=1)), RDM

# 主函数
def main():
    global c0, fc, lambd, M, N, delta_f, Ts, CPsize, bitsPerSymbol, qam

    # ISAC Transmitter
    # 生成数据
    data = np.random.randint(0, qam, (M, N))
    TxData = qammod(data, qam)

    # OFDM调制
    TxSignal = np.fft.ifft(TxData, axis=0)
    TxSignal_cp = np.vstack([TxSignal[-CPsize:, :], TxSignal])
    TxSignal_cp = TxSignal_cp.reshape(-1)

    # 信道
    SNR = 30
    r = [30]
    v = [20]
    RxSignal = sensingSignalGen(TxSignal_cp, r, v, SNR)
    k = len(r)

    # OFDM雷达接收器
    # 1. 2DFFT基于方法
    Rx = RxSignal[:len(TxSignal_cp)].reshape(-1, N)
    Rx = Rx[CPsize:CPsize + M, :]
    Rx_dem = np.fft.fft(Rx, axis=0)
    CIM_2dfft = Rx_dem * np.conj(TxData)

    RDM_2dfft = np.fft.fft(np.fft.ifft(CIM_2dfft, axis=0).conj().T, 10 * N, axis=0)

    # 绘制距离多普勒图
    plt.figure(1)
    range_2dfft = np.linspace(0, c0/(2*delta_f), M+1)[:M]
    velocity_2dfft = np.linspace(0, lambd/2/Ts, 10*N+1)[:10*N]

    X, Y = np.meshgrid(range_2dfft, velocity_2dfft)
    RDM_2dfft_norm = 10 * np.log10(np.abs(RDM_2dfft) / np.max(np.abs(RDM_2dfft)))

    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, RDM_2dfft_norm, cmap='viridis')
    plt.title('2D-FFT based method')
    plt.xlabel('range(m)')
    plt.ylabel('velocity(m/s)')
    plt.show()

    # 2. CCC-based方法
    plt.figure(2)
    mildM = 512
    Qbar = 64
    mildQ = 128

    r_cc, RDM = cccSensing(RxSignal, TxSignal_cp, mildM, Qbar, mildQ)

    Tsa = 1 / delta_f / M
    mildN = int((len(TxSignal_cp) - Qbar - mildQ) / (mildM - Qbar))
    range_ccc = np.linspace(0, c0/2 * Tsa * mildM, mildM+1)[:mildM]
    doppler_ccc = np.linspace(0, lambd/(mildM-Qbar)/Tsa/2, 10*mildN+1)[:10*mildN]

    RDM_norm = 10 * np.log10(np.abs(RDM) / np.max(np.abs(RDM)))
    X, Y = np.meshgrid(range_ccc, doppler_ccc)

    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, RDM_norm, cmap='viridis')
    plt.title('CCC based method')
    plt.xlabel('range(m)')
    plt.ylabel('velocity(m/s)')
    plt.show()

    # 3.1 MUSIC based方法
    CIM = Rx_dem * np.conj(TxData)
    P_music_range, P_music_velo = MUSICforOFDMsensing(CIM, k)

    plt.figure(3)
    plt.suptitle('MUSIC for OFDM sensing')

    plt.subplot(1, 2, 1)
    plt.plot(np.linspace(0, 100, len(P_music_range)), np.abs(P_music_range)/np.max(np.abs(P_music_range)))
    plt.ylabel('Pmusic')
    plt.xlabel('range(m)')
    plt.ylim([1e-3, 1])
    plt.title('MUSIC for range estimation')

    plt.subplot(1, 2, 2)
    plt.plot(np.linspace(0, 100, M), np.abs(P_music_velo)/np.max(np.abs(P_music_velo)))
    plt.ylabel('Pmusic')
    plt.xlabel('velocity(m/s)')
    plt.ylim([1e-3, 1])
    plt.title('MUSIC for velocity estimation')
    plt.show()

    # 3.2 ESPRIT based方法
    range_est, velocity_est = ESPRITforOFDMsensing(CIM, k)
    print('The estimation result of TLS-ESPRIT is :')
    print(f'Range = {range_est}')
    print(f'Velocity = {velocity_est}')

if __name__ == "__main__":
    main()





























































































































































