#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 18:08:02 2025

@author: jack
"""

# https://github.com/bb16177/OTFS-Simulation
# https://github.com/ironman1996/OTFS-simple-simulation
# https://github.com/eric-hs-rou/doubly-dispersive-channel-simulation
# https://zhuanlan.zhihu.com/p/608867803

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import scipy
import commpy
from Modulations import modulator

#%%
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

def ISFFT(X):
    """
    Inverse Symplectic Finite Fourier Transform
    Parameters:
        X : 2D numpy array (m x n)
    Returns:
        X_out : 2D numpy array after ISFFT
    """
    M, N = X.shape
    # ISFFT: DFT along rows (delay domain) and IDFT along columns (Doppler domain)
    X_out = np.fft.ifft(np.fft.fft(X, n=M, axis=0), n=N, axis=1) * np.sqrt(N / M)
    return X_out

def SFFT(X):
    """
    Symplectic Finite Fourier Transform
    Parameters:
        X : 2D numpy array (m x n)
    Returns:
        X_out : 2D numpy array after SFFT
    """
    M, N = X.shape
    # SFFT: IDFT along rows (delay domain) and DFT along columns (Doppler domain)
    X_out = np.fft.fft(np.fft.ifft(X, n=M, axis=0), n=N, axis=1) * np.sqrt(M / N)
    return X_out

def Heisenberg(M, N, X_tf):
    # 海森堡变换: TF域 → 时域
    s = np.zeros(M*N, dtype=complex)
    for n in range(N):
        for m in range(M):
            s[n*M + m] = X_tf[m, n] * np.exp(1j*2*np.pi*m*n/M)
    return s

def Wigner(M, N, r):
    # 维格纳变换: 时域 → TF域
    Y_tf = np.zeros((M, N), dtype=complex)
    for n in range(N):
        for m in range(M):
            Y_tf[m, n] = r[n*M + m] * np.exp(-1j*2*np.pi*m*n/M)
    return Y_tf

def generate_channel(num_paths=2, max_delay=3, max_doppler=0.1):
    """生成严格归一化的多径多普勒信道"""
    delays = np.random.randint(0, max_delay+1, size=2)  # 2条路径
    dopplers = np.array([0, max_doppler])  # 一条静态，一条动态

    # 关键修正：功率归一化
    gains = np.array([0.8, 0.2]) * np.exp(1j * np.random.uniform(0, 2*np.pi, 2))
    gains /= np.sqrt(np.sum(np.abs(gains)**2))

    return delays, dopplers, gains

def apply_channel(x, delays, dopplers, gains):
    """应用信道（修正多普勒和循环移位）"""
    y = np.zeros_like(x, dtype=complex)
    L = len(x)
    for d, nu, g in zip(delays, dopplers, gains):
        # 关键修正：规范化的多普勒相位
        phase = np.exp(1j * 2 * np.pi * nu * np.arange(L) / L)
        shifted = np.roll(x * phase, d)
        y += g * shifted
    return y

#%%
def otfs_sensing(M=128, N=64, EbN0dB=np.arange(0, 17, 2), N_frames=2000):
    c = 3e8
    fc = 30e9
    deltaf = 240e3  # 子载波间隔 240kHz
    T = 1/deltaf    # 脉冲成形滤波器周期
    lambda_ = c/fc
    # 目标信息
    targetRange = 35  # 目标距离35m
    targetSpeed = 30  # 目标速度30m/s

    delay = 2*targetRange/c
    kp = int(np.round(delay*M*deltaf))

    doppler = 2*targetSpeed/lambda_
    lp = int(doppler*N*T)
    # 生成相位旋转向量
    dd = np.arange(0, M*N)
    D = np.exp(1j * 2*np.pi/(M*N) * dd)

    """OTFS系统仿真 (修正功率和噪声计算)"""
    SER_sim = np.zeros_like(EbN0dB, dtype=float)
    # 创建QAM调制器
    QAM_mod = 4
    bps = int(np.log2(QAM_mod))

    EsN0dB = 10 * np.log10(bps) + EbN0dB
    MOD_TYPE = "qam"
    modem, Es, bps = modulator(MOD_TYPE, QAM_mod)
    map_table, demap_table = modem.getMappTable()

    ## 初始化误差数组
    errorRange_TF = np.zeros_like(EsN0dB, dtype=float)
    errorVelo_TF = np.zeros_like(EsN0dB, dtype=float)
    errorRange_MF = np.zeros_like(EsN0dB, dtype=float)
    errorVelo_MF = np.zeros_like(EsN0dB, dtype=float)

    for idx, snr in enumerate(EsN0dB):
        print(f"{idx+1}/{EsN0dB.size}")
        sigma2 = 10 ** (-snr / 10)  # 噪声方差
        errors = 0
        total_symbols = 0
        for _ in range(N_frames):
            # === 发射端 ===
            bits = np.random.randint(0, 2, size = M*N*bps).astype(np.int8)
            X_dd = modem.modulate(bits)
            sym_int = np.array([demap_table[sym] for sym in X_dd])
            X_dd1 = X_dd.reshape(M, N, order = 'F') #/ np.sqrt(Es)

            # OTFS调制
            X_tf = ISFFT(X_dd1)
            s = np.fft.ifft(X_tf, axis=0).T.flatten() # Heisenberg变换
            ## or
            # s = Heisenberg(M, N, X_tf)
            # === 信道 ===
            # 通过时延多普勒信道
            r = np.zeros_like(s, dtype=complex)
            # 修正为（确保正确循环移位）
            temp = s * (D**lp)
            r += np.exp(1j*2*np.pi*np.random.rand()) * np.roll(temp, kp)
            #### r = s
            # 添加AWGN噪声 (修正噪声功率)
            noise_power = np.mean(np.abs(r)**2) * sigma2
            # print(f"{idx} {noise_power}")
            noise = np.sqrt(noise_power/2) * (np.random.randn(*r.shape) + 1j*np.random.randn(*r.shape))
            r += noise

            # === 接收端 ===
            Y_tf = np.fft.fft(r.reshape(M, N, order = 'F'), axis=0) # Wigner变换
            ## or
            # Y_tf = Wigner(M, N, r)
            Y_dd = SFFT(Y_tf)

            ## Sensing Based on TF domain, 基于TF域的感知
            H_tf = Y_tf * np.conj(X_tf)
            rdm_tf = np.fft.fft(np.fft.ifft(H_tf, axis=0), n=10*N, axis=1).conj() * np.sqrt(M/N)
            MM = np.max(np.abs(rdm_tf))
            I1, I2 = np.where(np.abs(rdm_tf) == MM)
            rangeEst = (I1[0])/(M*deltaf)*c/2
            veloEst = (I2[0])/(N*10*T)*lambda_/2

            errorRange_TF[idx] += (rangeEst - targetRange)**2/N_frames
            errorVelo_TF[idx] += (veloEst - targetSpeed)**2/N_frames

            ## Sensing based on match filtering in DD domain, # 基于DD域匹配滤波的感知
            y_vec = Y_dd.T.flatten()
            h_vec = np.fft.ifft(np.fft.fft(X_dd).conj() * np.fft.fft(y_vec), )
            H_est = h_vec.reshape(M, N, order = 'F')

            MM = np.max(np.abs(H_est))
            I1, I2 = np.where(np.abs(H_est) == MM)
            rangeEst = (I1[0])/(M*deltaf)*c/2
            veloEst = (I2[0])/(N*T)*lambda_/2

            errorRange_MF[idx] += (rangeEst - targetRange)**2/N_frames
            errorVelo_MF[idx] += (veloEst - targetSpeed)**2/N_frames

            # === 解调与SER计算 ===
            # QPSK解调 (相位判决)
            Y_dd = Y_dd.T
            uu_hat = modem.demodulate(Y_dd.flatten(), 'hard')
            sym_int_hat = []
            for j in range(M*N):
                sym_int_hat.append( int(''.join([str(num) for num in uu_hat[j*bps:(j+1)*bps]]), base = 2) )
            sym_int_hat = np.array(sym_int_hat)

            errors += np.sum(sym_int != sym_int_hat)
            total_symbols += sym_int.size

        SER_sim[idx] = errors / total_symbols

    # 理论QPSK SER
    SER_theory = ser_awgn(EbN0dB, MOD_TYPE, QAM_mod)

    return  SER_sim, SER_theory, errorRange_TF, errorVelo_TF, errorRange_MF, errorVelo_MF

#%% 运行仿真
M = 1024
N = 120
N_frames = 100
EbN0dB = np.arange(-40, 12, 20)
SER_sim, SER_theory, errorRange_TF, errorVelo_TF, errorRange_MF, errorVelo_MF = otfs_sensing(M = M, N = N, EbN0dB=EbN0dB, N_frames=N_frames)

###>>>>>>>>>>
colors = plt.cm.jet(np.linspace(0, 1, 5))
fig, axs = plt.subplots(1, 1, figsize = (10, 8), constrained_layout = True)

axs.semilogy(EbN0dB, SER_sim, 'bo-', label='OTFS (仿真)')
axs.semilogy(EbN0dB, SER_theory, 'r--', label='QPSK (理论)')

axs.set_xlabel('Eb/N0 (dB)')
axs.set_ylabel('SER')
axs.legend()

plt.title('OTFS-QPSK 符号错误率性能')
plt.show()
plt.close('all')

###>>>>>>>>>>
errorRange_TF = np.sqrt(errorRange_TF);
errorVelo_TF = np.sqrt(errorVelo_TF);
errorRange_MF = np.sqrt(errorRange_MF);
errorVelo_MF = np.sqrt(errorVelo_MF);

colors = plt.cm.jet(np.linspace(0, 1, 5))
fig, axs = plt.subplots(1, 1, figsize = (10, 8), constrained_layout = True)
axs.semilogy(EbN0dB, errorRange_TF, 'bo-', label='TF Range')
axs.semilogy(EbN0dB, errorRange_MF, 'r--', label='DD Range')
axs.set_xlabel('Eb/N0 (dB)')
axs.set_ylabel('RMSE')
axs.legend()
plt.title('RMSE')
plt.show()
plt.close('all')

colors = plt.cm.jet(np.linspace(0, 1, 5))
fig, axs = plt.subplots(1, 1, figsize = (10, 8), constrained_layout = True)
axs.plot(EbN0dB, errorVelo_TF, 'bo-', label='TF Velo')
axs.plot(EbN0dB, errorVelo_MF, 'r--', label='DD Velo')
axs.set_xlabel('Eb/N0 (dB)')
axs.set_ylabel('RMSE')
axs.legend()

plt.title('RMSE')
plt.show()
plt.close('all')














