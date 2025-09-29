#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 02:47:57 2025

@author: jack
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.special import erfc
import commpy
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

def ISFFT(X):
    """ISFFT变换"""
    M, N = X.shape
    return np.fft.ifft(np.fft.fft(X, axis=0), axis=1) * np.sqrt(N / M)

def SFFT(X):
    """SFFT变换"""
    M, N = X.shape
    return np.fft.fft(np.fft.ifft(X, axis=0), axis=1) * np.sqrt(M / N)

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

def generate_otfs_channel(M, N, max_delay=3, max_doppler=0.1):
    """生成OTFS多径多普勒信道 - 更实际的参数"""
    # 3条路径：直射+2条多径
    delays = np.array([0, 1, 2])  # 合理的时延
    dopplers = np.array([0.0, 0.05, -0.03])  # 适中的多普勒

    # 复增益（主路径强，多径弱）
    gains = np.array([0.7, 0.2, 0.1]) * np.exp(1j * np.random.uniform(0, 2*np.pi, 3))
    gains /= np.linalg.norm(gains)  # 功率归一化

    return delays, dopplers, gains
def construct_otfs_channel_matrix_simple(M, N, delays, dopplers, gains):
    """简化的OTFS信道矩阵构建 - 更稳定"""
    MN = M * N
    H = np.zeros((MN, MN), dtype=complex)

    for i, (delay, doppler, gain) in enumerate(zip(delays, dopplers, gains)):
        # 创建时延矩阵（循环移位）
        D_mat = np.eye(MN)
        D_mat = np.roll(D_mat, delay, axis=1)

        # 创建多普勒对角矩阵
        k = np.arange(MN)
        phase = np.exp(1j * 2 * np.pi * doppler * k / (M * N))
        P_mat = np.diag(phase)
        H += gain * P_mat @ D_mat
    return H

def otfs_mmse_detection_simple(Y_dd_vec, H_channel, noise_power):
    """简化但稳定的MMSE检测"""
    MN = len(Y_dd_vec)
    H_H = H_channel.conj().T

    # 直接求逆，添加小的正则化项
    covariance = H_channel @ H_H + (noise_power + 1e-10) * np.eye(MN)
    W_mmse = H_H @ np.linalg.inv(covariance)

    return W_mmse @ Y_dd_vec

def otfs_zf_detection_simple(Y_dd_vec, H_channel):
    """简化的ZF检测"""
    H_pinv = np.linalg.pinv(H_channel)
    return H_pinv @ Y_dd_vec

def otfs_mp_detection_improved(Y_dd_vec, H_channel, noise_power, max_iter=10):
    """改进的消息传递检测"""
    MN = len(Y_dd_vec)

    # 初始化
    x_est = np.zeros(MN, dtype=complex)
    residual = Y_dd_vec.copy()

    for iter in range(max_iter):
        # 计算梯度
        gradient = H_channel.conj().T @ residual

        # 更新估计
        x_est += 0.1 * gradient  # 学习率

        # 更新残差
        residual = Y_dd_vec - H_channel @ x_est

    return x_est
def otfs_simulation_corrected(M=32, N=16, EbN0dB=np.arange(0, 16, 2), N_frames=500):
    """修正的OTFS仿真"""
    SER_otfs = np.zeros_like(EbN0dB, dtype=float)
    SER_awgn = np.zeros_like(EbN0dB, dtype=float)
    # 创建QAM调制器
    QAM_mod = 4
    bps = int(np.log2(QAM_mod))

    EsN0dB = 10 * np.log10(bps) + EbN0dB
    MOD_TYPE = "qam"
    modem, Es, bps = modulator(MOD_TYPE, QAM_mod)
    map_table, demap_table = modem.getMappTable()

    for idx, snr_db in enumerate(EsN0dB):
        print(f"{idx+1}/{EsN0dB.size}")
        sigma2 = 10 ** (-snr_db / 10)  # 噪声方差
        errors_otfs = 0
        errors_awgn = 0
        total_symbols = 0

        for frame in range(N_frames):
            # ========== 发射端 ==========
            # QPSK符号（功率归一化）
            bits = np.random.randint(0, 2, size = M*N*bps).astype(np.int8)
            X_dd = modem.modulate(bits)
            d = np.array([demap_table[sym] for sym in X_dd])
            X_dd = X_dd.reshape(M, N,)
            # OTFS调制
            X_tf = ISFFT(X_dd)
            s = np.fft.ifft(X_tf, axis=0).flatten()

            # ========== 信道 ==========
            delays, dopplers, gains = generate_otfs_channel(M, N)
            H_channel = construct_otfs_channel_matrix_simple(M, N, delays, dopplers, gains)

            # 应用信道（矩阵乘法）
            r_otfs = H_channel @ s

            # AWGN信道（对比）
            r_awgn = s.copy()

            # 添加噪声
            noise_power_otfs = np.mean(np.abs(r_otfs)**2) * sigma2
            noise = np.sqrt(noise_power_otfs/2) * (np.random.randn(*r_otfs.shape) + 1j*np.random.randn(*r_otfs.shape))
            r_otfs += noise

            noise_power_awgn = np.mean(np.abs(r_awgn)**2) * sigma2
            noise = np.sqrt(noise_power_awgn/2) * (np.random.randn(*r_awgn.shape) + 1j*np.random.randn(*r_awgn.shape))
            r_awgn += noise

            # ========== 接收端 ==========
            # OTFS解调
            Y_tf_otfs = np.fft.fft(r_otfs.reshape(M, N), axis=0)
            Y_dd_otfs = SFFT(Y_tf_otfs)
            Y_dd_vec_otfs = Y_dd_otfs.flatten()
            # ========== 检测算法 ==========
            # 根据SNR选择检测算法
            if snr_db > 8:  # 高SNR用ZF
                symbols_est_otfs = otfs_zf_detection_simple(Y_dd_vec_otfs, H_channel)
            else:  # 低中SNR用MMSE
                symbols_est_otfs = otfs_mmse_detection_simple(Y_dd_vec_otfs, H_channel, noise_power_otfs)

            # AWGN解调
            Y_tf_awgn = np.fft.fft(r_awgn.reshape(M, N), axis=0)
            Y_dd_awgn = SFFT(Y_tf_awgn)
            symbols_est_awgn = Y_dd_awgn.flatten()

            # ========== 解调 ==========
            # QPSK解调 (相位判决)
            uu_hat_otfs = modem.demodulate(symbols_est_otfs, 'hard')
            d_hat_otfs = []
            for j in range(M*N):
                d_hat_otfs.append( int(''.join([str(num) for num in uu_hat_otfs[j*bps:(j+1)*bps]]), base = 2) )
            d_hat_otfs = np.array(d_hat_otfs)
            errors_otfs += np.sum(d != d_hat_otfs)

            uu_hat_awgn = modem.demodulate(symbols_est_awgn, 'hard')
            d_hat_awgn = []
            for j in range(M*N):
                d_hat_awgn.append( int(''.join([str(num) for num in uu_hat_awgn[j*bps:(j+1)*bps]]), base = 2) )
            d_hat_awgn = np.array(d_hat_awgn)
            errors_awgn += np.sum(d != d_hat_awgn)

            total_symbols += d.size
        SER_otfs[idx] = errors_otfs / total_symbols
        SER_awgn[idx] = errors_awgn / total_symbols

        print(f"  OTFS SER: {SER_otfs[idx]:.6f}, AWGN SER: {SER_awgn[idx]:.6f}")

    # 理论QPSK SER
    SER_theory = ser_awgn(EbN0dB, MOD_TYPE, QAM_mod)

    return EbN0dB, SER_otfs, SER_awgn, SER_theory

# 运行仿真
M, N = 32, 16
EbN0dB = np.arange(0, 12, 2)
EbN0dB, SER_otfs, SER_awgn, SER_theory = otfs_simulation_corrected( M=M, N=N, EbN0dB=EbN0dB, N_frames = 100 )


colors = plt.cm.jet(np.linspace(0, 1, 5))
fig, axs = plt.subplots(1, 1, figsize = (10, 8), constrained_layout = True)

axs.semilogy(EbN0dB, SER_otfs, 'bo-', linewidth=1, markersize=10, label=f'OTFS (Multipath-Doppler, M={M}, N={N})')
axs.semilogy(EbN0dB, SER_awgn, 'gs-', linewidth=3, markersize=10, label='OTFS (AWGN Channel)')
axs.semilogy(EbN0dB, SER_theory, 'r--', linewidth=2, label='Theoretical QPSK (AWGN)')

axs.set_xlabel('Eb/N0 (dB)')
axs.set_ylabel('SER')
axs.legend()

plt.title('OTFS Performance with Advanced Detection in Multipath-Doppler Channel', fontsize=16, pad=20)
plt.show()
plt.close('all')

