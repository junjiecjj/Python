#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 23:46:10 2025

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

### 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2          # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6         # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 18
np.random.seed(42)

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

def otfs_SER(M=128, N=64, EbN0dB=np.arange(0, 17, 2), N_frames=2000):
    """OTFS系统仿真 (修正功率和噪声计算)"""
    SER_sim = np.zeros_like(EbN0dB, dtype=float)
    # 创建QAM调制器
    QAM_mod = 4
    bps = int(np.log2(QAM_mod))

    EsN0dB = 10 * np.log10(bps) + EbN0dB
    MOD_TYPE = "qam"
    modem, Es, bps = modulator(MOD_TYPE, QAM_mod)
    map_table, demap_table = modem.getMappTable()

    for idx, snr in enumerate(EsN0dB):
        print(f"{idx+1}/{EsN0dB.size}")
        sigma2 = 10 ** (-snr / 10)  # 噪声方差
        errors = 0
        total_symbols = 0

        for _ in range(N_frames):
            # === 发射端 ===
            bits = np.random.randint(0, 2, size = M*N*bps).astype(np.int8)
            X_dd = modem.modulate(bits)
            d = np.array([demap_table[sym] for sym in X_dd])
            # X_dd = X_dd.reshape(M, N) #/ np.sqrt(Es)
            # # # OTFS调制
            # X_tf = ISFFT(X_dd)
            # # s = np.fft.ifft(X_tf, axis=0).T.flatten()      # Heisenberg变换, (1)
            # s = np.fft.ifft(X_tf, axis=0).flatten()        # Heisenberg变换, (2)
            # # s = Heisenberg(M, N, X_tf)                       # Heisenberg变换, (3)

            X_dd = X_dd.reshape(M, N, order = 'F') #/ np.sqrt(Es)
            X_tf = ISFFT(X_dd)
            s = np.fft.ifft(X_tf, axis=0).T.flatten()
            # === 信道 ===
            # delays, dopplers, gains = generate_channel( max_delay=2, max_doppler=0.05)
            # r = apply_channel(s, delays, dopplers, gains)
            ##### 多普勒信道 (单径)
            nu = 0.02  # 归一化多普勒
            dop = np.exp(1j * 2 * np.pi * nu * np.arange(len(s)) / len(s))
            r = s * dop  # 忽略时延，仅测试多普勒
            # r = s

            # 添加噪声 (修正噪声功率)
            noise_power = np.mean(np.abs(r)**2) * sigma2
            noise = np.sqrt(noise_power/2) * (np.random.randn(*r.shape) + 1j*np.random.randn(*r.shape))
            r += noise

            # === 接收端 ===
            # # Y_tf = np.fft.fft(r.reshape(N, M).T, axis=0)   # Wigner变换, 对应(1)
            # Y_tf = np.fft.fft(r.reshape(M, N), axis=0)     # Wigner变换, 对应(2)
            # # Y_tf = Wigner(M, N, r)                           # Wigner变换, 对应(3)
            # Y_dd = SFFT(Y_tf)

            Y_tf = np.fft.fft(r.reshape(M, N, order = 'F'), axis=0)
            Y_dd = SFFT(Y_tf)
            Y_dd = Y_dd.T

            # === 解调与SER计算 ===
            # QPSK解调 (相位判决)
            uu_hat = modem.demodulate(Y_dd.flatten(), 'hard')
            d_hat = []
            for j in range(M*N):
                d_hat.append( int(''.join([str(num) for num in uu_hat[j*bps:(j+1)*bps]]), base = 2) )
            d_hat = np.array(d_hat)

            errors += np.sum(d != d_hat)
            total_symbols += d.size

        SER_sim[idx] = errors / total_symbols

    # 理论QPSK SER
    SER_theory = ser_awgn(EbN0dB, MOD_TYPE, QAM_mod)

    return  SER_sim, SER_theory

# 运行仿真
M = 64
N = 32
N_frames = 1000
EbN0dB = np.arange(0, 12, 2)
SER_sim, SER_theory = otfs_SER(M = M, N = N, EbN0dB=EbN0dB, N_frames=N_frames)

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










