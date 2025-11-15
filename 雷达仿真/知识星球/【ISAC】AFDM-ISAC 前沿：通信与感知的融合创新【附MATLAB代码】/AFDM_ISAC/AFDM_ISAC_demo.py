#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 21:55:50 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# from scipy.signal import awgn
from DigiCommPy.modem import PSKModem, QAMModem, PAMModem, FSKModem
from DigiCommPy.channels import awgn
from DigiCommPy.errorRates import ser_awgn

# 设置随机种子
np.random.seed(42)

# ===================== 本地函数定义 =====================
def idaft_col(x, c1, c2):
    """单列 IDAFT"""
    N = len(x)
    n = np.arange(N).reshape(-1, 1)
    m = np.arange(N).reshape(1, -1)

    E1 = np.exp(1j * 2 * np.pi * c1 * (n**2))           # 时间域 chirp
    E2 = np.exp(1j * 2 * np.pi * c2 * (m**2))           # DAFT 域 chirp
    s = E1.flatten() * (np.fft.ifft(E2 * x) * np.sqrt(N))
    return s

def afdm_mod(X, c1, c2):
    """AFDM调制 - 逐列 IDAFT"""
    N, K = X.shape
    s_blocks = np.zeros((N, K), dtype=complex)
    for k in range(K):
        s_blocks[:, k] = idaft_col(X[:, k], c1, c2)
    return s_blocks


def daft_colwise(S, c1, c2):
    """逐列 DAFT（IDAFT 的逆）"""
    N, K = S.shape
    n = np.arange(N).reshape(-1, 1)  # 列向量 (N, 1)
    m = np.arange(N).reshape(-1, 1)  # 列向量 (N, 1)

    # E1: 形状 (N, 1)
    E1 = np.exp(-1j * 2 * np.pi * c1 * (n**2))

    # E2: 形状 (N, 1)
    E2 = np.exp(-1j * 2 * np.pi * c2 * (m**2))

    # E1 * S: 广播 E1 到 (N, K)
    temp1 = E1 * S

    # FFT 然后除以 sqrt(N)
    temp2 = np.fft.fft(temp1, axis=0) / np.sqrt(N)

    # E2 * temp2: 广播 E2 到 (N, K)
    Y = E2 * temp2

    return Y


def cpp_add(s_blocks, Ncp):
    """CPP 添加：每块头部拼接 Ncp 个尾样点"""
    N, K = s_blocks.shape
    out = np.zeros((N + Ncp, K), dtype=complex)
    for k in range(K):
        blk = s_blocks[:, k]
        out[:, k] = np.concatenate([blk[-Ncp:], blk])
    # 按列优先顺序展平，与MATLAB一致
    return out.flatten('F')

def cpp_remove(rx, N, Ncp, K):
    """去 CPP：按块切分并去掉前 Ncp"""
    # 按列优先顺序重塑，与MATLAB一致
    rx_mat = rx.reshape(N + Ncp, K, order='F')
    S_blocks = rx_mat[Ncp:, :]
    return S_blocks

def apply_radar_channel(s_tx, N, Ncp, K, li, fi, h_i):
    """雷达信道模拟：r[n] = Σ_i h_i · s[n-li] · e^{j2π f_i n}"""
    P = len(li)
    L = (N + Ncp) * K
    n = np.arange(L).reshape(-1, 1)
    r = np.zeros(L, dtype=complex)
    # 按列优先顺序重塑，与MATLAB一致
    s_mat = s_tx.reshape(N + Ncp, K, order='F')

    for p in range(P):
        s_shift = np.zeros_like(s_mat, dtype=complex)
        for k in range(K):
            s_shift[:, k] = np.roll(s_mat[:, k], li[p])
        # 按列优先顺序展平，与MATLAB一致
        r += h_i[p] * (s_shift.flatten('F') * np.exp(1j * 2 * np.pi * fi[p] * n).flatten())
    return r

def rdm_fccr(R, S):
    """时域 FCCR：fast-time FFT → 乘积 → IFFT → slow-time FFT"""
    Rf = np.fft.fft(R, axis=0)
    Sf = np.fft.fft(S, axis=0)
    Z = np.fft.ifft(Rf * np.conj(Sf), axis=0)
    RD = np.fft.fft(Z, axis=1)
    return RD

def mag2db_norm(Y):
    """归一幅度 → dB（避免 -Inf）"""
    Yn = Y / (np.max(Y) + np.finfo(float).eps)
    return 20 * np.log10(Yn + np.finfo(float).eps)

def plot_constellation(X, Xhat, M):
    """星座图（仅演示）"""
    Nplot = min(4000, len(X))
    idx = np.random.permutation(len(X))[:Nplot]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 发射端星座图
    ax1.plot(np.real(X[idx]), np.imag(X[idx]), 'bo', markersize=2, alpha=0.6)
    ax1.set_aspect('equal', 'box')
    ax1.grid(True)
    ax1.set_title(f'TX {M}-QAM')
    ax1.set_xlabel('I')
    ax1.set_ylabel('Q')

    # 接收端星座图
    ax2.plot(np.real(Xhat[idx]), np.imag(Xhat[idx]), 'ro', markersize=2, alpha=0.6)
    ax2.set_aspect('equal', 'box')
    ax2.grid(True)
    ax2.set_title('RX Equalized')
    ax2.set_xlabel('I')
    ax2.set_ylabel('Q')

    plt.tight_layout()
    plt.show()

# ===================== 系统参数 =====================
N = 1024                     # AFDM 子载波数
Nsym = 64                    # 慢时脉冲数
M = 16                       # QAM 阶数
SNRdB = 15                   # AWGN

fc = 24e9                    # 载频 [Hz]
DeltaF = 22.729e3            # 子载波间隔 [Hz]
Fs = N * DeltaF              # 采样率 [Hz]
Ts = 1 / Fs                  # 采样间隔 [s]

alphamax = 2
kv = 4
c1 = (2 * (alphamax + kv) + 1) / (2 * N)
c2 = 1 / (128 * N)
Ncp = 128                    # CPP（需 > 最大时延采样）
TAFDM = (N + Ncp) / Fs       # 单符号时长（含CPP）

c_light = 3e8

# ===================== 主程序 =====================

# 使用您的调制解调代码
MOD_TYPE = "qam"
coherence = 'coherent'
modem_dict = {'psk': PSKModem, 'qam': QAMModem, 'pam': PAMModem, 'fsk': FSKModem}

if MOD_TYPE.lower() == 'fsk':
    modem = modem_dict[MOD_TYPE.lower()](M, coherence)
else:
    modem = modem_dict[MOD_TYPE.lower()](M)


# ===================== 发射端：调制 & AFDM 调制 =====================
d = np.random.randint(low=0, high=M, size=N*Nsym)
X_vec = modem.modulate(d)                    # 使用您的调制器
# 按列优先顺序重塑，与MATLAB一致
X = X_vec.reshape(N, Nsym, order='F')       # DAFT 域符号

s_blocks = afdm_mod(X, c1, c2)              # N x Nsym（时域）
s_tx = cpp_add(s_blocks, Ncp)               # (N+Ncp)*Nsym x 1

# ===================== 目标真值（请确保 max(li)<Ncp） =====================
R_true = np.array([400, 320, 180])          # m
v_true = np.array([25, 10, -10])            # m/s
P = len(R_true)

tau = 2 * R_true / c_light                  # s
li = np.round(tau / Ts).astype(int)         # 样点
fd = 2 * v_true * fc / c_light              # Hz
fi = fd * Ts                                # 归一化（乘以 n 的相位增量）

h_i = (np.random.randn(P) + 1j * np.random.randn(P)) / np.sqrt(2 * P)

# ===================== 接收：叠加回波 + 噪声 =====================
r_rx = apply_radar_channel(s_tx, N, Ncp, Nsym, li, fi, h_i)
r_rx = awgn(r_rx, SNRdB,)

# ===================== 通信侧接收：DAFT & 简易等化 =====================
Y_blocks = cpp_remove(r_rx, N, Ncp, Nsym)   # N x Nsym（时域）
Y_daft = daft_colwise(Y_blocks, c1, c2)     # N x Nsym（DAFT）

Hhat = Y_daft / (X + 1e-12)                 # 粗糙 LS，默认通信信息完全已知
Xhat = Y_daft / (Hhat + 1e-12)

if MOD_TYPE.lower()=='fsk': # demodulate (Refer Chapter 3)
    dCap = modem.demodulate(Xhat.flatten(order='F'), coherence)
else: #demodulate (Refer Chapter 3)
    dCap = modem.demodulate(Xhat.flatten(order='F'))

plot_constellation(X.flatten('F'), Xhat.flatten('F'), M)

# ===================== 公共坐标换算（物理轴） =====================
# 距离分辨率：ΔR = c/(2B) = c/(2*N*Δf) = c/(2*Fs)
range_bin_m = c_light / (2 * Fs)

# Doppler 轴：K 点慢时 FFT，Δfd = 1/(K*TAFDM)，范围 ±1/(2*TAFDM)
fd_axis = (np.arange(Nsym) - np.floor(Nsym/2)) * (1/(Nsym * TAFDM))  # Hz
vel_axis = fd_axis * (c_light / (2 * fc))                           # m/s

# ===================== 感知 ：时域 FCCR =====================
RD_fccr = np.abs(rdm_fccr(Y_blocks, s_blocks))     # N x Nsym
RD_show = np.fft.fftshift(RD_fccr[:Ncp, :], axes=1)  # 仅对 Doppler 维中心化
ranges_plot = np.arange(Ncp) * range_bin_m

# 2D RDM
plt.figure(figsize=(10, 6))
plt.imshow(mag2db_norm(RD_show), extent=[vel_axis[0], vel_axis[-1], ranges_plot[-1], ranges_plot[0]], aspect='auto', cmap='turbo')
plt.colorbar()
plt.xlabel('Velocity (m/s)')
plt.ylabel('Range (m)')
plt.title('Time-Domain FCCR (Range–Doppler)')

# 叠加真实目标
scatter = plt.scatter(v_true, R_true, s=80, c='r', marker='o', edgecolors='k', linewidths=1.2)
plt.legend([scatter], ['True targets'], loc='upper center', bbox_to_anchor=(0.5, -0.15))
plt.grid(True)
plt.box(True)
plt.show()

# 3D RDM
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

VV, RR = np.meshgrid(vel_axis, ranges_plot)
Zfccr = mag2db_norm(RD_show)

surf = ax.plot_surface(VV, RR, Zfccr, rstride = 1, cstride = 1,  cmap='hsv', )
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20)

ax.set_proj_type('ortho')

ax.view_init(azim=-120, elev=30)
ax.set_xlabel('Velocity (m/s)')
ax.set_ylabel('Range (m)')
ax.set_zlabel('Magnitude (dB)')
ax.set_title('3D Range–Doppler (FCCR)')
ax.grid(True)

# 3D 叠加真值（Z 取最近网格值）
kidxT = [np.argmin(np.abs(vel_axis - v)) for v in v_true]
ridxT = np.minimum(np.maximum(np.round(R_true / range_bin_m).astype(int), 0), Ncp-1)
zT = [Zfccr[ridxT[i], kidxT[i]] for i in range(P)]

ax.scatter3D(v_true, R_true, zT, s = 100, c='b', marker='o',  edgecolors='k', linewidths=1.2)

plt.show()
plt.close()
print('仿真完成。')



























