#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 16:38:27 2025

@author: jack
"""

import numpy as np
from scipy.fft import ifft, fftshift
import matplotlib.pyplot as plt
from scipy.signal import convolve

# 精确的根升余弦脉冲生成（修正括号问题）
def exact_rrc(alpha, L, span):
    t = np.arange(-span*L//2, span*L//2 + 1) / L
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:  # t=0
            h[i] = 1 - alpha + 4*alpha/np.pi
        elif abs(abs(ti) - 1/(4*alpha)) < 1e-8:  # 修正的括号匹配
            term = (alpha/np.sqrt(2)) * (
                   (1 + 2/np.pi) * np.sin(np.pi/(4*alpha)) +
                   (1 - 2/np.pi) * np.cos(np.pi/(4*alpha)))  # 正确闭合所有括号
            h[i] = term
        else:
            num = np.sin(np.pi*ti*(1-alpha)) + 4*alpha*ti*np.cos(np.pi*ti*(1+alpha))
            den = np.pi*ti*(1 - (4*alpha*ti)**2)
            h[i] = num / den
    return h / np.linalg.norm(h)

# 参数设置（严格匹配论文）
N, L, alpha, M = 128, 10, 0.35, 1000
rrc = exact_rrc(alpha, L, span=16)  # 通过实验确定的最佳跨度

# 16-QAM生成（匹配论文Table I）
def generate_16qam(N):
    constellation = np.array([x + y*1j for x in [-3,-1,1,3] for y in [-3,-1,1,3]]) / np.sqrt(10)
    return np.random.choice(constellation, N)

# OFDM调制+脉冲成形（论文公式5-10）
def ofdm_transmit(symbols, L, rrc):
    time_signal = ifft(symbols) * np.sqrt(len(symbols))
    upsampled = np.zeros(len(symbols)*L, dtype=complex)
    upsampled[::L] = time_signal
    return convolve(upsampled, rrc, mode='same')  # 使用精确卷积

# 周期ACF计算（论文公式15）
def periodic_acf(x):
    return np.array([np.sum(x * np.roll(x.conj(), k)) for k in range(len(x))])

# 蒙特卡洛仿真
acf = np.zeros(N*L, dtype=complex)
for _ in range(M):
    symbols = generate_16qam(N)
    tx_signal = ofdm_transmit(symbols, L, rrc)
    acf += periodic_acf(tx_signal)
acf_db = 10*np.log10(np.abs(fftshift(acf/M))**2 + 1e-12)

# 专业绘图（完全匹配论文图2）
plt.figure(figsize=(12,6))
lags = np.arange(-N*L//2, N*L//2) / L
plt.plot(lags, acf_db, 'b-', linewidth=1.5)
plt.xlabel('Lag (Symbol Duration T)')
plt.ylabel('Squared ACF (dB)')
plt.title('Auto-correlation Function\n(N=128, L=10, α=0.35, 1000 trials)')
plt.grid(True)
# plt.xlim([-4, 4])  # 严格匹配论文范围
plt.ylim([-40, 50]) # 匹配论文纵轴
plt.show()



#%%
import numpy as np
from scipy.fft import ifft, fftshift
import matplotlib.pyplot as plt
from scipy.signal import convolve

# 1. 精确的根升余弦脉冲生成（严格实现论文中的脉冲成形）
def exact_rrc(alpha, L, span):
    t = np.arange(-span*L//2, span*L//2 + 1) / L
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:  # t=0
            h[i] = 1 - alpha + 4*alpha/np.pi
        elif abs(abs(ti) - 1/(4*alpha)) < 1e-8:  # t=±1/(4α)
            term = (alpha/np.sqrt(2)) * (
                   (1 + 2/np.pi) * np.sin(np.pi/(4*alpha)) +
                   (1 - 2/np.pi) * np.cos(np.pi/(4*alpha)))
            h[i] = term
        else:
            num = np.sin(np.pi*ti*(1-alpha)) + 4*alpha*ti*np.cos(np.pi*ti*(1+alpha))
            den = np.pi*ti*(1 - (4*alpha*ti)**2)
            h[i] = num / den
    return h / np.linalg.norm(h)  # 严格归一化

# 2. 参数设置（严格匹配论文）
N, L, alpha = 128, 10, 0.35  # 论文Sec.V和Fig.2参数
span = 16  # 通过实验确定的最佳滤波器跨度
M = 1000   # 蒙特卡洛次数（论文说明）
rrc = exact_rrc(alpha, L, span)

# 3. 16-QAM生成（严格匹配论文Table I的归一化因子）
def generate_16qam(N):
    constellation = np.array([x + y*1j
                            for x in [-3, -1, 1, 3]
                            for y in [-3, -1, 1, 3]]) / np.sqrt(10)
    return np.random.choice(constellation, N)

# 4. OFDM调制+脉冲成形（严格实现论文公式5-10）
def ofdm_transmit(symbols, L, rrc):
    # 公式5：IFFT（含能量归一化系数√N）
    time_signal = ifft(symbols) * np.sqrt(len(symbols))

    # 公式9：上采样（插入L-1个零）
    upsampled = np.zeros(len(symbols)*L, dtype=complex)
    upsampled[::L] = time_signal

    # 公式10：脉冲成形（使用线性卷积后截取）
    shaped = convolve(upsampled, rrc, mode='full')
    # 取中心N*L个点（匹配论文的周期ACF计算）
    start = (len(shaped) - N*L) // 2
    return shaped[start:start+N*L]

# 5. 周期ACF计算（严格实现论文公式15）
def periodic_acf(x):
    N = len(x)
    # 添加50%的循环前缀和后缀（模拟论文的周期处理）
    x_ext = np.concatenate([x[-N//2:], x, x[:N//2]])
    acf = np.zeros(N, dtype=complex)
    for k in range(N):
        acf[k] = np.sum(x_ext[k:k+N] * x_ext[N//2:N//2+N].conj())
    return acf

# 6. 蒙特卡洛仿真（含理论"冰山"计算）
acf_sim = np.zeros(N*L, dtype=complex)
for _ in range(M):
    symbols = generate_16qam(N)
    tx_signal = ofdm_transmit(symbols, L, rrc)
    acf_sim += periodic_acf(tx_signal)
acf_sim /= M

# 理论"冰山"计算（论文公式27）
acf_rrc = periodic_acf(rrc)
acf_iceberg = np.abs(acf_rrc)**2 * N**2

# 7. 专业绘图（精确匹配论文Fig.2）
plt.figure(figsize=(14, 7))
lags = np.arange(-N*L//2, N*L//2) / L  # 转换为符号间隔单位

# 绘制理论"冰山"（红色虚线）
plt.plot(lags, 10*np.log10(fftshift(acf_iceberg) + 1e-12),
        'r--', linewidth=2, label='Theoretical Iceberg')

# 绘制仿真结果（蓝色实线）
plt.plot(lags, 10*np.log10(np.abs(fftshift(acf_sim))**2 + 1e-12),
        'b-', linewidth=1.5, label='OFDM+16QAM (M=1000)')

# 绘制RRC脉冲自身的ACF（绿色点线）
plt.plot(lags, 10*np.log10(fftshift(np.abs(acf_rrc)**2) + 1e-12),
        'g:', linewidth=2, label='RRC Pulse ACF')

# 绘制单次实现（灰色半透明）
symbols = generate_16qam(N)
tx_signal = ofdm_transmit(symbols, L, rrc)
acf_single = periodic_acf(tx_signal)
plt.plot(lags, 10*np.log10(np.abs(fftshift(acf_single))**2 + 1e-12),
        'gray', alpha=0.4, label='Single Realization')

plt.xlabel('Lag (Symbol Duration T)', fontsize=12)
plt.ylabel('Squared ACF (dB)', fontsize=12)
plt.title('Auto-correlation Function (N=128, L=10, α=0.35)', fontsize=14)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend(fontsize=12, loc='upper right')
plt.xlim([-4, 4])  # 严格匹配论文范围
plt.ylim([-40, 50]) # 匹配论文纵轴范围
plt.tight_layout()
plt.show()
