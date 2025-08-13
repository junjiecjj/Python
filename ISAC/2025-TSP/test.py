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
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.special import sinc
from scipy import signal  # 用于窗口函数

# 系统参数设置
N = 64           # 减少符号数以加快演示速度
L = 8            # 降低过采样率
alpha = 0.35     # 滚降因子
M = 50           # 减少相干积分次数
num_realizations = 50  # 减少蒙特卡洛仿真次数

# 生成16-QAM符号（简化版）
qam_symbols = np.array([-3-3j, -3-1j, -3+3j, -3+1j,
                        -1-3j, -1-1j, -1+3j, -1+1j,
                        1-3j, 1-1j, 1+3j, 1+1j,
                        3-3j, 3-1j, 3+3j, 3+1j])
qam_symbols = qam_symbols / np.sqrt(10)  # 手动归一化功率

# 手动实现RRC滤波器（不使用kaiser窗）
def rrc_filter(num_taps, alpha, L):
    t = np.linspace(-num_taps//2, num_taps//2, num_taps) / L
    h = np.zeros_like(t)

    for i, tt in enumerate(t):
        if tt == 0:
            h[i] = 1 - alpha + 4*alpha/np.pi
        elif abs(tt) == 1/(4*alpha):
            val = alpha/np.sqrt(2)
            h[i] = val * ((1+2/np.pi)*np.sin(np.pi/(4*alpha)) + (1-2/np.pi)*np.cos(np.pi/(4*alpha)))
        else:
            num = np.sin(np.pi*tt*(1-alpha)) + 4*alpha*tt*np.cos(np.pi*tt*(1+alpha))
            den = np.pi*tt*(1-(4*alpha*tt)**2)
            h[i] = num / den

    # 使用矩形窗替代kaiser窗
    return h / np.sqrt(np.sum(h**2))  # 能量归一化

num_taps = 6 * L  # 减少滤波器长度
rrc_taps = rrc_filter(num_taps, alpha, L)

# 计算ACF
rrc_acf = fftconvolve(rrc_taps, rrc_taps[::-1], mode='full')
rrc_acf_squared = np.abs(rrc_acf)**2
rrc_acf_squared = rrc_acf_squared / np.max(rrc_acf_squared)

# 初始化存储
E_Rk_squared = np.zeros(N * L)
E_Rk_integrated = np.zeros(N * L)

# 简化的蒙特卡洛仿真
for _ in range(num_realizations):
    s = np.random.choice(qam_symbols, N)
    x = np.fft.ifft(s) * np.sqrt(N)
    x_up = np.zeros(N * L, dtype=complex)
    x_up[::L] = x
    x_tilde = fftconvolve(x_up, rrc_taps, mode='same')
    R_k = np.fft.ifft(np.fft.fft(x_tilde) * np.conj(np.fft.fft(x_tilde)))
    R_k = np.fft.fftshift(R_k)
    E_Rk_squared += np.abs(R_k)**2
    if _ < M:
        E_Rk_integrated += np.abs(R_k)**2

E_Rk_squared /= num_realizations
E_Rk_integrated /= M

# 绘图
delay = np.arange(-N*L//2, N*L//2) / L
plt.figure(figsize=(10, 5))
plt.plot(delay, 10*np.log10(rrc_acf_squared), 'k--', label='Pulse ACF')
plt.plot(delay, 10*np.log10(E_Rk_squared), 'b-', label='Avg ACF')
plt.plot(delay, 10*np.log10(E_Rk_integrated), 'r-', label=f'Integrated (M={M})')
plt.xlim([-3, 3])
plt.ylim([-60, 5])
plt.xlabel('Delay (T)')
plt.ylabel('Power (dB)')
plt.title('ACF Comparison')
plt.legend()
plt.grid()
plt.show()
