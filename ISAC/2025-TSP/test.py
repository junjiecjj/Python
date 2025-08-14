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
from scipy.linalg import circulant
from scipy.signal import firwin
import matplotlib.pyplot as plt

def rrc_pulse(L, alpha, span):
    """生成根升余弦脉冲"""
    t = np.arange(-span*L//2, span*L//2 + 1) / L
    p = np.zeros_like(t)
    for i, tt in enumerate(t):
        if abs(tt) < 1e-8:  # t=0
            p[i] = 1.0 - alpha + 4*alpha/np.pi
        elif abs(abs(tt) - 1/(4*alpha)) < 1e-8:
            p[i] = (alpha/np.sqrt(2)) * ((1+2/np.pi)*np.sin(np.pi/(4*alpha)) +
                   (1-2/np.pi)*np.cos(np.pi/(4*alpha)))
        else:
            p[i] = (np.sin(np.pi*tt*(1-alpha)) + 4*alpha*tt*np.cos(np.pi*tt*(1+alpha))) / \
                   (np.pi*tt*(1-(4*alpha*tt)**2))
    return p / np.sqrt(np.sum(p**2))  # 能量归一化

def discrete_pulse_shaping(x, p, L):
    """离散脉冲成型，确保维度匹配"""
    # 上采样
    x_up = np.zeros(len(x) * L, dtype=complex)
    x_up[::L] = x

    # 创建合适大小的循环矩阵
    N = len(x_up)
    p_pad = np.concatenate([p, np.zeros(N - len(p))])
    P = circulant(np.roll(p_pad, len(p)//2))[:N, :N]

    return P @ x_up

# 系统参数
N = 64  # 符号数
L = 8   # 过采样率
alpha = 0.35  # 滚降因子
span = 6      # 脉冲跨度

# 1. 生成调制符号
F = np.fft.fft(np.eye(N)) / np.sqrt(N)  # 归一化DFT矩阵
U = F.conj().T  # OFDM调制基

# 生成随机QAM符号
qam_symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j, 3+3j, 3-3j, -3+3j, -3-3j]) / np.sqrt(10)
s = np.random.choice(qam_symbols, N)

# 调制到时域
x = U @ s

# 2. 脉冲成型
p = rrc_pulse(L, alpha, span)
x_tilde = discrete_pulse_shaping(x, p, L)

# 3. 计算ACF
def compute_acf(signal):
    N = len(signal)
    acf = np.fft.ifft(np.abs(np.fft.fft(signal))**2)
    return np.fft.fftshift(acf)

# 绘制结果
plt.figure(figsize=(12, 8))

plt.subplot(311)
plt.title("Modulated Symbols (Time Domain)")
plt.plot(np.real(x), 'b-', label="Real")
plt.plot(np.imag(x), 'r-', label="Imag")
plt.legend()

plt.subplot(312)
plt.title("Pulse-shaped Signal")
plt.plot(np.real(x_tilde), 'b-', label="Real")
plt.plot(np.imag(x_tilde), 'r-', label="Imag")
plt.legend()

plt.subplot(313)
plt.title("Autocorrelation Function")
acf = compute_acf(x_tilde)
delay = np.arange(-len(acf)//2, len(acf)//2) / L
plt.plot(delay, 10*np.log10(np.abs(acf) + 1e-10))
plt.xlabel("Delay (T)")
plt.ylabel("Power (dB)")
plt.xlim([-5, 5])
# plt.ylim([-50, 10])
plt.grid()

plt.tight_layout()
plt.show()



#%% Section.II-B最后一段“Towards that end, we consider ISAC signaling with CP, which corre- spond to the periodic convolution processing of the MF at the sensing Rx. Without loss of generality, the CP is assumed to be larger than the maximum delay of the communication paths and sensing targets.”什么意思，能给出代码吗
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

def add_cp(x, cp_len):
    """添加循环前缀"""
    return np.concatenate([x[-cp_len:], x])

def remove_cp(x, cp_len):
    """去除循环前缀"""
    return x[cp_len:]

def isac_transceiver_with_cp():
    # 系统参数
    N = 128                 # OFDM符号数
    cp_len = 32             # CP长度(必须大于最大延迟)
    max_delay = 20          # 最大延迟

    # 1. 生成OFDM信号
    s = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)  # 随机QAM符号
    x = ifft(s) * np.sqrt(N)  # IFFT调制

    # 2. 添加CP (发射端)
    x_cp = add_cp(x, cp_len)

    # 3. 模拟信道效应(更明显的多径和目标反射)
    h = np.zeros(N + cp_len, dtype=complex)
    h[0] = 1.0                      # 直射路径 (最强)
    h[8] = 0.6 * np.exp(1j*np.pi/4) # 强多径
    h[15] = 0.4 * np.exp(1j*np.pi/3) # 目标反射1
    h[20] = 0.3 * np.exp(1j*np.pi/6) # 目标反射2

    # 通过信道 (线性卷积)
    y = np.convolve(x_cp, h, mode='same')[:len(x_cp)]
    y += 0.05 * (np.random.randn(len(y)) + 1j*np.random.randn(len(y)))  # 添加噪声

    # 4. 去除CP (接收端)
    y_no_cp = remove_cp(y, cp_len)

    # 5. 匹配滤波处理(圆卷积)
    R = fft(y_no_cp) * np.conj(fft(x))
    r = ifft(R)

    # 6. 结果可视化
    delay = np.arange(-N//2, N//2)
    plt.figure(figsize=(12, 6))

    # 绘制幅度(dB)
    plt.plot(delay, 20*np.log10(np.fft.fftshift(np.abs(r)) + 1e-10),
             'b-', linewidth=1.5, label='MF Output')

    # 标记关键延迟点
    delays = [0, 8, 15, 20]
    colors = ['r', 'g', 'm', 'c']
    labels = ['Direct Path', 'Multipath', 'Target 1', 'Target 2']

    for d, c, lbl in zip(delays, colors, labels):
        plt.axvline(x=d-N//2, color=c, linestyle='--', alpha=0.7)
        plt.text(d-N//2+1, -10, lbl, color=c, fontsize=10)

    plt.xlabel('Delay (samples)', fontsize=12)
    plt.ylabel('Correlation Amplitude (dB)', fontsize=12)
    plt.title('Matched Filter Output with CP Processing\n(Clearly Showing Multipath and Targets)', fontsize=14)
    plt.xlim([-N//2, N//2])
    # plt.ylim([-40, 20])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

isac_transceiver_with_cp()





