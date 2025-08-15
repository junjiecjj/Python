#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:23:08 2025

@author: jack
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#%%  https://blog.csdn.net/flex_tan/article/details/54884251
phase_offset = 0.0 # 载波相位补偿
freq_offset = 0.3  # 载波频率补偿
wn = 0.01          # PLL 带宽
zeta = 0.707       # pll damping factor
K = 1000           # pll loop gain
n = 500            # number of samples
## generate loop filter parameters
t1 = K/(wn*wn)     # tau1
t2 = 2*zeta / wn   # tau2

## feed-forward cofficients (numerator)
b0 = (4 * K / t1)*(1 + t2/2.0)
b1 = (8*K/t1)
b2 = (4 * K /t1) * (1. - t2/2.0)

## feed-forward cofficients (denominator)
a1 = -2.0
a2 = 1.0

## filter buffer
v0 = 0.0
v1 = 0.0
v2 = 0.0

## initialize states
phi = phase_offset # 输入信号初始相位
phi_hat = 0.0      # PLL 初始相位

delta_phi = np.zeros(n)
for i in range(n):
    # 计算输入波形及更新相位
    x = np.exp(1j * phi)
    phi += freq_offset

    # 根据相位估计计算PLL输出
    y = np.exp(1j * phi_hat)

    # 计算误差估计
    delta_phi[i] = np.angle(x * np.conjugate(y))

    # 更新缓存
    v2 = v1 # shift center register to upper register
    v1 = v0 # shift lower register to center register

    # compute new lower register
    v0 = delta_phi[i] - v1 * a1 - v2 * a2

    # 计算新的相位
    phi_hat = v0 * b0 + v1 * b1 + v2 * b2
    phi_hat = np.mod(phi_hat, 2*np.pi)

plt.plot(delta_phi)

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# 参数设置
fs = 44100                  # 采样率 (Hz)
duration = 0.02             # 信号时长 (秒)
f_carrier = 1000            # 载波频率 (Hz)
f_mod = 200                 # 调制频率 (Hz)
beta = 2                    # 调制指数
A = 1.0                     # 信号幅度

# 锁相环参数
K_p = 0.5                   # 鉴相器增益
K_i = 0.05                  # 积分增益
K_0 = 0.3                   # NCO增益

# 生成时间轴
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# 1. 生成FM调制信号
modulating_signal = A * np.sin(2 * np.pi * f_mod * t)
fm_signal = A * np.cos(2 * np.pi * f_carrier * t +
                       beta * np.sin(2 * np.pi * f_mod * t))

# 2. 修正后的PLL实现
class PLL:
    def __init__(self, fs, K_p, K_i, K_0, freq_estimate):
        self.fs = fs
        self.K_p = K_p
        self.K_i = K_i
        self.K_0 = K_0
        self.vco_phase = 0
        self.freq_estimate = freq_estimate
        self.integrator = 0
        self.t_step = 1/fs

    def update(self, input_signal):
        # VCO输出
        vco_out = np.cos(2 * np.pi * self.freq_estimate * self.t_step + self.vco_phase)

        # 鉴相器
        error = input_signal * vco_out

        # 环路滤波器
        self.integrator += self.K_i * error
        filtered_error = self.K_p * error + self.integrator

        # 更新VCO
        self.freq_estimate += self.K_0 * filtered_error
        self.vco_phase += 2 * np.pi * self.freq_estimate * self.t_step
        self.vco_phase %= (2 * np.pi)

        return filtered_error

# 初始化PLL
pll = PLL(fs, K_p, K_i, K_0, f_carrier)

# 解调过程
demodulated_signal = np.zeros_like(t)
for i in range(len(t)):
    demodulated_signal[i] = pll.update(fm_signal[i])

# 3. 后处理
demodulated_signal -= np.mean(demodulated_signal)  # 去直流

# 低通滤波
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return lfilter(b, a, data)

filtered_output = lowpass_filter(demodulated_signal, 1.5*f_mod, fs)

# 4. 结果可视化
plt.figure(figsize=(12, 8))

# 图1: 载波调制信号
plt.subplot(3, 1, 1)
plt.plot(t, fm_signal, color='blue', alpha=0.7)
plt.title('FM Modulated Carrier Signal (Transmitted)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.xlim(0, 0.01)

# 图2: 解调对比
plt.subplot(3, 1, 2)
plt.plot(t, modulating_signal, label='Original Signal', color='green')
plt.plot(t, filtered_output, label='Demodulated', color='red', linestyle='--')
plt.title('Demodulation Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.xlim(0, 0.01)

# 图3: 解调过程信号
plt.subplot(3, 1, 3)
plt.plot(t, demodulated_signal, color='purple')
plt.title('PLL Error Signal (Before Filtering)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.xlim(0, 0.01)

plt.tight_layout()
plt.show()


#%%



#%%



#%%



#%%































