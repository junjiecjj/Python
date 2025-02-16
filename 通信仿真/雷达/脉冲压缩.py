#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:03:56 2022

@author: jack

https://blog.csdn.net/innovationy/article/details/121572508?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-121572508-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-121572508-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=6

https://blog.csdn.net/jiangwenqixd/article/details/109521694?spm=1001.2101.3001.6650.5&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-5-109521694-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-5-109521694-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=10

https://blog.csdn.net/RICEresearchNOTE/article/details/140855697

https://blog.51cto.com/u_16213651/8904378

https://blog.csdn.net/qq_44648285/article/details/143471871

https://zhuanlan.zhihu.com/p/692354746

https://blog.csdn.net/qq_43485394/article/details/122655901

https://blog.51cto.com/u_16213651/8904378



"""


import numpy as np
import matplotlib.pyplot as plt

# 参数设置
A = 1.0  # 信号幅度
fs = 1000  # 采样频率
T = 1  # 信号持续时间
t = np.linspace(0, T, fs * T, endpoint=False)

# 生成长脉冲信号
pulse_duration = 0.2  # 脉冲持续时间
pulse = np.zeros_like(t)
pulse[int(0.5 * fs):int((0.5 + pulse_duration) * fs)] = A

# 添加杂波
noise = np.random.normal(0, 0.1, pulse.shape)
received_signal = pulse + noise

# 匹配滤波器设计
matched_filter = np.flip(pulse)

# 进行匹配滤波
compressed_signal = np.convolve(received_signal, matched_filter, mode='same')

# 绘图
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, received_signal)
plt.title('Received Signal with Noise')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, compressed_signal)
plt.title('Compressed Signal After Matched Filtering')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
fs = 100e6       # 采样频率 100 MHz
T = 10e-6        # 脉冲宽度 10 μs
B = 30e6         # 带宽 30 MHz
f0 = 1e6         # 起始频率 1 MHz
SNR_dB = 10      # 信噪比 (dB)
delay = 5e-6     # 目标时延 5 μs

# 生成线性调频信号 (LFM)
t = np.arange(0, T, 1/fs)                  # 时间向量
N = len(t)
chirp_signal = np.exp(1j * np.pi * (B/T) * t**2 + 1j * 2 * np.pi * f0 * t)  # 复数LFM信号

# 模拟回波信号（添加时延和多普勒频移）
delay_samples = int(delay * fs)             # 时延对应的采样点数
echo_signal = np.zeros(N + delay_samples, dtype=complex)
echo_signal[delay_samples:delay_samples + N] = chirp_signal  # 添加时延

# 添加高斯白噪声
noise_power = 10**(-SNR_dB/10) * np.mean(np.abs(echo_signal)**2)
noise = np.sqrt(noise_power/2) * (np.random.randn(len(echo_signal)) + 1j * np.random.randn(len(echo_signal)))
echo_signal += noise

# 脉冲压缩（匹配滤波）
matched_filter = np.conj(chirp_signal[::-1])  # 匹配滤波器：发射信号的共轭反转
compressed_signal = np.convolve(echo_signal, matched_filter, mode='valid')

# 结果可视化
plt.figure(figsize=(12, 8))

# 发射信号（实部）
plt.subplot(3, 1, 1)
plt.plot(t*1e6, np.real(chirp_signal))
plt.title("Transmitted LFM Signal (Real Part)")
plt.xlabel("Time (μs)")
plt.ylabel("Amplitude")

# 接收回波信号（实部）
t_echo = np.arange(len(echo_signal)) / fs * 1e6
plt.subplot(3, 1, 2)
plt.plot(t_echo, np.real(echo_signal))
plt.title("Received Echo Signal with Noise (Real Part)")
plt.xlabel("Time (μs)")
plt.ylabel("Amplitude")

# 脉冲压缩结果（幅度）
t_compressed = np.arange(len(compressed_signal)) / fs * 1e6
plt.subplot(3, 1, 3)
plt.plot(t_compressed, 20 * np.log10(np.abs(compressed_signal)))
plt.title("Pulse Compression Output (dB)")
plt.xlabel("Time (μs)")
plt.ylabel("Amplitude (dB)")
plt.tight_layout()
plt.show()













