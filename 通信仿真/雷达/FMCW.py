#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:03:56 2022

@author: jack

https://blog.csdn.net/innovationy/article/details/121572508?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-121572508-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-121572508-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=6

https://blog.csdn.net/jiangwenqixd/article/details/109521694?spm=1001.2101.3001.6650.5&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-5-109521694-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-5-109521694-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=10


https://mp.weixin.qq.com/s?__biz=MzkxMTMwMTg4Mg==&mid=2247491272&idx=1&sn=8c816033438a549fdaeb20e51b154896&chksm=c11f135df6689a4bb0528639e9f437c86e941ef816e78f2b9a935f568db8be529f85234cafd5&token=134337482&lang=zh_CN#rd


https://mp.weixin.qq.com/s?__biz=MzkxMTMwMTg4Mg==&mid=2247484183&idx=1&sn=fbf605f11d510343c5beda9bdc5c32a4&chksm=c11f0e82f66887941da61052bbfeee2fa37227e7a94d8ebb23fc1e9cc88a357f95ff4a10d012&scene=21#wechat_redirect

https://zhuanlan.zhihu.com/p/570487666

https://zhuanlan.zhihu.com/p/687473210

https://blog.51cto.com/u_12413309/6242987

https://www.cnblogs.com/kinologic/p/14105907.html
"""


#%%
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
c = 3e8  # 光速 (m/s)
f0 = 24e9  # 起始频率 (Hz)
B = 250e6  # 带宽 (Hz)
T = 1e-3  # 调制周期 (s)
N = 1024  # 采样点数
fs = 2e6  # 采样频率 (Hz)
SNR = 10  # 信噪比 (dB)
num_paths = 3  # 多路径数量

# 生成发射信号
t = np.linspace(0, T, N, endpoint=False)
tx_signal = np.cos(2 * np.pi * (f0 * t + (B / (2 * T)) * t ** 2))

# 模拟多路径效应
delays = [50, 100, 150]  # 多路径延迟（单位：采样点）
attenuations = [0.8, 0.5, 0.3]  # 多路径衰减系数
rx_signal = np.zeros_like(tx_signal)
for delay, attenuation in zip(delays, attenuations):
    rx_signal += attenuation * np.roll(tx_signal, delay)

# 添加噪声
noise_power = 10 ** (-SNR / 10)  # 噪声功率
noise = np.random.normal(0, np.sqrt(noise_power), N)
rx_signal += noise

# 计算差频信号
mix_signal = tx_signal * rx_signal

# 傅里叶变换得到频谱
fft_result = np.fft.fft(mix_signal)
freq = np.fft.fftfreq(N, 1 / fs)

# 找到峰值频率
peak_freq = freq[np.argmax(np.abs(fft_result))]

# 计算距离
distance = (c * peak_freq * T) / (2 * B)
print(f"Distance: {distance:.2f} meters")

# 计算速度 (假设有多普勒频移)
doppler_shift = 1000  # 假设多普勒频移为1kHz
velocity = (doppler_shift * c) / (2 * f0)
print(f"Velocity: {velocity:.2f} m/s")

# 计算角度 (假设有两个接收天线)
phase_diff = np.angle(fft_result[delays[0]])  # 使用第一个路径的相位差
wavelength = c / f0
antenna_distance = wavelength / 2  # 天线间距
angle = np.arcsin((phase_diff * wavelength) / (2 * np.pi * antenna_distance))
print(f"Angle: {np.degrees(angle):.2f} degrees")

# 绘制频谱
plt.plot(freq, np.abs(fft_result))
plt.title("Frequency Spectrum with Noise and Multipath")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()





#%%












