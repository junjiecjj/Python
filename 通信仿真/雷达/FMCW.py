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
import numpy as np
# import scipy
# from sklearn.metrics import pairwise_distances
# import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
# import commpy

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22  # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22


#%%
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
c = 3e8        # 光速 (m/s)
f0 = 24e9      # 起始频率 (Hz)
B = 250e6      # 带宽 (Hz)
T = 1e-3       # 调制周期 (s)
N = 1024       # 采样点数
fs = 2e6       # 采样频率 (Hz)
SNR = 10       # 信噪比 (dB)
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

import numpy as np
import matplotlib.pyplot as plt

# ====== 参数设置 ======
c = 3e8             # 光速 (m/s)
fc = 77e9           # 载波频率 (Hz)
B = 300e6           # 带宽 (Hz)
Tc = 50e-6          # Chirp 时长 (s)
N_chirps = 64       # 每帧的 Chirp 数
N_rx = 4            # 接收天线数量
d_antenna = 0.5 * c / fc  # 天线间距 (半波长)

# 目标参数 (模拟单目标)
target_distance = 100   # 距离 (米)
target_velocity = 30    # 速度 (m/s, 朝向雷达为负)
target_angle = 30       # 角度 (度)

# ====== 生成 FMCW 信号 ======
fs = 2 * B           # 采样率
t = np.linspace(0, Tc, int(fs * Tc))  # 单个 Chirp 的时间轴

# 发射信号 (Chirp)
tx_chirp = np.exp(1j * np.pi * (B/Tc) * t**2)

# 模拟接收信号 (含延迟和多普勒频移)
tau = 2 * target_distance / c          # 距离延迟
doppler_shift = 2 * target_velocity * fc / c  # 多普勒频移

# 生成接收信号 (每个 Chirp 和天线)
rx_signals = np.zeros((N_rx, N_chirps, len(t)), dtype=np.complex_)
for rx in range(N_rx):
    for chirp in range(N_chirps):
        phase_shift = 2 * np.pi * rx * d_antenna * np.sin(np.radians(target_angle)) / (c / fc)
        time_shift = t - tau + (chirp * Tc) * (2 * target_velocity / c)
        rx_chirp = np.exp(1j * (np.pi * (B/Tc) * (time_shift)**2 + doppler_shift * t + phase_shift))
        rx_signals[rx, chirp, :] = rx_chirp + 0.01 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))  # 添加噪声

# ====== 信号处理 ======
# 混频生成中频信号 (IF)
if_signals = rx_signals * np.conj(tx_chirp)

# 距离估计 (FFT)
range_fft = np.fft.fft(if_signals, axis=2)
range_profile = np.mean(np.abs(range_fft), axis=(0,1))
range_bins = np.fft.fftfreq(len(t), 1/fs) * c * Tc / (2 * B)
distance_est = range_bins[np.argmax(range_profile)]

# 速度估计 (Chirp间相位变化)
phase_changes = np.angle(range_fft[:, :, np.argmax(range_profile)])
velocity_fft = np.fft.fft(phase_changes, axis=1)
velocity_profile = np.mean(np.abs(velocity_fft), axis=0)
velocity_bins = np.fft.fftfreq(N_chirps, Tc) * c / (2 * fc)
velocity_est = velocity_bins[np.argmax(velocity_profile)]

# 角度估计 (天线间相位差)
angle_fft = np.fft.fft(np.mean(range_fft[:, :, np.argmax(range_profile)], axis=1), axis=0)
angle_bins = np.arcsin(np.fft.fftshift(np.fft.fftfreq(N_rx, d_antenna/(c/fc)))) * 180/np.pi
angle_est = angle_bins[np.argmax(np.abs(angle_fft))]

# ====== 结果输出 ======
print(f"真实值: 距离={target_distance}m, 速度={target_velocity}m/s, 角度={target_angle}°")
print(f"估计值: 距离={distance_est:.2f}m, 速度={velocity_est:.2f}m/s, 角度={angle_est:.2f}°")

# ====== 可视化 ======
plt.figure(figsize=(12, 8))

# 距离谱
plt.subplot(3, 1, 1)
plt.plot(range_bins, 20*np.log10(range_profile))
plt.title("距离谱 (FFT)")
plt.xlabel("距离 (m)")
plt.ylabel("幅度 (dB)")

# 速度谱
plt.subplot(3, 1, 2)
plt.plot(velocity_bins, 20*np.log10(np.abs(velocity_profile)))
plt.title("速度谱 (FFT)")
plt.xlabel("速度 (m/s)")
plt.ylabel("幅度 (dB)")

# 角度谱
plt.subplot(3, 1, 3)
plt.plot(angle_bins, 20*np.log10(np.fft.fftshift(np.abs(angle_fft))))
plt.title("角度谱 (FFT)")
plt.xlabel("角度 (°)")
plt.ylabel("幅度 (dB)")
plt.tight_layout()
plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt

# ====== 参数设置 ======
c = 3e8             # 光速 (m/s)
fc = 77e9           # 载波频率 (Hz)
B = 400e6           # 带宽 (Hz) 增大带宽以提高距离分辨率
Tc = 100e-6         # Chirp 时长 (s)
N_chirps = 128      # 每帧的 Chirp 数 (增多以提高速度分辨率)
N_rx = 4            # 接收天线数量
d_antenna = 0.5 * c / fc  # 天线间距 (半波长)

# 目标参数 (单目标)
target_distance = 80.0    # 距离 (米)
target_velocity = -25.0   # 速度 (m/s, 负值表示远离雷达)
target_angle = 20.0       # 角度 (度)

# ====== 生成 FMCW 信号 ======
fs = 2 * B          # 采样率 (满足 Nyquist)
num_samples = int(Tc * fs)
t = np.linspace(0, Tc, num_samples)  # 单个 Chirp 的时间轴

# 发射信号 (线性调频)
tx_chirp = np.exp(1j * np.pi * (B / Tc) * t**2)

# 接收信号建模 (考虑延迟、多普勒、天线相位差)
tau = 2 * target_distance / c  # 双程延迟
doppler_shift = 2 * target_velocity * fc / c  # 多普勒频移

# 接收信号矩阵 [N_rx, N_chirps, num_samples]
rx_signals = np.zeros((N_rx, N_chirps, num_samples), dtype=np.complex_)

for rx in range(N_rx):
    # 天线相位差 (基于入射角)
    phase_shift_rx = 2 * np.pi * rx * d_antenna * np.sin(np.radians(target_angle)) / (c / fc)

    for chirp in range(N_chirps):
        # 时间延迟 + 多普勒引起的相位变化 (每个 Chirp 增加相位偏移)
        t_delay = t - tau  # 回波延迟
        phase_doppler = 2 * np.pi * doppler_shift * (t + chirp * Tc)  # 多普勒相位

        # 生成接收信号 (忽略时间拉伸，仅相位近似)
        rx_signal = np.exp(1j * (
            np.pi * (B / Tc) * (t_delay)**2 +  # 延迟后的 Chirp
            phase_doppler +                    # 多普勒相位
            phase_shift_rx                     # 天线相位差
        ))

        # 添加噪声 (信噪比 SNR ≈ 20 dB)
        noise = 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        rx_signals[rx, chirp, :] = rx_signal + noise

# ====== 信号处理 ======
# 混频生成中频信号 (IF)
if_signals = rx_signals * np.conj(tx_chirp)  # [N_rx, N_chirps, num_samples]

# --- 距离估计 (Range FFT) ---
range_fft = np.fft.fft(if_signals, axis=2)  # 沿时间轴做FFT
range_profile = np.mean(np.abs(range_fft), axis=(0, 1))  # 平均所有天线和Chirp

# 距离轴计算
range_bins = np.fft.fftshift(np.fft.fftfreq(num_samples, 1/fs)) * c * Tc / (2 * B)
distance_est = range_bins[np.argmax(range_profile)]

# --- 速度估计 (Doppler FFT) ---
# 提取峰值距离门对应的相位变化
peak_bin = np.argmax(np.mean(np.abs(range_fft), axis=(0, 1)))
phase_sequence = np.angle(range_fft[:, :, peak_bin])  # [N_rx, N_chirps]

# 对相位序列做FFT (沿Chirp轴)
velocity_fft = np.fft.fft(phase_sequence, axis=1)
velocity_profile = np.mean(np.abs(velocity_fft), axis=0)  # 平均天线

# 速度轴计算
velocity_bins = np.fft.fftshift(np.fft.fftfreq(N_chirps, Tc)) * c / (2 * fc)
velocity_est = velocity_bins[np.argmax(velocity_profile)]

# --- 角度估计 (Angle FFT) ---
# 提取峰值距离门和Chirp的信号
angle_fft_input = range_fft[:, :, peak_bin]  # [N_rx, N_chirps]

# 平均所有Chirp，保留天线维度
angle_fft_input = np.mean(angle_fft_input, axis=1)  # [N_rx]

# 沿天线轴做FFT
angle_fft = np.fft.fft(angle_fft_input * np.hamming(N_rx), n=1024)  # 补零提高分辨率
angle_fft = np.fft.fftshift(angle_fft)

# 角度轴计算
angle_bins = np.arcsin(np.linspace(-1, 1, 1024) * (c / fc) / (N_rx * d_antenna)) * 180 / np.pi
angle_est = angle_bins[np.argmax(np.abs(angle_fft))]

# ====== 结果输出 ======
print(f"真实值: 距离={target_distance}m, 速度={target_velocity}m/s, 角度={target_angle}°")
print(f"估计值: 距离={distance_est:.2f}m, 速度={velocity_est:.2f}m/s, 角度={angle_est:.2f}°")

# ====== 可视化 ======
plt.figure(figsize=(12, 10))

# 距离谱
plt.subplot(3, 1, 1)
plt.plot(range_bins, 20 * np.log10(range_profile))
plt.title("距离谱 (Range FFT)")
plt.xlabel("距离 (m)")
plt.ylabel("幅度 (dB)")
plt.grid(True)

# 速度谱
plt.subplot(3, 1, 2)
plt.plot(velocity_bins, 20 * np.log10(np.abs(velocity_profile)))
plt.title("速度谱 (Doppler FFT)")
plt.xlabel("速度 (m/s)")
plt.ylabel("幅度 (dB)")
plt.grid(True)

# 角度谱
plt.subplot(3, 1, 3)
plt.plot(angle_bins, 20 * np.log10(np.abs(angle_fft)))
plt.title("角度谱 (Angle FFT)")
plt.xlabel("角度 (°)")
plt.ylabel("幅度 (dB)")
plt.grid(True)

plt.tight_layout()
plt.show()

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

# 生成发射信号
t = np.linspace(0, T, N, endpoint=False)
tx_signal = np.cos(2 * np.pi * (f0 * t + (B / (2 * T)) * t ** 2))

# 假设接收信号是发射信号的延迟版本
delay = 100  # 延迟点数
rx_signal = np.roll(tx_signal, delay)

# 计算差频信号
mix_signal = tx_signal * rx_signal

# 傅里叶变换得到频谱
fft_result = np.fft.fft(mix_signal)
freq = np.fft.fftfreq(N, 1/fs)

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
phase_diff = np.angle(fft_result[delay])  # 假设相位差
wavelength = c / f0
antenna_distance = wavelength / 2  # 天线间距
angle = np.arcsin((phase_diff * wavelength) / (2 * np.pi * antenna_distance))
print(f"Angle: {np.degrees(angle):.2f} degrees")

# 绘制频谱
plt.plot(freq, np.abs(fft_result))
plt.title("Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.show()

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




























