#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 21:50:43 2025

@author: jack

下面有些不对，主要是ambgfun在matlab是自带的，但是python没有，待修正。


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftshift, ifft
from scipy.linalg import pinv
from scipy.signal.windows import hamming
import matplotlib.gridspec as gridspec
# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
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


def nextpow2(n):
    """计算下一个2的幂"""
    return int(np.ceil(np.log2(n)))

def ambgfun(x, fs, PRF):
    """
    模糊函数计算
    """
    N = len(x)
    # 时延范围
    max_delay = int(N / 2)
    delays = np.arange(-max_delay, max_delay) / fs

    # 多普勒范围
    max_doppler = PRF / 2
    dopplers = np.linspace(-max_doppler, max_doppler, 256)

    # 初始化模糊函数矩阵
    afmag = np.zeros((len(dopplers), len(delays)))

    # 计算模糊函数
    for i, delay in enumerate(delays):
        delay_samples = int(delay * fs)
        for j, doppler in enumerate(dopplers):
            # 时延信号
            if delay_samples >= 0:
                x1 = x[:N-delay_samples]
                x2 = x[delay_samples:] * np.exp(1j * 2 * np.pi * doppler * np.arange(N-delay_samples) / fs)
            else:
                x1 = x[-delay_samples:]
                x2 = x[:N+delay_samples] * np.exp(1j * 2 * np.pi * doppler * np.arange(N+delay_samples) / fs)

            # 计算互相关
            afmag[j, i] = np.abs(np.sum(x1 * np.conj(x2)))

    # 归一化
    afmag = afmag / np.max(afmag)

    return afmag, delays, dopplers

def received_pulseburst(St, num_pulses, fd, PRF, td, N_PRT, N_d, N_st):
    """生成脉冲串接收信号"""
    received = np.zeros(num_pulses * N_PRT, dtype=complex)
    for n in range(num_pulses):
        # 生成单脉冲回波（含多普勒相位和时间延迟）
        doppler_phase = np.exp(1j * 2 * np.pi * fd * (n / PRF + td))
        # 计算回波在接收窗口中的位置
        start_idx = n * N_PRT + N_d
        end_idx = start_idx + N_st

        # 截断处理防止越界
        if end_idx > num_pulses * N_PRT:
            end_idx = num_pulses * N_PRT
            valid_len = end_idx - start_idx
            received[start_idx:end_idx] = received[start_idx:end_idx] + St[:valid_len] * doppler_phase
        else:
            received[start_idx:end_idx] = received[start_idx:end_idx] + St * doppler_phase

    return received

def received_array(St, array_phase, N_R, num_pulses, fd, PRF, td, N_PRT, N_d, N_st):
    """生成阵列接收信号"""
    # 初始化多通道接收信号
    rx_array_signal = np.zeros((num_pulses * N_PRT, N_R), dtype=complex)
    for n in range(num_pulses):
        # 生成单脉冲回波（含多普勒相位和时间延迟）
        doppler_phase = np.exp(1j * 2 * np.pi * fd * (n / PRF + td))
        # 生成阵列接收信号
        for k in range(N_R):
            # 每个阵元的相位补偿
            R_phase = array_phase[k] * doppler_phase
            # 计算信号位置
            start_idx = n * N_PRT + N_d
            end_idx = start_idx + N_st
            # 截断处理
            if end_idx > num_pulses * N_PRT:
                end_idx = num_pulses * N_PRT
                valid_len = end_idx - start_idx
                rx_array_signal[start_idx:end_idx, k] = (
                    rx_array_signal[start_idx:end_idx, k] +
                    St[:valid_len] * R_phase
                )
            else:
                rx_array_signal[start_idx:end_idx, k] = (
                    rx_array_signal[start_idx:end_idx, k] +
                    St * R_phase
                )

    return rx_array_signal

# def main():
###########################################################################
#                           参数设置
###########################################################################
eps = 0.0001
c = 3e8
B = 1e6                         # 带宽1MHz
tau = 100e-6                    # 脉宽100μs
fs = 5 * B                      # 采样率
Ts = 1 / fs                     # 采样间隔
K = B / tau                     # 调频斜率
N = int(tau * fs)               # 采样点数

###########################################################################
#                           T1: LFM信号分析
###########################################################################
print("=" * 60)
print("T1: LFM信号分析")
print("=" * 60)

# 生成LFM信号
t = np.linspace(-tau/2, tau/2, N)  # 选取采样点，在-T/2与T/2间生成N个点
lfm = np.exp(1j * np.pi * K * t**2)

# 图形1: LFM信号时域波形（两个子图）
plt.figure(1, figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t * 1e6, np.real(lfm), 'b')
plt.xlabel('时间 (μs)')
plt.ylabel('幅度')
plt.title('LFM信号时域波形（实部）')
plt.grid(True)
plt.axis('tight')

plt.subplot(2, 1, 2)
plt.plot(t * 1e6, np.imag(lfm), 'r')
plt.xlabel('时间 (μs)')
plt.ylabel('幅度')
plt.title('LFM信号时域波形（虚部）')
plt.grid(True)
plt.axis('tight')
plt.tight_layout()
plt.show()

# 图形2: LFM信号频域波形
f = np.linspace(-fs/2, fs/2, N)
S = fftshift(fft(lfm))
plt.figure(2, figsize=(10, 6))
plt.plot(f / 1e6, np.abs(S))
plt.xlabel('频率 (MHz)')
plt.ylabel('幅度')
plt.title('LFM信号频域波形')
plt.grid(True)
plt.axis('tight')
plt.show()

# 图形3: LFM信号模糊函数3D图
print("计算模糊函数...")
afmag, delay, doppler = ambgfun(lfm, fs, 10e3)

plt.figure(3, figsize=(12, 8))
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(delay * 1e6, doppler / 1e3)  # 注意单位：多普勒用kHz
surf = ax.plot_surface(X, Y, afmag, cmap='jet', linewidth=0, antialiased=True)
ax.set_xlabel('时延τ (μs)')
ax.set_ylabel('多普勒频率F_D (kHz)')
ax.set_zlabel('幅度')
plt.title('LFM信号模糊函数')
plt.colorbar(surf)
# 设置视角
ax.view_init(elev=30, azim=45)
plt.show()

# 图形4: LFM信号模糊函数等高线图
plt.figure(4, figsize=(10, 8))
contour = plt.contour(delay * 1e6, doppler / 1e3, afmag)
plt.clabel(contour, inline=True, fontsize=8)
plt.xlabel('时延τ (μs)')
plt.ylabel('多普勒频率F_D (kHz)')
plt.title('LFM信号模糊函数等高线图')
plt.xlim([-100, 100])
plt.ylim([-10, 10])  # 调整范围以匹配MATLAB
plt.grid(True)
plt.show()

###########################################################################
#                           T2: 单目标回波处理
###########################################################################
print("=" * 60)
print("T2: 单目标回波处理")
print("=" * 60)

R_t = 90e3  # 目标距离90km
v_t = 60    # 目标速度60m/s

# 生成LFM信号St
t = np.arange(0, tau, Ts)
St = np.exp(1j * np.pi * K * t**2)

td = 2 * R_t / c  # 目标时延 (s)
N_d = int(td * fs)  # 目标时延采样点数
fc = 1e9       # 假设载频1GHz
fd = 2 * v_t / (c / fc)  # 多普勒频移 (Hz)

# 生成单目标回波信号
echo = np.concatenate([np.zeros(N_d), St * np.exp(1j * 2 * np.pi * fd * td)])
t1 = (np.arange(len(echo))) * Ts * 1e3  # ms

# 图形5: 回波信号时域波形（两个子图）
plt.figure(5, figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t1, np.real(echo), 'b')
plt.xlabel('时间 (ms)')
plt.ylabel('幅度')
plt.title('回波信号时域波形（实部）')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t1, np.imag(echo), 'r')
plt.xlabel('时间 (ms)')
plt.ylabel('幅度')
plt.title('回波信号时域波形（虚部）')
plt.grid(True)
plt.tight_layout()
plt.show()

# 匹配滤波
matched_filter = np.conj(St[::-1])
mf = np.convolve(echo, matched_filter)
mf = mf[len(St)-1:]
mf_abs = np.abs(mf)
t_mf = (np.arange(len(mf))) * Ts * c / 2 / 1e3  # km

# 图形6: 匹配滤波输出
plt.figure(6, figsize=(10, 6))
plt.plot(t_mf, mf_abs)
plt.xlabel('距离 (km)')
plt.ylabel('幅度')
plt.title('匹配滤波输出')
plt.xlim([0, 100])
plt.grid(True)
plt.show()

###########################################################################
#                           T3: 脉冲串信号处理
###########################################################################
print("=" * 60)
print("T3: 脉冲串信号处理")
print("=" * 60)

R_t = 90e3  # 目标距离90km
v_t = 60    # 目标速度60m/s
t = np.arange(0, tau, Ts)
St = np.exp(1j * np.pi * K * t**2)  # 生成LFM信号St

PRF = 1e3       # 脉冲重复频率1kHz
PRT = 1 / PRF   # 脉冲重复间隔1ms
fc = 1e9        # 载频1GHz
lambda_val = c / fc  # 波长
SNR_dB = -10    # 信噪比-10dB
fd = 2 * v_t / lambda_val  # 多普勒频移 (Hz)
N_PRT = int(PRT * fs)          # 单个PRT的采样点数
num_pulses = 64                # 模拟的脉冲数量
N_st = len(St)  # 单个脉冲内的采样点数

# 调用脉冲串接收信号函数
received = received_pulseburst(St, num_pulses, fd, PRF, td, N_PRT, N_d, N_st)

# 生成复高斯噪声
rx_power = np.mean(np.abs(received)**2)
SNR_linear = 10**(SNR_dB/10)
noise_power = rx_power / SNR_linear
noise = np.sqrt(noise_power/2) * (np.random.randn(len(received)) + 1j * np.random.randn(len(received)))
rx_signal = received + noise

# 图形7: 接收信号（两个子图）
plt.figure(7, figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(np.real(rx_signal))
plt.xlabel('采样点')
plt.ylabel('幅度')
plt.title('接收信号实部')
plt.xlim([0, min(10000, len(rx_signal))])

plt.subplot(2, 1, 2)
N_fft = 2**nextpow2(len(rx_signal))
f_fft = fs/2 * np.linspace(-1, 1, N_fft)
spectrum = fftshift(fft(rx_signal, N_fft))
plt.plot(f_fft/1e6, 20*np.log10(np.abs(spectrum)))
plt.xlabel('频率 (MHz)')
plt.ylabel('幅度 (dB)')
plt.title('接收信号频谱')
plt.xlim([-fs/2e6, fs/2e6])
plt.grid(True)
plt.tight_layout()
plt.show()

# 匹配滤波与MTD处理
# 分帧处理，将接收信号分割为各脉冲
rx_pulses = rx_signal.reshape(N_PRT, num_pulses).T

# 生成匹配滤波器
h_mf = np.conj(St[::-1])

# 预分配存储空间
mf_output = np.zeros((num_pulses, N_PRT + N_st - 1), dtype=complex)
for i in range(num_pulses):
    pulse = rx_pulses[i, :]
    mf_output[i, :] = np.convolve(pulse, h_mf, mode='full')

# 距离轴校正（补偿匹配滤波引入的延迟）
range_bins_mf = ((np.arange(mf_output.shape[1]) - (N_st-1)) * (c/(2*fs)))
valid_idx = range_bins_mf >= 0  # 去除负距离
range_bins_mf = range_bins_mf[valid_idx] / 1e3
mf_output = mf_output[:, valid_idx]

# 图形8: 匹配滤波后的波形（4个子图）
plt.figure(8, figsize=(12, 10))
for i in range(4):
    plt.subplot(4, 1, i+1)
    plt.plot(np.abs(mf_output[i, :]))
    plt.xlabel('采样点')
    plt.ylabel('幅度')
    if i == 0:
        plt.title('匹配滤波后的波形')
plt.tight_layout()
plt.show()

# MTD处理（多普勒FFT）
mtd = fftshift(fft(mf_output, axis=0), axes=0)
mtd_abs = np.abs(mtd)
mtd_db = 20 * np.log10(mtd_abs / np.max(mtd_abs) + eps)

# 图形9: MTD后的波形（4个子图）
plt.figure(9, figsize=(12, 10))
for i in range(4):
    plt.subplot(4, 1, i+1)
    plt.plot(mtd_abs[:, i])
    plt.xlabel('采样点')
    plt.ylabel('幅度')
    if i == 0:
        plt.title('MTD后的波形')
plt.tight_layout()
plt.show()

# 计算速度轴
doppler_bins = (np.arange(num_pulses) - num_pulses//2) * PRF / num_pulses  # 多普勒频率（Hz）
speed_bins_mf = doppler_bins * lambda_val / 2  # 速度轴（m/s）

# 找到幅度最大的点（即目标位置）
max_idx = np.unravel_index(np.argmax(mtd_abs), mtd_abs.shape)
doppler_idx, range_idx = max_idx
detected_range_mf = range_bins_mf[range_idx]
detected_speed_mf = speed_bins_mf[doppler_idx]

# 图形10: MTD结果图
plt.figure(10, figsize=(12, 8))
maxvalue = np.max(mtd_abs)
plt.imshow(mtd_abs.T, extent=[speed_bins_mf[0], speed_bins_mf[-1], range_bins_mf[0], range_bins_mf[-1]],
           aspect='auto', cmap='jet', origin='lower', vmin=0, vmax=maxvalue)
plt.colorbar(label='幅度')
plt.xlabel('速度 (m/s)')
plt.ylabel('距离 (km)')
plt.title('MTD')
plt.xlim([50, 70])
plt.ylim([80, 100])

# 标记真实目标
plt.plot(60, 90, 'go', markersize=8, label='真实目标')
plt.text(60, 90, f'(90.0 km, 60.0 m/s)', fontsize=12, color='g',
         horizontalalignment='center')

# 标记检测目标
plt.plot(detected_speed_mf, detected_range_mf, 'ro', markersize=8, label='检测目标')
plt.text(detected_speed_mf, detected_range_mf,
         f'({detected_range_mf:.1f} km, {detected_speed_mf:.1f} m/s)',
         fontsize=12, color='r', horizontalalignment='center')
plt.legend()
plt.show()

# 图形11: MTD输出三维图
# 修复维度不匹配问题
X, Y = np.meshgrid(range_bins_mf, speed_bins_mf)
# 确保Z矩阵的维度与X和Y匹配
# Z = mtd_db  # 转置以匹配meshgrid的维度

fig = plt.figure(11, figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, mtd_db , cmap='jet', alpha=0.8)
ax.set_xlabel('距离 (km)')
ax.set_ylabel('速度 (m/s)')
ax.set_zlabel('幅度 (dB)')
ax.set_title('MTD输出"距离-速度-幅度"三维图')
plt.colorbar(surf)

# 标记检测目标
ax.plot([detected_range_mf], [detected_speed_mf], [mtd_db[doppler_idx, range_idx]], 'ro', markersize=10, label='检测目标')
plt.legend()
plt.show()

###########################################################################
#                           T4: 阵列信号处理
###########################################################################
print("=" * 60)
print("T4: 阵列信号处理")
print("=" * 60)

N_R = 16           # 阵元数
d = lambda_val / 2     # 阵元间距（m）
theta_target = 0   # 目标方位角 (度)

# 阵列方向图生成
theta = np.arange(-90, 90.1, 0.1)  # 方位角范围（-90°到90°）
weights_uniform = np.ones(N_R)
weights_hamming = hamming(N_R)  # Hamming窗加权

# 计算阵列方向图
array_pattern_u = np.zeros(len(theta))
array_pattern_w = np.zeros(len(theta))
for i, theta_i in enumerate(theta):
    steering = np.exp(-1j * 2 * np.pi * d / lambda_val * np.sin(np.deg2rad(theta_i)) * np.arange(N_R))
    array_pattern_u[i] = np.abs(np.dot(weights_uniform, steering))
    array_pattern_w[i] = np.abs(np.dot(weights_hamming, steering))

array_pattern_db = 20 * np.log10(array_pattern_u / np.max(array_pattern_u))
array_pattern_w_db = 20 * np.log10(array_pattern_w / np.max(array_pattern_w))

# 图形12: 阵列方向图
plt.figure(12, figsize=(10, 6))
plt.plot(theta, array_pattern_db, 'b', linewidth=1.5, label='不加窗')
plt.plot(theta, array_pattern_w_db, 'r', linewidth=1.5, label='Hamming窗')
plt.xlabel('方位角 (°)')
plt.ylabel('增益 (dB)')
plt.title('16阵元均匀线阵方向图加窗前后对比')
plt.legend()
plt.xlim([-90, 90])
plt.ylim([-50, 0])
plt.grid(True)
plt.show()

# 模拟阵列发送和接收信号
# 生成阵列导向矢量
array_phase = np.exp(-1j * 2 * np.pi * d / lambda_val * np.sin(np.deg2rad(theta_target)) * np.arange(N_R))

# 调用阵列接收信号函数
rx_array_signal = received_array(St, array_phase, N_R, num_pulses, fd, PRF, td, N_PRT, N_d, N_st)

# 添加阵列噪声
noise = np.sqrt(noise_power/2) * (np.random.randn(*rx_array_signal.shape) + 1j * np.random.randn(*rx_array_signal.shape))
rx_array_signal = rx_array_signal + noise

# DBF处理
# 计算加权导向矢量
weighted_steering = array_phase * weights_hamming  # 应用窗函数
# 波束形成权值归一化
w = weighted_steering / np.dot(array_phase.conj(), weighted_steering)  # 保持主瓣增益
# 进行波束形成
rx_beamformed = np.dot(rx_array_signal, w.conj())

# 分帧处理
rx_pulses = rx_beamformed.reshape(N_PRT, num_pulses).T

# 匹配滤波
dbf_mf_output = np.zeros((num_pulses, N_PRT + N_st - 1), dtype=complex)
for i in range(num_pulses):
    dbf_mf_output[i, :] = np.convolve(rx_pulses[i, :], h_mf, mode='full')

# 距离轴校正
range_bins = ((np.arange(dbf_mf_output.shape[1]) - (N_st-1)) * (c/(2*fs)))
valid_idx = range_bins >= 0
range_bins = range_bins[valid_idx] / 1e3
dbf_mf_output = dbf_mf_output[:, valid_idx]

# MTD处理
mtd_output = fftshift(fft(dbf_mf_output, axis=0), axes=0)
mtd_output_abs = np.abs(mtd_output)
mtd_output_db = 20 * np.log10(mtd_output_abs / np.max(mtd_output_abs) + eps)

# 计算速度轴
speed_bins = doppler_bins * lambda_val / 2  # 速度轴（m/s）

# 找到检测目标
max_idx = np.unravel_index(np.argmax(mtd_output_abs), mtd_output_abs.shape)
doppler_idx, range_idx = max_idx
detected_range = range_bins[range_idx]
detected_speed = speed_bins[doppler_idx]

# 图形13: DBF-脉冲压缩-MTD三维图
# 修复维度不匹配问题
X, Y = np.meshgrid(range_bins, speed_bins)
# 确保Z矩阵的维度与X和Y匹配
Z = mtd_output_db.T  # 转置以匹配meshgrid的维度

fig = plt.figure(13, figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z.T, cmap='jet', alpha=0.8)
ax.set_xlabel('距离 (km)')
ax.set_ylabel('速度 (m/s)')
ax.set_zlabel('幅度 (dB)')
ax.set_title('DBF-脉冲压缩-MTD输出"距离-速度-幅度"三维图')
plt.colorbar(surf)

# 标记检测目标
ax.plot([detected_range], [detected_speed], [mtd_output_db[doppler_idx, range_idx]],
        'ro', markersize=10, label='检测目标')
plt.legend()
plt.show()

print("所有图形生成完成！")

