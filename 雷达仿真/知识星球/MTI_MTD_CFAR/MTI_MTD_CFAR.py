#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 11:19:02 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math
from mpl_toolkits.mplot3d import Axes3D
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


# 关闭所有图形
plt.close('all')

# 信号参数
C = 3.0e8
RF = 5e9  # RF
Lambda = C / RF
PulseNumber = 32   # 回波脉冲数
BandWidth = 2.0e6  # 发射信号带宽
TimeWidth = 40.0e-6  # 发射信号时宽
PRT = 200e-6
PRF = 1 / PRT
Fs = 10.0e6  # 采样频率
AWGNpower = 0  # dB

SampleNumber = int(Fs * PRT)  # 计算一个脉冲周期的采样点数
TotalNumber = SampleNumber * PulseNumber  # 总的采样点数
BlindNumber = int(Fs * TimeWidth)  # 计算一个脉冲周期的盲区-遮挡样点数

# 目标参数
TargetNumber = 3  # 目标个数
SigPower = np.array([1, 1, 1])  # 目标功率,无量纲
TargetDistance = np.array([5000, 15000, 20000])  # 目标距离,单位m
DelayNumber = np.array([int(Fs * 2 * dist / C) for dist in TargetDistance])  # 把目标距离换算成采样点（距离门）
TargetVelocity = np.array([0, 50, 200])  # 目标径向速度 单位m/s
TargetFd = 2 * TargetVelocity / Lambda  # 计算目标多卜勒

# 信号产生
number = int(Fs * TimeWidth)  # 回波的采样点数=脉压系数长度=暂态点数目+1
if number % 2 != 0:
    number = number + 1

Chirp = np.zeros(number, dtype=complex)
for i in range(-int(number/2), int(number/2)):
    idx = i + int(number/2)
    Chirp[idx] = np.exp(1j * (np.pi * (BandWidth / TimeWidth) * (i / Fs) ** 2))

coeff = np.conj(np.flip(Chirp))

# 回波串
SignalAll = np.zeros(TotalNumber, dtype=complex)  # 所有脉冲的信号,先填0
for k in range(TargetNumber):  # 依次产生各个目标
    SignalTemp = np.zeros(SampleNumber, dtype=complex)  # 一个脉冲
    # 一个脉冲的1个目标（未加多普勒速度）
    start_idx = DelayNumber[k]
    end_idx = DelayNumber[k] + number
    if end_idx <= SampleNumber:
        SignalTemp[start_idx:end_idx] = np.sqrt(SigPower[k]) * Chirp

    Signal = np.zeros(TotalNumber, dtype=complex)
    for i in range(PulseNumber):
        start_sig = i * SampleNumber
        end_sig = (i + 1) * SampleNumber
        Signal[start_sig:end_sig] = SignalTemp

    # 目标的多普勒速度*时间=目标的多普勒相移
    FreqMove = np.exp(1j * 2 * np.pi * TargetFd[k] * np.arange(TotalNumber) / Fs)
    Signal = Signal * FreqMove
    SignalAll = SignalAll + Signal

# 添加高斯白噪声
def awgn(signal, snr):
    """
    添加高斯白噪声
    snr: 信噪比 (dB)
    """
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power / (10**(snr/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise

Echo = awgn(SignalAll, AWGNpower)

for i in range(PulseNumber):  # 在接收机闭锁期,接收的回波为0
    start_idx = i * SampleNumber
    end_idx = i * SampleNumber + number
    if end_idx <= TotalNumber:
        Echo[start_idx:end_idx] = 0

# 时域脉压
pc_time0 = np.convolve(Echo, coeff)

# 频域脉压
nfft = 2**int(np.ceil(np.log2(len(Echo))))  # FFT点数
Echo_fft = np.fft.fft(Echo, nfft)
coeff_fft = np.fft.fft(coeff, nfft)
pc_fft = Echo_fft * coeff_fft
pc_freq0 = np.fft.ifft(pc_fft)
pc_freq1 = pc_freq0[number-1:TotalNumber+number-1]  # 去掉暂态点 number-1个,后填充点若干

# 按照脉冲号、距离门号重排数据
pc = np.zeros((PulseNumber, SampleNumber), dtype=complex)
for i in range(PulseNumber):
    start_idx = i * SampleNumber
    end_idx = (i + 1) * SampleNumber
    pc[i, :] = pc_freq1[start_idx:end_idx]

# MTI
mti = np.zeros((PulseNumber-1, SampleNumber), dtype=complex)
for i in range(PulseNumber-1):  # 滑动对消，少了一个脉冲
    mti[i, :] = pc[i+1, :] - pc[i, :]

# 修复绘图部分 - 确保形状匹配
xmti = C / 2 * np.arange(0, SampleNumber) / Fs  # 距离轴
ymti = np.arange(1, PulseNumber)  # 脉冲数轴

# 创建3D图形 - 修复形状不匹配问题
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(xmti, ymti)
Z = np.abs(mti)
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title('MTI 结果')
ax.set_xlabel('距离/m')
ax.set_ylabel('脉冲数')
plt.show()

# 绘制2D图形
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(xmti, np.abs(pc[0, :]))
plt.grid(True)
plt.title('脉冲压缩结果')
plt.xlabel('距离/m')

plt.subplot(2, 1, 2)
plt.plot(xmti, np.abs(mti[0, :]))
plt.grid(True)
plt.title('MTI一次对消结果')
plt.xlabel('距离/m')
plt.tight_layout()
plt.show()

# MTD
mtd = np.zeros((PulseNumber, SampleNumber), dtype=complex)
for i in range(SampleNumber):
    buff = pc[:, i]
    buff_fft = np.fft.fft(buff)
    mtd[:, i] = buff_fft

xmtd = C / 2 * np.arange(0, SampleNumber) / Fs  # 距离轴
ymtd = Lambda / 2 * np.arange(0, PulseNumber) * PRF / PulseNumber  # 速度轴

# MTD 3D图 - 修复形状不匹配问题
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X_mtd, Y_mtd = np.meshgrid(xmtd, ymtd)
Z_mtd = np.abs(mtd)
ax.plot_surface(X_mtd, Y_mtd, Z_mtd, cmap='viridis')
ax.set_title('MTD 结果')
ax.set_xlabel('距离/m')
ax.set_ylabel('速度/m·s⁻¹')
ax.set_xlim([np.min(xmtd), np.max(xmtd)])
ax.set_ylim([np.min(ymtd), np.max(ymtd)])
plt.show()

mtd_result = np.max(np.abs(mtd), axis=0)

# CFAR
Pfa = 1e-6
alpha = SampleNumber * (Pfa ** (-1 / SampleNumber) - 1)
num = 60
protect = 20

cfar_result = np.zeros(SampleNumber)
# 第1点恒虚警处理时噪声均值由其后面的num点的噪声决定
cfar_result[0] = np.mean(mtd_result[1:num+1])

for i in range(1, num):  # 第2点到第num点的恒虚警的噪声均由其前面和后面的num点的噪声共同决定
    cfar_result[i] = (np.mean(mtd_result[:i]) + np.mean(mtd_result[i+1:i+num+1])) / 2

for i in range(num, SampleNumber - num):
    # 正常的数据点恒虚警处理的噪声均值由其前面和后面各num点噪声共同决定
    left_mean = np.mean(mtd_result[i-num:i-protect])
    right_mean = np.mean(mtd_result[i+protect:i+num])
    cfar_result[i] = max(left_mean, right_mean)

for i in range(SampleNumber - num, SampleNumber - 1):
    # 倒数第num点到倒数第2点恒虚警处理的噪声均值由其前面num点和后面的噪声共同决定
    cfar_result[i] = (np.mean(mtd_result[i-num:i]) + np.mean(mtd_result[i+1:SampleNumber])) / 2

# 最后一点的恒虚警处理的噪声均值由其前面的num点的噪声决定
cfar_result[SampleNumber-1] = np.mean(mtd_result[SampleNumber-num-1:SampleNumber-1])

s_result = np.zeros(SampleNumber)
for i in range(SampleNumber):
    if mtd_result[i] >= alpha * cfar_result[i]:
        s_result[i] = mtd_result[i]

# 绘制CFAR结果
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(xmtd, mtd_result)
plt.plot(xmtd, alpha * cfar_result, 'r')
plt.xlim([np.min(xmtd), np.max(xmtd)])
plt.xlabel('距离/m')
plt.title('MTD处理后求模结果(信号最大通道)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(xmtd, s_result)
plt.xlim([np.min(xmtd), np.max(xmtd)])
plt.xlabel('距离/m')
plt.title('恒虚警处理结果')
plt.grid(True)
plt.tight_layout()
plt.show()
