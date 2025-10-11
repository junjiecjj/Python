#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 11:04:47 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.constants import c, pi
from scipy.signal import firwin, freqz
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 参数定义 - 严格按MATLAB源码
FFTsize = 4096
d = 0.65
C = c
lamda = 0.03
TimeWidth = 160e-6
BandWidth = 1e6
K = BandWidth / TimeWidth
D = 0.25
Ae = 0.25 * pi * D**2
G = 4 * pi * Ae / lamda**2
RCS = 1500
k = 1.38e-23
T = 290
F = 3
L = 4
Lp = 5
N_CI = 64
Pt_CI = 30
Ru = 80000
theta_3dB = 6
PRT = 800e-6
Fs = 2e6
Ts = 1 / Fs
Va = 600
Vs = 20
alpha = 30
beta = 1
Rs = 20000
nTr = int(PRT * Fs)
nTe = int(TimeWidth * Fs)
nTe = nTe + (nTe % 2)
P_fa = 10e-6

print("开始雷达系统设计分析...")

# (1) 模糊函数和-4dB等高线图 - 严格按MATLAB
print("计算模糊函数...")
eps = 1e-10
tau = np.arange(-TimeWidth, TimeWidth, TimeWidth/1600)
fd = np.arange(-BandWidth, BandWidth, BandWidth/1000)
X, Y = np.meshgrid(tau, fd)
temp1 = 1 - np.abs(X) / TimeWidth
temp2 = pi * TimeWidth * (K * X + Y) * temp1 + eps
ambg = np.abs(temp1 * np.sin(temp2) / temp2)

# 模糊函数3D图 - 严格按MATLAB的mesh
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X*1e6, Y*1e-6, ambg, cmap='jet', alpha=0.9)
ax.set_xlabel('τ/μs')
ax.set_ylabel('fd/MHz')
ax.set_zlabel('|χ(τ,fd)|')
ax.set_title('模糊函数图')
plt.show()

# 距离模糊函数图 - 修正：使用正确的索引
m1, m2 = np.unravel_index(np.argmax(ambg), ambg.shape)
plt.figure(figsize=(8, 6))
plt.plot(tau*1e6, 20*np.log10(np.abs(ambg[m1, :])))
plt.xlabel('τ/μs')
plt.ylabel('|X(τ,0)|/dB')
plt.title('|X(τ,0)|距离模糊图')
plt.grid(True)
plt.axis([-100, 100, -60, 0])
plt.show()

# 速度模糊函数图 - 修正：使用正确的索引
plt.figure(figsize=(8, 6))
plt.plot(fd*1e-6, 20*np.log10(np.abs(ambg[:, m2])))
plt.xlabel('fd/MHz')
plt.ylabel('|X(0,fd)|/dB')
plt.title('|X(0,fd)|速度模糊图')
plt.grid(True)
plt.axis([-1, 1, -60, 0])
plt.show()

# 模糊函数的等高线图
plt.figure(figsize=(8, 6))
plt.contour(tau*1e6, fd*1e-6, ambg, levels=20, colors='blue')
plt.xlabel('τ/μs')
plt.ylabel('fd/MHz')
plt.title('模糊函数的等高线图')
plt.axis([-150, 150, -1, 1])
plt.grid(True)
plt.show()

# -4dB等高线图
plt.figure(figsize=(8, 6))
plt.contour(tau*1e6, fd*1e-6, ambg, levels=[10**(-4/20)], colors='blue')
plt.xlabel('τ/μs')
plt.ylabel('fd/MHz')
plt.title('模糊函数的-4dB切割等高线图局部放大')
plt.axis([-2, 2, -0.01, 0.01])
plt.grid(True)
plt.show()

# 计算-3dB时宽带宽
I2, J2 = np.where(np.abs(20*np.log10(ambg) - (-3)) < 0.1)
tau_3db = np.abs(tau[J2[-1]] - tau[J2[0]]) * 1e6
B_3db = np.abs(fd[I2[-1]] - fd[I2[0]]) * 1e-6
print(f"-3dB时宽: {tau_3db:.2f} μs")
print(f"-3dB带宽: {B_3db:.2f} MHz")

# （4）相干积累提升SNR
print("计算相干积累效果...")
N_pulse = theta_3dB / 60 / PRT
R_max = 100000
R_CI = np.linspace(R_max/400, R_max, 400)

SNR_1 = (10*np.log10(Pt_CI * TimeWidth * G**2 * RCS * lamda**2) -
         10*np.log10((4*pi)**3 * k * T * R_CI**4) - F - L - Lp)
SNR_N = SNR_1 + 10*np.log10(N_CI)

plt.figure(figsize=(8, 6))
plt.plot(R_CI*1e-3, SNR_1, 'b-.', label='相干积累前')
plt.plot(R_CI*1e-3, SNR_N, 'r-', label='相干积累后')
plt.title('相干积累前后信噪比-距离关系曲线')
plt.xlabel('R/km')
plt.ylabel('SNR/dB')
plt.legend()
plt.grid(True)
plt.axis([30, 100, -10, 40])
plt.show()

# （5）脉冲压缩 - 严格按MATLAB修正窗函数问题
print("进行脉冲压缩分析...")
t = np.linspace(-nTe/2, nTe/2-1, nTe) / nTe * TimeWidth
f = np.linspace(-256, 255, 512) / 512 * (2 * BandWidth)

# 线性调频信号
Slfm = np.exp(1j * pi * K * t**2)

# 时域匹配滤波函数
Ht = np.conj(Slfm[::-1])
Hf = np.fft.fftshift(np.fft.fft(Ht, 512))

plt.figure(figsize=(8, 6))
plt.plot(t*1e6, np.real(Ht), 'r-', label='实部')
plt.plot(t*1e6, np.imag(Ht), 'b-.', label='虚部')
plt.title('线性调频信号匹配滤波函数h(t)')
plt.xlabel('时延/μs')
plt.ylabel('幅度')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(f*1e-6, np.abs(Hf))
plt.title('线性调频信号匹配滤波器H(f)')
plt.xlabel('频率/MHz')
plt.ylabel('幅度')
plt.grid(True)
plt.show()

# 产生回波信号
Echo = np.zeros(int(PRT * Fs), dtype=complex)
DelayNumber = int(2 * Ru / C * Fs)
Echo[DelayNumber:DelayNumber+len(Slfm)] = Slfm

# 修正窗函数处理 - 严格按MATLAB
Echo_fft = np.fft.fft(Echo, 2048)

# 创建窗函数矩阵 - 修正：使用正确的维度
window = np.column_stack([
    np.ones(nTe),
    signal.windows.taylor(nTe, nbar=5, sll=35, norm=False)
])

# 修正：分别处理每个窗
Hf_s_list = []
for i in range(window.shape[1]):
    Ht_windowed = Ht * window[:, i]
    Hf_s = np.fft.fft(Ht_windowed, 2048)
    Hf_s_list.append(Hf_s)

Hf_s_matrix = np.column_stack(Hf_s_list)

# 脉压处理
Echo_temp = np.fft.ifft(Echo_fft.reshape(-1, 1, order='F') * Hf_s_matrix, axis=0)
Echo_pc = Echo_temp[nTe:nTe+nTr, :]

# 计算加窗损失
PC_max = np.max(20*np.log10(np.abs(Echo_pc)), axis=0)
PC_lose = PC_max - PC_max[0]
print(f"加窗损失: {PC_lose}")

# 显示脉压结果 - 严格按MATLAB
Slfm_pc = 20*np.log10(np.abs(Echo_pc[DelayNumber-nTe//2:DelayNumber+nTe//2, :]))
plt.figure(figsize=(8, 6))
plt.plot(Slfm_pc[:, 0] - PC_max[0], label='不加窗')
plt.plot(Slfm_pc[:, 1] - PC_max[0], 'r', label='加泰勒窗')
plt.xlabel('时延/μs')
plt.ylabel('幅度/dB')
plt.title('回波信号归一化脉压结果')
plt.legend()
plt.grid(True)
plt.axis([0, 320, -60, 0])
plt.show()

# （6）搜索状态仿真 - 严格按MATLAB
print("进行搜索状态仿真...")
V = Vs * np.cos((alpha + beta) / 180 * pi)

# 信号经过ADC
Signal_ad = 256 * (Echo / np.max(np.abs(Echo)))

# 64个周期回波 - 修正长度匹配
total_samples = N_CI * nTr
Signal_N = np.tile(Signal_ad, N_CI)
t_N = np.arange(0, total_samples) / Fs
Signal_N = Signal_N * np.exp(1j * 2 * pi * (2 * V / lamda) * t_N)

# 噪声信号 - 严格按MATLAB的normrnd
Noise_N = (1/np.sqrt(2)) * (np.random.normal(0, 1024, total_samples) +
                           1j * np.random.normal(0, 1024, total_samples))
Echo_N = Signal_N + Noise_N
Echo_N = Echo_N.reshape(nTr, N_CI, order='F')

plt.figure(figsize=(8, 6))
plt.plot(np.abs(Echo_N))
plt.title('原始信号')
plt.xlabel('时域采样点')
plt.ylabel('幅度(模值)')
plt.grid(True)
plt.show()

# 频域脉压 - 使用不加窗的匹配滤波器
Echo_N_fft = np.fft.fft(Echo_N, 2048, axis=0)
Hf_N = np.fft.fft(Ht, 2048)
Hf_N = np.tile(Hf_N.reshape(-1, 1, order='F'), (1, N_CI))

Echo_N_temp = np.fft.ifft(Echo_N_fft * Hf_N, axis=0)
Echo_N_pc = Echo_N_temp[nTe:nTe+nTr, :]

plt.figure(figsize=(8, 6))
for i in range(N_CI):
    plt.plot((np.arange(nTr))/Fs * C/2 * 1e-3, 20*np.log10(np.abs(Echo_N_pc[:, i])))
plt.title('回波信号脉压结果')
plt.xlabel('距离单元/km')
plt.ylabel('幅度/dB')
plt.grid(True)
plt.show()

# MTD处理
Echo_N_mtd = np.fft.fftshift(np.fft.fft(Echo_N_pc.T, axis=0), axes=0)

# 3D显示 - 严格按MATLAB的mesh
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
X_mtd, Y_mtd = np.meshgrid(np.arange(nTr)/Fs * C/2 * 1e-3, np.arange(-32, 32)/PRT/64)
surf = ax.plot_surface(X_mtd, Y_mtd, np.abs(Echo_N_mtd), cmap='jet', alpha=0.8)
ax.set_xlabel('距离/km')
ax.set_ylabel('多谱勒频率/Hz')
ax.set_zlabel('幅度')
plt.title('64脉冲相干积累结果')
plt.show()

# 等高线图
plt.figure(figsize=(8, 6))
plt.contour(np.arange(nTr)/Fs * C/2 * 1e-3, np.arange(-32, 32)/PRT/64, np.abs(Echo_N_mtd), colors='blue')
plt.xlabel('距离/km')
plt.ylabel('多谱勒频率/Hz')
plt.title('64脉冲相干积累等高线图')
plt.grid(True)
plt.show()

# 找到最大值
index_i, index_j = np.unravel_index(np.argmax(np.abs(Echo_N_mtd)), Echo_N_mtd.shape)
V_fd = 2 * V / lamda
mtd_fd = (index_i - 1) / PRT / 64

SNR_echo = 20 * np.log10(256 / 1024)
SNR_pc = SNR_echo + 10 * np.log10(BandWidth * TimeWidth)
SNR_ci = SNR_pc + 10 * np.log10(64)

print(f"目标多普勒频率: {V_fd:.2f} Hz")
print(f"检测到多普勒频率: {mtd_fd:.2f} Hz")
print(f"原始SNR: {SNR_echo:.2f} dB")
print(f"脉压后SNR: {SNR_pc:.2f} dB")
print(f"相干积累后SNR: {SNR_ci:.2f} dB")

# 恒虚警处理 - 严格按MATLAB
N_mean = 8
N_baohu = 4
K0_CFAR = (1/P_fa)**(1/N_mean) - 1

CFAR_data = np.abs(Echo_N_mtd[index_i, :])
K_CFAR = K0_CFAR / N_mean * np.concatenate([np.ones(N_mean//2), np.zeros(N_baohu+1), np.ones(N_mean//2)])

CFAR_noise = np.convolve(CFAR_data, K_CFAR, mode='valid')
# 修正：确保长度匹配
valid_length = len(CFAR_data) - len(K_CFAR) + 1
CFAR_noise = CFAR_noise[:valid_length]

plt.figure(figsize=(8, 6))
start_idx = (N_mean + N_baohu) // 2
plt.plot(np.arange(start_idx, start_idx+len(CFAR_noise))/Fs * C/2 * 1e-3,
         20*np.log10(CFAR_noise), 'r-.', label='恒虚警电平')
plt.plot(np.arange(nTr)/Fs * C/2 * 1e-3, 20*np.log10(CFAR_data), 'b-', label='信号电平')
plt.xlabel('距离/km')
plt.ylabel('幅度/dB')
plt.title('恒虚警处理')
plt.legend()
plt.grid(True)
plt.show()

# (7) 单脉冲测角 - 严格按MATLAB
print("进行单脉冲测角分析...")
theta = np.linspace(-theta_3dB, theta_3dB, 1200) * pi / 180

Ftheta1 = np.exp(-2.778 * ((theta - theta_3dB/2 * pi/180)**2) / (theta_3dB * pi/180)**2)
Ftheta2 = np.exp(-2.778 * ((theta + theta_3dB/2 * pi/180)**2) / (theta_3dB * pi/180)**2)

Fsum = Ftheta1 + Ftheta2
Fdif = Ftheta1 - Ftheta2
Ferr = np.real(Fsum * np.conj(Fdif)) / (Fsum * np.conj(Fsum))

plt.figure(figsize=(8, 6))
plt.plot(theta*180/pi, Ftheta1, 'r-.', label='波束1')
plt.plot(theta*180/pi, Ftheta2, 'b-.', label='波束2')
plt.plot(theta*180/pi, Fsum, 'r', label='和波束')
plt.plot(theta*180/pi, Fdif, 'b', label='差波束')
plt.plot(theta*180/pi, Ferr, 'k', label='误差信号')
plt.xlabel('角度/度')
plt.ylabel('幅度')
plt.title('和差波束')
plt.legend()
plt.grid(True)
plt.show()

# 计算误差信号斜率
K_theta = np.polyfit(theta[1100:1300]*180/pi, Ferr[1100:1300], 1)
print(f"误差信号斜率为: {1/K_theta[0]:.4f}")

# 计算偏离电轴中心时的归一化误差信号
theta_pianli = np.array([0.5, 1])
Ferr_pinli = theta_pianli * K_theta[0]
print(f"0.5°，1°的误差信号为: {Ferr_pinli[0]:.4f}, {Ferr_pinli[1]:.4f}")

# Monte Carlo分析 - 严格按MATLAB
print("进行Monte Carlo分析...")
SNR_MC = np.arange(5, 31)
N_MC = 100

Ftheta1_MC = np.exp(-2.778 * ((0 - theta_3dB/2 * pi/180)**2) / (theta_3dB * pi/180)**2)
Ftheta2_MC = np.exp(-2.778 * ((0 + theta_3dB/2 * pi/180)**2) / (theta_3dB * pi/180)**2)

Fsum_MC = np.tile(Ftheta1_MC + Ftheta2_MC, (N_MC, len(SNR_MC)))
Fdif_MC = np.tile(Ftheta1_MC - Ftheta2_MC, (N_MC, len(SNR_MC)))

Nsum_MC = (1/np.sqrt(2)) * (np.random.normal(0, 1, (N_MC, len(SNR_MC))) +
                             1j * np.random.normal(0, 1, (N_MC, len(SNR_MC)))) * 10**(-SNR_MC/20)
Ndif_MC = (1/np.sqrt(2)) * (np.random.normal(0, 1, (N_MC, len(SNR_MC))) +
                             1j * np.random.normal(0, 1, (N_MC, len(SNR_MC)))) * 10**(-SNR_MC/20)

Echo_sum_MC = Fsum_MC + Nsum_MC
Echo_dif_MC = Fdif_MC + Ndif_MC
theta_MC = np.real(Echo_sum_MC * np.conj(Echo_dif_MC)) / (Echo_sum_MC * np.conj(Echo_sum_MC))
theta_guji = theta_MC / K_theta[0]

plt.figure(figsize=(8, 6))
for i in range(N_MC):
    plt.plot(SNR_MC, theta_guji[i, :], '.', alpha=0.3)
plt.grid(True)
plt.xlabel('SNR/dB')
plt.ylabel('单词测量误差/度')
plt.title('SNR与单次测角误差')
plt.show()

std_theta_wucha = np.std(theta_guji, axis=0)
plt.figure(figsize=(8, 6))
plt.plot(SNR_MC, std_theta_wucha)
plt.grid(True)
plt.title('均方根误差')
plt.xlabel('SNR/dB')
plt.ylabel('均方根误差/°')
plt.show()

# 和差通道时域脉压 - 严格按MATLAB
theta_hecha = 1
SNR = 20

Ftheta1_hecha = np.exp(-2.778 * ((theta_hecha*pi/180 - theta_3dB/2*pi/180)**2) / (theta_3dB*pi/180)**2)
Ftheta2_hecha = np.exp(-2.778 * ((theta_hecha*pi/180 + theta_3dB/2*pi/180)**2) / (theta_3dB*pi/180)**2)

Signal_hecha = np.zeros(int(PRT * Fs), dtype=complex)
DelayNumber_hecha = int(2 * Rs / C * Fs)
Signal_hecha[DelayNumber_hecha:DelayNumber_hecha+len(Slfm)] = Slfm
Signal_hecha = Signal_hecha / np.max(np.abs(Signal_hecha))

Echo_boshu1 = Signal_hecha + (1/np.sqrt(2)) * (np.random.normal(0, 1, nTr) +
                                               1j * np.random.normal(0, 1, nTr)) * 10**(-SNR/20)
Echo_boshu2 = Signal_hecha + (1/np.sqrt(2)) * (np.random.normal(0, 1, nTr) +
                                               1j * np.random.normal(0, 1, nTr)) * 10**(-SNR/20)

Echo_sum_hecha = Ftheta1_hecha * Echo_boshu1 + Ftheta2_hecha * Echo_boshu2
Echo_dif_hecha = Ftheta1_hecha * Echo_boshu1 - Ftheta2_hecha * Echo_boshu2

Echo_sum_pc = np.convolve(Echo_sum_hecha, Ht, mode='same')
Echo_dif_pc = np.convolve(Echo_dif_hecha, Ht, mode='same')

Echo_err = np.real(Echo_sum_pc * np.conj(Echo_dif_pc)) / (Echo_sum_pc * np.conj(Echo_sum_pc))

plt.figure(figsize=(8, 6))
plt.plot((np.arange(len(Echo_sum_pc)))/Fs * C/2 * 1e-3, 20*np.log10(np.abs(Echo_sum_pc)))
plt.title('弹目距离20Km时和通道时域脉压结果')
plt.xlabel('距离/Km')
plt.ylabel('幅度/dB')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot((np.arange(len(Echo_dif_pc)))/Fs * C/2 * 1e-3, 20*np.log10(np.abs(Echo_dif_pc)))
plt.title('弹目距离20Km时差通道时域脉压结果')
plt.xlabel('距离/Km')
plt.ylabel('幅度/dB')
plt.grid(True)
plt.show()

print(f"设定20km处方位角{theta_hecha}°，误差信号为 {Echo_err[266]:.4f}，测量值为 {Echo_err[266]*(1/K_theta[0]):.4f}°")

# (8) 中频正交采样 - 完整实现所有图形
print("进行中频正交采样分析...")
F_if = 60e6
M_ad = 3
Fs_ad = 4 * F_if / (2 * M_ad - 1)
BandWidth_track = 10e6
TimeWidth_track = 10e-6
nTe_track = int(TimeWidth_track * Fs_ad)
nTe_track = nTe_track + (nTe_track % 2)

t_track = np.linspace(-nTe_track/2, nTe_track/2-1, nTe_track) / nTe_track * TimeWidth_track

# 跟踪时线性调频信号
Slfm_track = np.cos(2 * pi * (F_if * t_track + BandWidth_track/TimeWidth_track/2 * t_track**2))

# 符号修正
Modify = np.array([[1, -1, -1, 1], [1, 1, -1, -1]])
Slfm_track = Slfm_track * np.tile(Modify[M_ad % 2, :], nTe_track // 4)

# 低通滤波器设计
f_lowpass = [BandWidth_track, 1.2 * BandWidth_track]
a_lowpass = [1, 0]
Rp_lowpass = 1
Rs_lowpass = 40

# 使用firpmord等效函数计算滤波器参数
n_lowpass, wn = signal.buttord(f_lowpass[0]/(Fs_ad/2), f_lowpass[1]/(Fs_ad/2), Rp_lowpass, Rs_lowpass)
n_lowpass = n_lowpass + 1 - (n_lowpass % 2)
h_lowpass = signal.firwin(n_lowpass, wn, window='hamming')

# 低通滤波器响应图
plt.figure(figsize=(10, 8))
w, h_response = freqz(h_lowpass, 1, 1024, fs=Fs_ad)
plt.subplot(2, 1, 1)
plt.plot(w, 20*np.log10(np.abs(h_response)))
plt.title('低通滤波器响应')
plt.ylabel('幅度/dB')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(w, np.angle(h_response))
plt.xlabel('频率/Hz')
plt.ylabel('相位/弧度')
plt.grid(True)
plt.tight_layout()
plt.show()

# IQ分解
I_track = signal.lfilter(h_lowpass[1::2], 1, Slfm_track[::2])
Q_track = signal.lfilter(h_lowpass[::2], 1, Slfm_track[1::2])
Sig_track = I_track[n_lowpass//2:] + 1j * Q_track[n_lowpass//2:]

# IQ正交性图 - 第一个
plt.figure(figsize=(6, 6))
plt.plot(np.real(Sig_track), np.imag(Sig_track), '.', markersize=2)
plt.axis([-0.6, 0.6, -0.6, 0.6])
plt.axis('equal')
plt.grid(True)
plt.xlabel('I通道')
plt.ylabel('Q通道')
plt.title('IQ正交性')
plt.show()

# 产生中频正交采样信号
R_target = 10000
SNR_IQ = 20

Slfmexp_track = np.exp(1j * 2 * pi * (F_if * t_track + BandWidth_track/TimeWidth_track/2 * t_track**2))
Ht_track = np.conj(Slfmexp_track[::-1])

Echo = np.zeros(int(PRT * Fs_ad))
DelayNumber = int(2 * R_target / C * Fs_ad)
Echo[DelayNumber:DelayNumber+len(Slfm_track)] = Slfm_track

# 目标回波中频信号图
plt.figure(figsize=(8, 6))
plt.plot(20*np.log10(np.abs(Echo)))
plt.grid(True)
plt.axis([3200, 3700, -40, 0])
plt.xlabel('距离单元')
plt.ylabel('幅度/dB')
plt.title('目标回波中频信号')
plt.show()

# 正交采样
cos_I = np.cos(2 * pi * F_if / Fs_ad * np.arange(len(Echo)))
sin_Q = np.sin(2 * pi * F_if / Fs_ad * np.arange(len(Echo)))

Echo_I = Echo * cos_I
Echo_Q = -Echo * sin_Q

I_track = signal.lfilter(h_lowpass[1::2], 1, Echo_I)
Q_track = signal.lfilter(h_lowpass[::2], 1, Echo_Q)

# 正交采样I/Q通道基带信号图
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(20*np.log10(np.abs(I_track)))
plt.axis([3200, 3700, -40, 0])
plt.ylabel('幅度/dB')
plt.title('正交采样I通道基带信号')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(20*np.log10(np.abs(Q_track)))
plt.axis([3200, 3700, -40, 0])
plt.xlabel('距离单元')
plt.ylabel('幅度/dB')
plt.title('正交采样Q通道基带信号')
plt.grid(True)
plt.tight_layout()
plt.show()

Echo_IQ_track = I_track[n_lowpass//2:] + 1j * Q_track[n_lowpass//2:]

# IQ正交性图 - 第二个（回波信号）
plt.figure(figsize=(6, 6))
plt.plot(np.real(Echo_IQ_track), np.imag(Echo_IQ_track), '.', markersize=2)
plt.axis('equal')
plt.grid(True)
plt.xlabel('I通道')
plt.ylabel('Q通道')
plt.title('IQ正交性')
plt.show()

# 脉冲压缩
Echo_IQ_track_pc = np.convolve(Echo_IQ_track, Ht_track, mode='same')

# 基带信号脉压处理图
plt.figure(figsize=(8, 6))
t_axis = np.arange(len(Echo_IQ_track_pc)) / Fs_ad * C/2 * 1e-3
plt.plot(t_axis, 20*np.log10(np.abs(Echo_IQ_track_pc)))
plt.xlabel('距离/km')
plt.ylabel('幅度/dB')
plt.title('基带信号脉压处理')
plt.axis([8, 12, -10, 50])
plt.grid(True)
plt.show()

print("雷达系统设计分析完成！")
