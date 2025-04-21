#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 19:43:36 2025

@author: jack

单目标，直接利用解析形式的差频信号，利用FFT矩阵进行结算.

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

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


def FFTmatrix(N):
     M = np.zeros((N, N), dtype = complex)
     tmp = np.arange(N)
     for n in range(N):
         M[n, :] =  np.exp(-1j*2*np.pi*n*tmp/N)
     return M

c = 3e8
Rmax  = 25     # Maximum Unambiguous Range in meters
Vmax = 20      # Maximum Velocity in m/s
Nchirps = 128  # Number of Chirps
Ns = 256       # Number of Samples per Chirp
B = 1.5e9      # Sweep Bandwidth in Hz
f0 = 78e9      # Carrier Frequency in Hz
Tc = 50e-6     # Chirp Duration in seconds
TRRI = 60e-6   # Ramp Repetion Interval
fs = 5e6       # ADC Sampling Rate in Hz
Ts = 1/fs      #
S = B/Tc

tarR = 5
tarV = 10

R_max = (fs * c)/(2*S)
v_max = c/(4 * Tc * f0)
print(f"tarR = {tarR}, R_max = {R_max}, tarV = {tarV}, v_max = {v_max}")

Rx = np.zeros((Ns, Nchirps), dtype = complex)
max_prop = 3 * Rmax/c

## get IF signal
for nc in range(Nchirps):
    Sam_idx = np.arange(Ns)
    t = max_prop + Sam_idx * Ts
    phase_Tx = 2 * np.pi * f0 * t + np.pi * S * t**2
    tau = 2*(tarR + tarV * t + tarV * TRRI * nc)/c
    phase_Rx = 2 * np.pi * f0 *(t - tau) + np.pi * S * (t-tau)**2
    phase_IF = phase_Tx - phase_Rx
    Rx[:,nc] = np.cos(phase_IF) + 1j * np.sin(phase_IF)

signal_power = np.sum(np.abs(Rx)**2)/Rx.size
SNRdB = 20
noise_pow = signal_power / (10.0**(SNRdB/10.0))

X = Rx + np.sqrt(noise_pow/2) * (np.random.randn(*Rx.shape) + 1j * np.random.randn(*Rx.shape))

## DFT Signal Processing
F_n = (1/np.sqrt(Ns))*FFTmatrix(Ns)              # DFT matrix with N*N
F_l = (1/np.sqrt(Nchirps))*FFTmatrix(Nchirps)    # DFT matrix with L*L
X_2d =  F_n  @ X @ (F_l.T)                       # The expression for finding the 2D DFT

## Range Resolution
del_R = (Tc * c) / (2 * B * Ts * Ns)

## Velocity Resolution
del_v = c / (2 * f0 * TRRI * Nchirps)

## Range and velocity limits
R_axis = np.arange(0, Ns/2 * del_R, del_R)
v_axis = np.arange(-del_v*Nchirps/2, del_v*Nchirps/2, del_v)
v_grid, R_grid = np.meshgrid(v_axis, R_axis)

## Accessing only N/2 DFT coeffients
X_2d = X_2d[:int(Ns/2),:] #   (1:radar_parameters.N/2, :);

## Swap columns with defined limits
tmp = X_2d[:, :int(Nchirps/2)].copy()
X_2d[:, :int(Nchirps/2)] = X_2d[:, int(Nchirps/2):]
X_2d[:, int(Nchirps/2):] = tmp

## Frequency grids for plotting
k_axis = np.arange(Ns/2)
p_axis = np.arange(-Nchirps/2, Nchirps/2, 1)
p_grid, k_grid = np.meshgrid(p_axis, k_axis);

## Plot magnitude of 2D DFT
fig = plt.figure(figsize=(10, 10) )
ax1 = fig.add_subplot(111, projection = '3d')
ax1.plot_surface(p_grid, k_grid, np.abs(X_2d), rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
ax1.grid(False)
ax1.set_proj_type('ortho')
ax1.set_xlabel('距离门数', )
ax1.set_ylabel('脉冲数', )
ax1.set_title('Magnitude', )
ax1.view_init(azim = -135, elev = 30)
plt.show()
plt.close()

## Plot magnitude of 2D DFT
fig = plt.figure(figsize=(10, 10) )
ax1 = fig.add_subplot(111, projection = '3d')
ax1.plot_surface(v_grid, R_grid, np.abs(X_2d), rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
ax1.grid(False)
ax1.set_proj_type('ortho')
ax1.set_xlabel('Velocity (m/s)', )
ax1.set_ylabel('Range (m)', )
ax1.set_title('Magnitude', )
ax1.view_init(azim = -135, elev = 30)
plt.show()
plt.close()
















































