#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 17:34:56 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
# import cv

# def csa_imaging():
# 成像参数
Fs = 32.317e6                  # 采样率, MHz
Fr = 7.2135e11                 # 距离调频率, MHz/us
start = 6.5956e-3              # 数据窗开始时间, ms
Tr = 4.175e-5                  # 脉宽, us
R0 = 9.88646462e5              # 最短斜距, m
f0 = 5.3e9                     # 雷达频率, GHz
lamda = 0.05667                # 雷达波长, m
Fa = 1.25698e3                 # 脉冲重复频率, Hz
Vr = 7062                      # 有效雷达速率, m/s
Kr = 0.72135e12
Ka = 1733                      # 方位调频率, Hz/s
Fc = -6900                     # 多普勒中心频率, Hz
c = 299790000                  # 电磁波传播速度

# 读取数据
# 假设数据保存在CDdata1.mat文件中
mat_data = loadmat('CDdata1.mat')
data = mat_data['data'].astype(np.complex128)  # 将数据转换成complex128型
length_a, length_r = data.shape  # 获得数据的大小

# 时间轴和频率轴设置
T_start = 6.5956e-3  # 数据窗开始时间
tau = np.arange(T_start, T_start + length_r/Fs, 1/Fs)[:length_r]  # 距离向时间

R_ref = (T_start + length_r/Fs)/2 * c  # 参考距离
f_a = np.linspace(-Fa/2 + Fc, Fa/2 + Fc - Fa/length_a, length_a)  # 方位频率
f_r = np.linspace(0, Fs - Fs/length_r, length_r)  # 距离频率

D = np.sqrt(1 - (f_a * lamda / (2 * Vr))**2)  # 距离徙动因子
alpha = 1/D - 1
R = R_ref / D  # 距离多普勒域中更精确的双曲线距离等式
Z = (R0 * c * f_a**2) / (2 * Vr**2 * f0**3 * D**3)
Km = Kr / (1 - Kr * Z)  # 校正后距离脉冲调频率

# STEP1: 方位向傅里叶变换，将基带信号变换到距离多普勒域
data = np.fft.fft(data, axis=0)

# STEP2: 将距离多普勒域的信号与线性变标方程相乘
tau1 = np.tile(tau, (length_a, 1))
tau2 = 2 * np.outer(R, np.ones(length_r)) / c
D_tau = tau1 - tau2
H1 = np.outer(Km * alpha, np.ones(length_r))
Ssc = np.exp(-1j * np.pi * H1 * D_tau**2)  # 线性变标方程
data = data * Ssc  # 校正补余RCM

# STEP3: 距离向傅里叶变换，从距离多普勒域变换到二维频域
data = np.fft.fftshift(np.fft.fft(np.fft.fftshift(data, axes=1), axis=1), axes=1)

# STEP4: 通过一个相位相乘，完成（距离压缩、SRC、一致RCMC）
H_r_1 = 1 / (Km * (1 + alpha))
H_r_1 = np.outer(H_r_1, np.ones(length_r))
H_r_2 = np.tile(f_r**2, (length_a, 1))
H_r = np.exp(-1j * np.pi * H_r_1 * H_r_2)

H_RCMC = np.exp(1j * 4 * np.pi * R_ref * np.outer(alpha, f_r) / c)

data = data * H_r * H_RCMC  # 距离压缩和一致RCMC

# STEP5: 距离向傅里叶逆变换，变回到距离多普勒域
data = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(data, axes=1), axis=1), axes=1)

# STEP6: 完成相位校正+方位向匹配滤波
r_0 = tau / 2 * c
H3_1 = Km * alpha * (1 + alpha) / (c**2)
H3 = np.outer(H3_1, np.ones(length_r))
phi = 4 * np.pi * H3 * np.tile((r_0 - R_ref)**2, (length_a, 1)) / (c**2)
data = data * np.exp(1j * phi)  # 完成相位校正

H_a = np.exp(1j * 4 * np.pi / lamda * np.outer(D - 1, r_0))
data = data * H_a  # 方位向匹配滤波

# STEP7: 完成方位向傅里叶逆变换
data = np.fft.ifft(data, axis=0)
data = np.fft.fftshift(data, axes=1)

# STEP8: 将图像聚焦显示
plt.figure(figsize=(10, 8))
plt.imshow(np.log10(10 * np.abs(data)), cmap='gray')
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.title('CSA Imaging Result')
plt.colorbar()
plt.tight_layout()
plt.show()

    # return data


