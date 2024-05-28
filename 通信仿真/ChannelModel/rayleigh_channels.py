#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:23:50 2024

@author: jack

"""


#%% https://blog.csdn.net/qq_45889056/article/details/135660662

import numpy as np

def generate_rayleigh_channels(num_channels):
# 生成实部和虚部
    real_parts = np.random.normal(0, 1/np.sqrt(2), num_channels)
    imag_parts = np.random.normal(0, 1/np.sqrt(2), num_channels)
    # 合成复数衰落系数数组
    h = real_parts + 1j * imag_parts
    return h
# 生成多个瑞利信道衰落系数
num_channels = 10
h_array = generate_rayleigh_channels(num_channels)
# print("Generated Rayleigh channel fading coefficients:", h_array)


#%%   https://blog.csdn.net/qq_45889056/article/details/134133182
#其中p为第K个用户的发射功率，g为第n个用户的大尺度信道增益，包括路径损耗和阴影，
#h~CN( 0 , 1)为第K个用户的瑞利衰落系数，N0为噪声功率谱密度。
def large_small(K,d,p,W): #
    N0_dBm = 3.981e-21  # -174dBm==3.981e-21
    path_loss = 128.1 + 37.6*np.log10(d/1000) #dB    小区半径为500m
    # the shadowing factor is set as 6 dB.
    shadow_factor = 6 #dB
    h_large=10** (-(path_loss + shadow_factor) / 10)

    sigma = np.sqrt(1 / 2)
    h_small = sigma * np.random.randn(K) + 1j * sigma * np.random.randn(K)

    h = abs(h_small)**2
    snr = h_large*h*p / (N0_dBm*W)
    return snr

# if __name__ == '__main__':
#     snr  = large_small(1,2000, 0.1, 10000000) # W设置为180KHz
#     snr_dB  = 10*np.log10(snr/10)
#     print(snr)
#     print( snr_dB)


#%% GPT
import numpy as np

# 设置参数
bandwidth = 180e3  # 带宽，单位为Hz
noise_power_density = 10**((-170)/10)  # 噪声功率谱密度，单位为dBm/Hz转换为W/Hz

# 计算大尺度衰落
def large_scale_fading(distance):
    path_loss = 20 * np.log10(distance)  # 路径损耗，单位为dB
    shadowing = np.random.normal(0, 2)  # 阴影衰落，假设服从均值为0，标准差为2的正态分布
    return path_loss + shadowing

# 计算小尺度衰落
def small_scale_fading(num_paths):
    channel_gain = np.sqrt(np.random.rayleigh(1, num_paths))  # 小尺度衰落，假设服从瑞利分布
    return channel_gain

# 设置传播距离
distance = 100  # 假设传播距离为100米

# 计算大尺度衰落
large_scale_gain = 10**(large_scale_fading(distance)/10)  # 转换为线性增益

# 计算小尺度衰落
num_paths = 4  # 假设有4个传播路径
small_scale_gain = small_scale_fading(num_paths)

# 计算信道增益矩阵
thermal_noise_power = noise_power_density * bandwidth  # 计算噪声功率
channel_gain_matrix = np.sqrt(large_scale_gain) * small_scale_gain / np.sqrt(thermal_noise_power)

print("信道增益矩阵：")
print(channel_gain_matrix)


#%% GPT
import numpy as np

# 参数设置
d_0 = 1.0  # 参考距离 (米)
f = 2.4e9  # 载波频率 (Hz)
c = 3e8    # 光速 (m/s)
n = 3.0    # 路径损耗指数，通常在2到4之间。
d = 100.0  # 发射机和接收机之间的距离 (米)

# 噪声功率谱密度和带宽
N0_dBm_per_Hz = -170  # 噪声功率谱密度 (dBm/Hz)
bandwidth = 180e3     # 带宽 (Hz)

# 计算噪声功率
N0 = 10**(N0_dBm_per_Hz / 10) * 1e-3  # 转换到瓦特/Hz
Pn = N0 * bandwidth  # 总噪声功率 (瓦特)

# 计算路径损耗 (dB)
PL_d0 = 20 * np.log10(4 * np.pi * d_0 * f / c)
PL_d = PL_d0 + 10 * n * np.log10(d / d_0)

# 转换路径损耗到线性尺度
L = 10**(-PL_d / 10)

# 发射天线和接收天线数量
Nt = 2  # 发射天线数量
Nr = 2  # 接收天线数量

# 生成瑞利衰落矩阵
F = (1/np.sqrt(2)) * (np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt))

# 计算信道增益矩阵
H = L * F

# 计算信噪比 (SNR) 和信道容量
Pt = 1  # 假设发射功率为 1 瓦特
SNR = (Pt * np.abs(H)**2) / Pn
capacity = np.sum(np.log2(1 + SNR))  # 信道容量 (比特/秒/赫兹)

# 输出结果
print("噪声功率 (瓦特):", Pn)
print("路径损耗 (dB):", PL_d)
print("路径损耗因子 (线性):", L)
print("信道增益矩阵 H:")
print(H)
print("信道容量 (比特/秒/赫兹):", capacity)




