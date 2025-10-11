#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 20:40:48 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=Mzk0NTQzNDQyMw==&mid=2247489087&idx=1&sn=b00d1bc595f6fdb94ab9a12aac2c42d2&chksm=c289c8be7681e12f7c5ac04b9473a73b1dac5ad289e9d3683f2654387003a407f935a02ae0d8&mpshare=1&scene=1&srcid=1008mC0aEXQrkjOCTyuedYeM&sharer_shareinfo=33824049d5c25c01d294429ebf4f499f&sharer_shareinfo_first=33824049d5c25c01d294429ebf4f499f&exportkey=n_ChQIAhIQPMplLfSY8Sr8ZPPq8MdXHRKfAgIE97dBBAEAAAAAACauLMpdKCEAAAAOpnltbLcz9gKNyK89dVj00JduiLNh4Y1bB1%2FfGmxyXduWuu3yVI6RaB4FCO06NdaWZNTvRwU3HKNJk6pTX40YZAjCOA0LB4NYyvV2ihHOC%2F65moXeVMtnbNqcBwQdYPkWjwZUfEA55pffCchcx464g9FnMdch43DPYrsAR9oHhDNRa5gB4qi1yZ3me4a3jq0YvNisXzvlpst8FqhprBHjr1QDH9PRKKnelzSybLCYNAGOPf2Lvj6nFavhYyPnhrdvBnp7R7zQgxgCnu%2FDHAqjIHn3MjX1lc7ZSDHfFMEW4XQikzkyuPJCisXKi8UBqS%2FI9jsgHV9SLp8e1KOWP%2BGbdzeF4pnSKR5o&acctmode=0&pass_ticket=%2B%2B5ib3wMkIrwLNE6olYu%2FfzO4m%2BX0b8OOUXhYwP66R2TZfXj7KPKII%2B8SqP6TOy3&wx_header=0#rd


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 16  # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 16  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '--'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 12


# 参数设置
fs = 100  # 采样率
tp = 0.2  # 脉冲宽度
Tr = 1    # 脉冲重复周期
M = 10

# 时间序列
t1 = np.arange(-Tr/2, Tr/2, 1/fs)
t = np.arange(-M*Tr/2, M*Tr/2, 1/fs)
N = len(t)

# 初始化信号
u = np.zeros(len(t))
s1 = np.zeros(len(t))

# 生成相参脉冲串
for i in range(M):
    # 创建矩形脉冲
    pulse = np.zeros(len(t))
    # 计算脉冲的位置
    start_idx = int((i*Tr - tp/2 - t[0]) * fs)
    end_idx = int((i*Tr + tp/2 - t[0]) * fs)

    # 确保索引在有效范围内
    start_idx = max(0, start_idx)
    end_idx = min(len(t), end_idx)

    if start_idx < end_idx:
        pulse[start_idx:end_idx] = 1

    u = u + pulse

# 图1: 脉冲串信号
plt.figure(1)
plt.plot(t, u)
plt.title('相参脉冲串信号')
plt.xlabel('时间/s')
plt.ylabel('幅度')
plt.grid(True)

# 模糊函数计算
fa_i = np.linspace(-1/tp, 1/tp, N)  # 多普勒频移序列
tao_i = np.linspace(-M*Tr, M*Tr, N)  # 时域延时序列
Tao_i, Fa_i = np.meshgrid(tao_i, fa_i)

f = np.arange(-N/2, N/2) * fs / N
U = np.fft.fftshift(np.fft.fft(u, N))  # 信号FFT

U1 = np.zeros((N, N), dtype=complex)
u1 = np.zeros((N, N), dtype=complex)
U_fin = np.zeros((N, N), dtype=complex)
u_fin = np.zeros((N, N), dtype=complex)

for i in range(N):
    # 应用多普勒频移
    u1[i, :] = u * np.exp(1j * 2 * np.pi * fa_i[i] * t)
    # 计算频域表示
    U1[i, :] = np.fft.fftshift(np.fft.fft(u1[i, :], N))
    # 频域相乘
    U_fin[i, :] = U1[i, :] * np.conj(U)
    # 回到时域得到模糊函数
    u_fin[i, :] = np.fft.fftshift(np.fft.ifft(U_fin[i, :], N))

# 图2: 模糊函数三维图
plt.figure(2, figsize=(10, 8))
ax = plt.axes(projection='3d')
ax.plot_surface(Tao_i, Fa_i, np.abs(u_fin), cmap='viridis')
ax.set_xlabel('时延/s')
ax.set_ylabel('多普勒频移/Hz')
ax.set_zlabel('幅度')
plt.title('相参脉冲串模糊函数三维图')

# 图3: 模糊函数等值线图
plt.figure(3)
plt.contour(Tao_i, Fa_i, np.abs(u_fin))
plt.xlabel('时延/s')
plt.ylabel('多普勒频移/Hz')
plt.title('相参脉冲串模糊函数等值线图')
plt.grid(True)

# 零多普勒切割面 (零频移切面)
U_2 = U * np.conj(U)
u_2 = np.fft.fftshift(np.fft.ifft(U_2, N))

# 图4: 零频移切面图
plt.figure(4)
plt.plot(t, np.abs(u_2) / np.max(np.abs(u_2)))
plt.grid(True)
plt.xlabel('时延/s')
plt.ylabel('归一化幅度')
plt.title('相参脉冲串零频移切面图')

# 零延时切割面
# 找到零延时的索引 (最接近0的时延)
zero_delay_idx = np.argmin(np.abs(tao_i))

# 图5: 零延时切面图
plt.figure(5)
plt.plot(fa_i, np.abs(u_fin[:, zero_delay_idx]))
plt.grid(True)
plt.xlabel('多普勒频移/Hz')
plt.ylabel('幅度')
plt.title('相参脉冲串零延时切面图')

plt.tight_layout()
plt.show()
