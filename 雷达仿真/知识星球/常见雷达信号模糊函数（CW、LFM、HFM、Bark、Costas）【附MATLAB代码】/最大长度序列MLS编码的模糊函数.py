#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 20:35:50 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=Mzg3ODkwOTgyMw==&mid=2247488206&idx=2&sn=2d0c24c91fa336cd3f1335bfdf6b2913&chksm=cf0dd2def87a5bc8321e6f07497229af7b0f918b291d3590bdf4672403c28d4c75c9d508dda5&cur_album_id=3692626176607780876&scene=189#wechat_redirect

理论计算的模糊函数
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 16  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 16  # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 16  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 16  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 16  # 设置 y 轴刻度字体大小
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
plt.rcParams['legend.fontsize'] = 16

"""
伪随机编码的模糊函数图
"""
u31 = np.array([1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1])   # 31位
u15 = np.array([1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1])

N = len(u31)
tau = N
samp_num = N * 10
n = np.ceil(np.log(samp_num) / np.log(2))
nfft = int(2 ** n)

# 生成伪随机编码波形
u = np.zeros(nfft)
j = 0
for index in range(0, samp_num, 10):
    u[index:index+10] = u31[j]
    j += 1

v = u.copy()
delay = np.linspace(0, 5 * tau, nfft)
freq_del = 8 / tau / 100

# 计算模糊函数
vfft = np.fft.fft(v, nfft)
freq_vals = np.arange(-4/tau, 4/tau + freq_del, freq_del)
ambig = np.zeros((nfft, len(freq_vals)))

j = 0
for freq in freq_vals:
    exf = np.exp(1j * 2 * np.pi * freq * delay)
    u_times_exf = u * exf
    ufft = np.fft.fft(u_times_exf, nfft)
    prod = ufft * np.conj(vfft)
    ambig[:, j] = np.fft.fftshift(np.abs(np.fft.ifft(prod)))
    j += 1

# 归一化
ambig_norm = ambig / np.max(ambig)

# 重新定义延迟轴
delay = np.linspace(-N, N, nfft)

# 1. 3D模糊函数图
fig = plt.figure(figsize=(8, 8), constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(freq_vals, delay)
cbar = ax.plot_surface(X, Y, ambig_norm, rstride=2, cstride=2, cmap='jet')
# plt.colorbar(cbar)
ax.set_xlabel('频率')
ax.set_ylabel('时延')
ax.set_zlabel('PRN编码模糊函数')
ax.set_title(f'{N}位伪随机编码模糊函数')
ax.grid(False)
ax.set_proj_type('ortho')
plt.show()
plt.close()

# 2. 距离模糊函数图
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(delay, ambig_norm[:, 50], 'k')  # 使用索引50接近中心频率
ax.set_xlabel('时延')
ax.set_ylabel('f=0归一化模糊切片')
ax.set_title('距离模糊函数图')
ax.grid(True)
ax.set_xlim([delay[0], delay[-1]])
plt.show()
plt.close()

# 3. 多普勒模糊函数图
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(freq_vals, ambig_norm[nfft//2, :], 'k')  # 使用中心时延
ax.set_xlabel('频率')
ax.set_ylabel('t=0归一化模糊切片')
ax.set_title('多普勒模糊函数图')
ax.grid(True)
ax.set_xlim([freq_vals[0], freq_vals[-1]])
plt.show()
plt.close()

# 4. 等高图
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contour(freq_vals, delay, ambig_norm, levels = 10, cmap = 'rainbow')
ax.set_xlabel('频率')
ax.set_ylabel('时延')
ax.set_title('等高图')
ax.set_xlim([freq_vals[0], freq_vals[-1]])
ax.set_ylim([delay[0], delay[-1]])
plt.show()
plt.close()




