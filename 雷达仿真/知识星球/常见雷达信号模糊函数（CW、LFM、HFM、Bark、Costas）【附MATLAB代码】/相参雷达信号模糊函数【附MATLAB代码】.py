#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 20:40:48 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=Mzk0NTQzNDQyMw==&mid=2247489087&idx=1&sn=b00d1bc595f6fdb94ab9a12aac2c42d2&chksm=c289c8be7681e12f7c5ac04b9473a73b1dac5ad289e9d3683f2654387003a407f935a02ae0d8&mpshare=1&scene=1&srcid=1008mC0aEXQrkjOCTyuedYeM&sharer_shareinfo=33824049d5c25c01d294429ebf4f499f&sharer_shareinfo_first=33824049d5c25c01d294429ebf4f499f&exportkey=n_ChQIAhIQPMplLfSY8Sr8ZPPq8MdXHRKfAgIE97dBBAEAAAAAACauLMpdKCEAAAAOpnltbLcz9gKNyK89dVj00JduiLNh4Y1bB1%2FfGmxyXduWuu3yVI6RaB4FCO06NdaWZNTvRwU3HKNJk6pTX40YZAjCOA0LB4NYyvV2ihHOC%2F65moXeVMtnbNqcBwQdYPkWjwZUfEA55pffCchcx464g9FnMdch43DPYrsAR9oHhDNRa5gB4qi1yZ3me4a3jq0YvNisXzvlpst8FqhprBHjr1QDH9PRKKnelzSybLCYNAGOPf2Lvj6nFavhYyPnhrdvBnp7R7zQgxgCnu%2FDHAqjIHn3MjX1lc7ZSDHfFMEW4XQikzkyuPJCisXKi8UBqS%2FI9jsgHV9SLp8e1KOWP%2BGbdzeF4pnSKR5o&acctmode=0&pass_ticket=%2B%2B5ib3wMkIrwLNE6olYu%2FfzO4m%2BX0b8OOUXhYwP66R2TZfXj7KPKII%2B8SqP6TOy3&wx_header=0#rd

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import square

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



# def coherent_pulse_train_ambiguity():
"""
相参脉冲串模糊函数计算
"""
# 参数设置
fs = 100           # 采样率
tp = 0.2           # 脉冲宽度
Tr = 1             # 脉冲重复周期
M = 10             # 脉冲个数

t1 = np.arange(-Tr/2, Tr/2, 1/fs)
t = np.arange(-M*Tr/2, M*Tr/2, 1/fs)
N = len(t)

# 生成相参脉冲串
u = np.zeros(len(t))
for i in range(M):
    # 创建矩形脉冲
    pulse_start = i * Tr - tp/2
    pulse_end = i * Tr + tp/2
    s1 = ((t >= pulse_start) & (t < pulse_end)).astype(float)
    u = u + s1

# 时域图
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(t, u)
ax.set_xlabel('时间/s')
ax.set_ylabel('幅度')
ax.set_title('相参脉冲串时域图')
ax.grid(True)
plt.show()
plt.close()

# 模糊函数计算
fa_i = np.linspace(-1/tp, 1/tp, N)      # 多普勒频移序列
tao_i = np.linspace(-M*Tr, M*Tr, N)     # 时域延时序列
Tao_i, Fa_i = np.meshgrid(tao_i, fa_i)

f = np.arange(-N/2, N/2) * fs / N
U = np.fft.fftshift(np.fft.fft(u, N))   # 信号FFT

u1 = np.zeros((N, N), dtype=complex)
U1 = np.zeros((N, N), dtype=complex)
U_fin = np.zeros((N, N), dtype=complex)
u_fin = np.zeros((N, N), dtype=complex)

for i in range(N):
    u1[i, :] = u * np.exp(1j * 2 * np.pi * fa_i[i] * t)
    U1[i, :] = np.fft.fftshift(np.fft.fft(u1[i, :], N))
    U_fin[i, :] = U1[i, :] * np.conj(U)
    u_fin[i, :] = np.fft.fftshift(np.fft.ifft(U_fin[i, :], N))

# 3D模糊函数图
fig = plt.figure(figsize=(8, 8), constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')
cbar = ax.plot_surface(Tao_i, Fa_i, np.abs(u_fin), rstride=2, cstride=2, cmap=plt.get_cmap('jet'))
# plt.colorbar(cbar)
ax.set_xlabel('时延/s')
ax.set_ylabel('多普勒频移/Hz')
ax.set_zlabel('相参脉冲串模糊函数三维图')
ax.grid(False)
ax.set_proj_type('ortho')
plt.show()
plt.close()

# 模糊函数等值线图
fig, ax = plt.subplots(figsize=(8, 6))
ax.contour(Tao_i, Fa_i, np.abs(u_fin))
ax.set_xlabel('时延/s')
ax.set_ylabel('多普勒频移/Hz')
ax.set_title('相参脉冲串模糊函数等值线图')
ax.grid(True)
plt.show()
plt.close()

# 零多普勒切面图
U_2 = U * np.conj(U)
u_2 = np.fft.fftshift(np.fft.ifft(U_2, N))

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(t, np.abs(u_2) / np.max(np.abs(u_2)))
ax.grid(True)
ax.set_xlabel('时延/s')
ax.set_ylabel('归一化幅度')
ax.set_title('相参脉冲串零频移切面图')
plt.show()
plt.close()

# 零延时切面图
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fa_i, np.abs(u_fin[:, N//2]))
ax.grid(True)
ax.set_xlabel('多普勒频移/Hz')
ax.set_ylabel('幅度')
ax.set_title('相参脉冲串零延时切面图')
plt.show()
plt.close()

# # 调用函数
# if __name__ == "__main__":
#     coherent_pulse_train_ambiguity()











