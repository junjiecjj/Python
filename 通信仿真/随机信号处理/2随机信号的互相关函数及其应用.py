#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:12:47 2025

@author: jack
"""

import scipy
import numpy as np
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

## my lib
from Xcorrs import xcorr, correlate_maxlag, correlate_template, get_lags

filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

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



#%% 2.3 互相关函数在信号处理中的应用

fs = 2000     # 采样频率
t = np.arange(0, 1, 1/fs)  #时间向量
c = 1e4       # 光速，假设信号以光速传播
d = 2000      # 目标实际距离（米）
delay = 2*d/c # 实际延迟时间（秒）


X = scipy.signal.chirp(t, 0, 1, 100);
Y = np.hstack((np.zeros(int(np.round(delay*fs))), X[:-int(np.round(delay*fs))])) + 0.5 * np.random.randn(X.size)

##>>>>>>  method 1
acf, lag = xcorr(Y, X, normed = True, detrend = True, maxlags = Y.size - 1)

time_delay = lag[np.argmax(np.abs(acf))]/fs
estimated_distance = time_delay * c / 2;

print(f'实际的目标距离为 {d} 米\n', )
print(f'估计的目标距离为 {estimated_distance} 米\n', )

##### plot
fig, axs = plt.subplots(3, 1, figsize = (12, 8), constrained_layout = True)
labelsize = 20

# x
axs[0].plot(t, X, label = 'x')
axs[0].set_title("发射信号 X(t)")
axs[0].set_xlabel('时间 (秒)',)
axs[0].set_ylabel('幅度',)

axs[1].plot(t, Y[0:len(t)], label = 'PSD',  )
axs[1].set_title("接收信号 Y(t)")
axs[1].set_xlabel('时间 (秒)')
axs[1].set_ylabel('幅度')

axs[2].plot(lag/fs, acf, label = 'PSD',  )
axs[2].set_title(r"互相关函数 $R_{XY}(\tau)$")
axs[2].set_xlabel('时间延迟 (秒)')
axs[2].set_ylabel('互相关值')

plt.show()
plt.close()

#%% < 深入浅出数字信号处理, page122 >
import numpy as np
import matplotlib.pyplot as plt

# 关闭所有图形窗口，清空工作空间
plt.close('all')

# 参数设置
c0 = 3e8  # 光速
fs = 40e6  # 采样频率
ts = 1 / fs  # 采样周期
fc = 10e6  # 雷达中频频率
tr = 1e-4  # 脉冲重复周期
M = 400
# tao = M * ts  # 脉冲宽度
D = 1500
d = D * ts  # 延时时间
R = c0 * d / 2  # 目标距离
A = 1  # 衰减系数

# 时间序列
t = np.arange(0, tr, ts)
N = len(t)

# 发射信号（中频）
rect1 = np.concatenate([np.ones(M), np.zeros(N - M)])
st = rect1 * np.cos(2 * np.pi * fc * t)

# 回波信号（中频）
rect2 = np.concatenate([np.zeros(D), np.ones(M), np.zeros(N - M - D)])
s = A * rect2 * np.cos(2 * np.pi * fc * (t - d))

# 回波噪声
v = (A / 2) * np.random.randn(N)
x = s + v  # 雷达回波


# 绘图：雷达发射信号和接收信号波形
f, axs = plt.subplots(2, 1, figsize = (12, 6))
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
axs[0].plot(st)
axs[0].set_title('Original Signal Spectrum')
axs[0].set_xlabel('Number of samples')
axs[0].set_ylabel('Transmitted signal')

# plt.subplot(1, 2, 2)
axs[1].plot(x)
axs[1].set_title('Filtered Signal Spectrum')
axs[1].set_xlabel('Number of samples')
axs[1].set_ylabel('Received signal')

plt.tight_layout()
plt.show()
plt.close()

# 最佳接收系统的输出（互相关）
y = np.correlate(x, st, mode='full')
m = np.arange(-(N - 1), N)  # 延迟样本数
d_est = m[y.argmax()] * ts* c0/2

# 绘制相关输出
f, axs = plt.subplots(1, 1, figsize = (12, 6))
axs.plot(m, y)
axs.set_xlabel('Delay Samples')
axs.set_ylabel('Correlation Output')
axs.grid(True)
plt.show()

# 输出目标距离信息
print(f"目标距离: {R:.2f} 米")
print(f"延时时间: {d*1e6:.2f} μs")
print(f"采样点数: {N}")
print(f"估计距离:{d_est} 米")













































































